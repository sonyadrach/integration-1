import logging
import os

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from . import crud, schemas
from .database import Base, engine, get_db

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("orders-api")

FORECAST_URL = os.getenv(
    "FORECAST_URL",
    "http://forecast-api:8001",
)

FORECAST_TIMEOUT = float(
    os.getenv("FORECAST_TIMEOUT", "3.0")
)
FORECAST_FEATURE_KEYS = {
    "gr_liv_area",
    "total_bsmt_sf",
    "garage_area",
    "year_built",
    "overall_qual",
    "ms_zoning",
}
# Tables are created automatically for the initial version.
# In production, replace this with Alembic migrations.
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Orders API",
    version="0.1.0",
)


@retry(
    stop=stop_after_attempt(2),
    wait=wait_fixed(0.2),
    retry=retry_if_exception_type(httpx.TimeoutException),
    reraise=True,
)
def _request_prediction(payload: dict) -> float:
    """Request a price prediction from the forecasting service.

    Only features relevant to the ML model are sent.
    """
    # фільтрування лише поля нерухомості 
    model_payload = {k: v for k, v in payload.items() if k in FORECAST_FEATURE_KEYS}

    with httpx.Client(timeout=FORECAST_TIMEOUT) as client:
        response = client.post(
            f"{FORECAST_URL}/predict",
            json=model_payload,
        )
        response.raise_for_status()
        return float(response.json()["predicted_price"])

@app.get("/health")
def health() -> dict[str, str]:
    """Return the health status of the Orders API."""
    return {"status": "ok"}


@app.post(
    "/orders",
    response_model=schemas.OrderRead,
    status_code=201,
)
def create_order(
    data: schemas.OrderCreate,
    db: Session = Depends(get_db),
):
    """Create an order and attempt to obtain a price prediction.

    If the forecasting service is unavailable, the order is still
    persisted with a null predicted price (graceful degradation).
    """
    try:
        price = _request_prediction(data.model_dump())
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        # Логує як warning без падіння обробника
        logger.warning(
            "Forecast service unavailable or returned error: %s. Proceeding with price=None.",
            exc,
        )
        price = None
    except Exception as exc:
        logger.warning("Unexpected error during price prediction: %s", exc)
        price = None

    return crud.create_order(
        db,
        data,
        price,
    )


@app.get(
    "/orders",
    response_model=list[schemas.OrderRead],
)
def list_orders(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """Return a paginated list of orders."""
    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 100.",
        )

    if offset < 0:
        raise HTTPException(
            status_code=400,
            detail="Offset must be non-negative.",
        )

    return crud.list_orders(
        db,
        limit,
        offset,
    )


@app.get(
    "/orders/{oid}",
    response_model=schemas.OrderRead,
)
def get_order(
    oid: int,
    db: Session = Depends(get_db),
):
    """Return a single order by ID."""
    order = crud.get_order(db, oid)

    if order is None:
        raise HTTPException(
            status_code=404,
            detail="Order not found.",
        )

    return order


@app.patch(
    "/orders/{oid}",
    response_model=schemas.OrderRead,
)
def update_order(
    oid: int,
    data: schemas.OrderUpdate,
    db: Session = Depends(get_db),
):
    """Update an existing order."""
    order = crud.get_order(db, oid)

    if order is None:
        raise HTTPException(
            status_code=404,
            detail="Order not found.",
        )

    return crud.update_order(
        db,
        order,
        data,
    )


@app.delete(
    "/orders/{oid}",
    status_code=204,
)
def delete_order(
    oid: int,
    db: Session = Depends(get_db),
) -> None:
    """Delete an existing order."""
    order = crud.get_order(db, oid)

    if order is None:
        raise HTTPException(
            status_code=404,
            detail="Order not found.",
        )

    crud.delete_order(db, order)