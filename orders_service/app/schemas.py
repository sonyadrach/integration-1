from datetime import datetime
from pydantic import BaseModel, ConfigDict


class OrderBase(BaseModel):
    client_name: str
    gr_liv_area: float
    total_bsmt_sf: float
    garage_area: float
    year_built: int
    overall_qual: int
    ms_zoning: str


class OrderCreate(OrderBase):
    pass


class OrderUpdate(BaseModel):
    client_name: str | None = None
    gr_liv_area: float | None = None
    total_bsmt_sf: float | None = None
    garage_area: float | None = None
    year_built: int | None = None
    overall_qual: int | None = None
    ms_zoning: str | None = None
    predicted_price: float | None = None


class OrderRead(OrderBase):
    id: int
    predicted_price: float | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

