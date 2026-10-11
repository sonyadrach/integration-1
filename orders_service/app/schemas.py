from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class OrderBase(BaseModel):
    client_name: str = Field(..., min_length=2, max_length=120)
    gr_liv_area: float = Field(..., gt=0, le=10000)
    total_bsmt_sf: float = Field(..., ge=0, le=10000)
    garage_area: float = Field(..., ge=0, le=2000)
    year_built: int = Field(..., ge=1872, le=2026)
    overall_qual: int = Field(..., ge=1, le=10)
    ms_zoning: str = Field(..., min_length=1, max_length=10)


class OrderCreate(OrderBase):
    pass


class OrderUpdate(BaseModel):
    client_name: str | None = Field(None, min_length=2, max_length=120)
    gr_liv_area: float | None = Field(None, gt=0, le=10000)
    total_bsmt_sf: float | None = Field(None, ge=0, le=10000)
    garage_area: float | None = Field(None, ge=0, le=2000)
    year_built: int | None = Field(None, ge=1872, le=2026)
    overall_qual: int | None = Field(None, ge=1, le=10)
    ms_zoning: str | None = Field(None, min_length=1, max_length=10)


class OrderRead(OrderBase):
    id: int
    predicted_price: float | None = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

