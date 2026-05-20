from fastapi import FastAPI
from titiler.core import factory as core_factory
from titiler.xarray import extensions as xr_extensions
from titiler.xarray import factory as xr_factory
from titiler.xarray import io as xr_io
import xarray as xr

from fastapi import HTTPException, Query

from starlette.middleware.cors import CORSMiddleware

import ml_vs30_model as vs30

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # Allows all origins (for development - be more specific in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def tif_resolve_url(url: str = Query(...)) -> str:
    resolved = vs30.constants.INPUT_VAR_TO_FFP_MAP.get(url)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: '{url}'")
    return resolved

def xr_resolve_url(url: str = Query(...)) -> str:
    ffp = vs30.constants.BASE_DATA_DIR / "grids" / url / "input_grid.nc"
    if not ffp.exists():
        raise HTTPException(
            status_code=404, detail=f"Dataset not found: '{url}'"
        )
    return str(ffp)

def xr_model_resolve_url(url: str = Query(...)) -> str:
    ffp = vs30.constants.BASE_DATA_DIR / "results/ind_results" / url / "nz_vs30_results.nc"
    if not ffp.exists():
        raise HTTPException(
            status_code=404, detail=f"Dataset not found: '{url}'"
        )
    return str(ffp)

# Cloud-Optimized GeoTIFFs
cog = core_factory.TilerFactory(path_dependency=tif_resolve_url)
app.include_router(cog.router, tags=["Cloud Optimized GeoTIFF"])

# Xarray
xr_inputs = xr_factory.TilerFactory(
    reader=xr_io.FsReader,
    router_prefix="/xr",
    path_dependency=xr_resolve_url,
    extensions=[
        xr_extensions.VariablesExtension(dataset_opener=xr.open_dataset),
    ],
)
app.include_router(xr_inputs.router, prefix="/xr", tags=["Xarray"])

xr_model = xr_factory.TilerFactory(
    reader=xr_io.FsReader,
    router_prefix="/model",
    path_dependency=xr_model_resolve_url,
    extensions=[
        xr_extensions.VariablesExtension(dataset_opener=xr.open_dataset),
    ],
)
app.include_router(xr_model.router, prefix="/model", tags=["Xarray Model"])

