from fastapi import FastAPI
from titiler.core.factory import TilerFactory
from fastapi import HTTPException, Query

from starlette.middleware.cors import CORSMiddleware

import ml_vs30_model as vs30

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for development - be more specific in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def resolve_url(url: str = Query(...)) -> str:
    resolved = vs30.constants.INPUT_VAR_TO_FFP_MAP.get(url)
    if not resolved:
        raise HTTPException(status_code=404, detail=f"Unknown dataset: '{url}'")
    return resolved

# Create a TilerFactory for Cloud-Optimized GeoTIFFs
cog = TilerFactory(path_dependency=resolve_url)

# Register all the COG endpoints automatically
app.include_router(cog.router, tags=["Cloud Optimized GeoTIFF"])
