"""Manifest as an app service."""

from typing import Any, Dict, cast

from fastapi import APIRouter, FastAPI, HTTPException

from manifest import Manifest
from manifest.response import Response as ManifestResponse
from web_app import schemas

app = FastAPI()
api_router = APIRouter()


@app.get("/")
async def root() -> Dict:
    """Root endpoint."""
    return {"message": "Hello to the Manifest App"}


@api_router.post("/prompt/", status_code=201, response_model=schemas.ManifestResponse)
def prompt_manifest(*, manifest_in: schemas.ManifestCreate) -> Dict:
    """Prompt a manifest session and query."""
    manifest = Manifest(
        client_name=manifest_in.client_name,
        client_connection=manifest_in.client_connection,
        engine=manifest_in.engine,
        cache_name=manifest_in.cache_name,
        cache_connection=manifest_in.cache_connection,
    )
    manifest_prompt_args: Dict[str, Any] = {
        "n": manifest_in.n,
        "max_tokens": manifest_in.max_tokens,
    }
    if manifest_in.temperature:
        manifest_prompt_args["temperature"] = manifest_in.temperature
    if manifest_in.top_k:
        manifest_prompt_args["top_k"] = manifest_in.top_k
    if manifest_in.top_p:
        manifest_prompt_args["top_p"] = manifest_in.top_p

    try:
        response = manifest.run(
            prompt=manifest_in.prompt, return_response=True, **manifest_prompt_args
        )
        response = cast(ManifestResponse, response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "response": response.get_response(),
        "cached": response.is_cached(),
        "request_params": response.get_request_obj(),
    }


app.include_router(api_router)
