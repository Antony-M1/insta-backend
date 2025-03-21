from fastapi import APIRouter, Depends, HTTPException, Path
from ..dependencies import get_token_header
from typing import Annotated


router = APIRouter(
    prefix="/items",
    tags=["items"],
    dependencies=[Depends(get_token_header)],
    responses={404: {"description": "Not Found"}}
)


fake_items_db = {"plumbus": {"name": "Plumbus"}, "gun": {"name": "Portal Gun"}}


@router.get("/")
async def read_items():
    return fake_items_db


@router.get("/{item_id}")
async def read_item(item_id: Annotated[str, Path()]):
    if item_id not in fake_items_db:
        raise HTTPException(status_code=404, detail="Item Not Found")
    return fake_items_db[item_id]


@router.put(
    "/{item_id}",
    tags=["custom"],
    responses={403: {"description": "Operation Forbidden"}}
)
async def update_item(item_id: Annotated[str, Path]):
    if item_id != "plumbus":
        raise HTTPException(
            status_code=403, detail="You only update the item: plumbus"
        )
    return {"item_id": item_id, "name": "The great Plumbus"}
