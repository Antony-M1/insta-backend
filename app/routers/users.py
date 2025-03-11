from fastapi import APIRouter


router = APIRouter()


@router.get("/users/", tags=['users'])
async def read_users():
    return [{"username": "Antony-M1"}, {"username": "Antony-M2"}]


@router.get("/user/me", tags=['users'])
async def read_user_me():
    return {"username": "Current User"}


@router.get("/users/{username}", tags=["users"])
async def read_user(username: str):
    return {"username": username}
