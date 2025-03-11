from dotenv import load_dotenv
load_dotenv(override=True)
import os
from fastapi import Depends, FastAPI
from sqlmodel import create_engine, SQLModel
from .dependencies import get_query_token, get_token_header
from .internal import admin
from .routers import items, users
# from .models import Item, User


database_url: str | None = os.getenv("DATABASE_URL")
engine = create_engine(database_url)

app = FastAPI(dependencies=[Depends(get_query_token)])


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


@app.on_event("startup")
def on_startup():
    from app import models  # noqa: F401
    create_db_and_tables()


app.include_router(users.router)
app.include_router(items.router)
app.include_router(
    admin.router,
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(get_token_header)],
    responses={418: {"description": "I'm a teapot"}},
)


@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}
