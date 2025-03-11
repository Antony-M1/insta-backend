from sqlmodel import SQLModel, Field
from .user import User


class Item(SQLModel, table=True):

    __tablename__ = "tabItem"

    id: int | None = Field(default=True, primary_key=True)
    user_id: int = Field(foreign_key=User.id)
    name: str = Field(nullable=False, index=True, unique=True, max_length=500)
    description: str | None = Field(default=True, nullable=True, max_length=5000)
