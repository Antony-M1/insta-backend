from datetime import datetime
from sqlmodel import SQLModel, Field
from pydantic import EmailStr


class User(SQLModel, table=True):

    __tablename__ = "tabUser"

    id: int = Field(primary_key=True, nullable=False)
    email: EmailStr = Field(index=True, max_length=150)
    username: str = Field(nullable=False, max_length=50)
    password: str = Field(nullable=False, max_length=50)
    first_name: str = Field(nullable=False, max_length=150)
    last_name: str | None = Field(default=None, max_length=150)
    date_of_birth: datetime = Field(nullable=False)
    bio: str | None = Field(nullable=True, default=None, max_length=1000)
