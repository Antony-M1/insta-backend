from fastapi import (
    FastAPI, Query, Path, Body, Cookie, Header,
    status, Form, File, UploadFile, HTTPException,
    Request, Depends
)
from fastapi.security import OAuth2PasswordBearer
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from sqlmodel import Field as F, SQLModel, Session, create_engine, select

from typing import Annotated, Literal, List, Any
from enum import Enum
from pydantic import (
    BaseModel, AfterValidator, Field, HttpUrl, EmailStr
)
from uuid import UUID
from datetime import datetime, time, timedelta


app = FastAPI()


# Root
@app.get("/")
async def root():
    return {'name': "Hello World"}


# Path Parameter
@app.get("/item/{item_id}")
async def get_item(item_id: int):  # Comes with the Type validation
    return {"item_id": item_id}


# Path Operation Orders are matters
@app.get("/users/me", tags=['User'])
async def read_user_me():
    return {"user_id": "The current user"}


@app.get("/users/{user_id}", tags=['User'])
async def read_user(user_id: str):  # Both are string so me should come first
    return {"user_id": user_id}


class UserType(str, Enum):
    guest = "guest"
    public = "public"
    private = "private"
    anonymous = "anonymous"


@app.get("users/type/{user_type}", tags=['User'])
async def get_model(user_type: UserType):  # Work as a select field
    if user_type is UserType.anonymous:
        return {"user_type": user_type, "message": "You can access anonymous data"}
    if user_type is UserType.private:
        return {"user_type": user_type, "message": "You are a Private User"}
    if user_type is UserType.public:
        return {"user_type": user_type, "message": "You are a Private user"}
    if user_type.value == UserType.guest.value:  # Access Through value as well
        return {"user_type": user_type, "message": "You are a Guest User"}


# Path parameters containing paths
'''
what if happen path parameter contains the `path` like this `/files/{file_path}`
but we need the exact file path like this `home/file/image/background.jbg`

We can achive through adding the `path` in the url itself
''' # noqa


@app.get("/file/{file_path:path}")
def get_file_path(file_path: str):
    return {"file_path": file_path}


fake_item_db = [{"item_name": "foo"}, {"item_name": "Baz"}, {"item_name": "Bar"}]


# Query Parameters
'''
The query is the set of key-value pairs that go after the ? in a URL, separated by & characters.
As they are part of the URL, they are "naturally" strings.
''' # noqa


@app.get("/items/", tags=["Item"])
async def read_item(skip: int = 0, limit: int = 10):
    data = fake_item_db[skip: skip+limit]
    return data

# Optional parameters
'''
Declaring the Optional query parameters
https://fastapi.tiangolo.com/tutorial/query-params/#optional-parameters
'''


@app.get("/items/{item_id}", tags=['Item'])
async def read_item_v2(item_id: str, q: str | None = None):
    if q:
        return {"item_id": item_id, "q": q}
    return {"item_id": item_id}


@app.get("/items", tags=["Item"])
async def read_item_v3(a: str | None = None, b: int | None = None):  # type converstion is done in the `b` parameter
    return {"a": a, "b": b}


# Multiple path and query parameters


@app.get("/user/{user_id}/item/{item_id}", tags=["User", "Item"])
def read_user_item(user_id: str, item_id: int, a: str | None = None, b: bool | None = None):
    return {
        "user_id": user_id,
        "item_id": item_id,
        "a": a,
        "b": b
    }

# Default Value


@app.get("/items/v2/{item_id}", tags=["Item"])
def read_user_item_v2(item_id: int, a: int = 0, b: str = "Hello Welcome"):
    return {
        "item_id": item_id,
        "a": a,
        "b": b
    }


# Request Body
'''
Client ----------------> API
        Response Body
the more common request method is `POST` others are PUT, DELETE & PATCH
'''


class Item(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None


@app.post("/items/", tags=['Item'])
async def create_item(item: Item):
    item_dict = item.model_dump()
    if item.tax is not None:
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return {"item": item_dict, "message": "Success"}


# Request body + path parameters
@app.put("/items/{item_id}", tags=['Item'])
async def update_item(item_id: int, item: Item):
    return {"item_id": item_id, "item": item}


# Request body + path + query parameters
@app.put("/items/v2/{item_id}", tags=["Item"])
async def update_item_v2(item_id: int, item: Item, skip: int, limit: int | None = None):
    return {
        "item_id": item_id,
        "skip": skip,
        "limit": limit,
        "item": item
    }


# Query Parameters and String Validations
"""
Additional declaration and validation for the Query Parameter
"""


@app.get("/qpsv/items/", tags=['Query Parameters and String Validations'])
async def read_items_qpsv(q: str | None = None):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}

    if q:
        results.update({"q": q})
    return results


# Additional validation
"""
We are going to enforce that even though q is optional, whenever it is provided, its length doesn't exceed 50 characters.
""" # noqa


def after_validate(q: str):
    print(f"After Validate {q}")


@app.get("/qpsv/v2/items/", tags=['Query Parameters and String Validations'])
async def read_items_qpsv_v2(
    q: Annotated[
        str | None,
        Query(
            max_length=50,
            alias="item-query",
            title="Query String",
            deprecated=True,
            # include_in_schema=False,
            description="Query string for the items to search in the database that have a good match"), AfterValidator(after_validate)] = None # noqa
            ):
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}

    if q:
        results.update({"q": q})
    return results


# Path Parameters and Numeric Validations = ppnv
'''
In the same way what we done for the Query Parameters and String Validation

`Path` Instead of `Query`
'''

# Even changing the order of the parameters it will detect


@app.get("/ppnv/v2/items/{item_id}", tags=["Path Parameters and Numeric Validations"])
async def read_items_ppnv_v2(
    q: str, item_id: Annotated[int, Path(title="The ID of the item to get")]
):
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    return results


@app.get("/ppnv/items/{item_id}", tags=["Path Parameters and Numeric Validations"])
async def read_items_ppnv(
    item_id: Annotated[int, Path(
            title="The Id of the Item to get",
            ge=1,
            le=1000
        )],
    q: Annotated[str | None, Query(alias="item-query")] = None
):
    results = {"item_id": item_id}

    if q:
        results.update({"q": q})
    return results


# Query Parameter Models
'''
We can use the Pydantic model to declare the list of 
Query Parameter.
'''


class FilterParams(BaseModel):
    model_config = {"extra": "forbid"}

    limit: int = Field(100, gt=0, le=100)
    offset: int = Field(0, ge=0)
    order_by: Literal["created_at", "updated_at"] = "created_at"
    tags: list[str] = []


@app.get("/qpm/items/", tags=['Query Parameter Models'])
async def read_items_qpm(
    filter_query: Annotated[FilterParams, Query()]
):
    return filter_query


# Body - Multiple Parameters
'''
We seen the Query and Path let see some advance use case
for the Request Body Declaration
'''


class User(BaseModel):
    username: str
    full_name: str | None = None


@app.put("/bmp/items/{item_id}", tags=['Body - Multiple Parameters'])
async def update_item_bmp(
    item_id: Annotated[int, Path(description="Item Id")],
    q: Annotated[str | None, Query(alias='item-query')],
    item: Item,
    user: User  # Multiple Body Parameter
):
    return {
        "item_id": item_id,
        "q": q,
        "item": item,
        "user": user,
    }

# Singular values in body


@app.put("/svb/items/{item_id}", tags=['Body - Multiple Parameters'])
async def update_item_svb(
    item_id: int,
    item: Item,
    user: User,
    importance: Annotated[int, Body(gt=10)],
    q: str | None = None
):
    return {
        "item_id": item_id,
        "item": item,
        "user": user,
        "important": importance,
        "q": q
    }


# Body - Fields
'''
The same way you can declare additional validation and metadata in
path operation function parameters with Query, Path and Body,
you can declare validation and metadata inside of Pydantic models
using Pydantic's Field
'''


class Item2(BaseModel):
    id: int
    name: str
    description: str | None = Field(title="Description", default=None)
    price: int = Field(gte=100, default=100)
    tax_percentage: int | None = Field(gt=10, le=100, title="Tax Percentage rage 10 to 100")


@app.put("/bf/items/{item_id}", tags=['Body - Fields'])
async def update_item_bf(
    item_id: Annotated[int, Path(description="Item ID")],
    item: Annotated[Item2, Body(embed=True)]
):
    return {
        "item_id": item_id,
        "item": item
    }


# Body - Nested Models
'''
We can achive the Nested Body models with the help of pydantic
'''


class Image(BaseModel):
    url: HttpUrl
    name: str


class ItemBNM(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None
    tags: set[str] = set()
    image: list[Image] | None = None


@app.put("/bnm/items/{item_id}", tags=['Body - Nested Models'])
async def update_item_bnm(
    item_id: int,
    item: ItemBNM
):
    return {
        "item_id": item_id,
        "item": item
    }


class Offer(BaseModel):
    name: str
    description: str | None = None
    price: float
    items: list[ItemBNM]


@app.post("/bnm/offers/", tags=['Body - Nested Models'])
async def create_offer_bnm(offer: Offer):
    return {"offer": offer}


@app.post("/bnm/images/multiple/", tags=['Body - Nested Models'])
async def create_multiple_images(images: list[Image]):
    return images


@app.post("/bnm/index-weights/", tags=['Body - Nested Models'])
async def create_index_weights(weights: dict[str, float]):
    return weights


# Declare Request Example Data

class Item(BaseModel):
    name: str = Field(examples=["Foo"])
    description: str | None = Field(default=None, examples=["A very nice Item"])
    price: float = Field(examples=[35.4])
    tax: float | None = Field(default=None, examples=[3.2])


@app.put("/dred/items/{item_id}", tags=['Declare Request Example Data'])
async def update_item_dred(item_id: int, item: Item):
    results = {"item_id": item_id, "item": item}
    return results


# Extra Data Types


@app.put("/edt/items/{item_id}", tags=['Extra Data Types'])
async def read_items_edt(
    item_id: UUID,
    start_datetime: Annotated[datetime, Body()],
    end_datetime: Annotated[datetime, Body()],
    process_after: Annotated[timedelta, Body()],
    repeat_at: Annotated[time | None, Body()] = None
):
    start_process = start_datetime + process_after
    duration = end_datetime - start_process
    return {
        "item_id": item_id,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "process_after": process_after,
        "repeat_at": repeat_at,
        "start_process": start_process,
        "duration": duration,
    }


# Cookie Parameters
'''
You can define cookie parameters the same way you define `Query` and `Path`
Parameters
'''


class Cookies(BaseModel):
    session_id: str
    fatebook_tracker: str | None = None
    googall_tracker: str | None = None


@app.get("/cpm/items/", tags=['Cookie'])
async def read_items_cpm(cookies: Annotated[Cookies, Cookie()]):
    return cookies


@app.get("/cp/items/", tags=["Cookie"])
async def read_items(
    ads_id: Annotated[str | None, Cookie()] = None,
    ads_content: Annotated[str | None, Cookie()] = None
):
    return {"ads_id": ads_id, "ads_content": ads_content}


# Header Parameters
'''
You can define Header parameters the same way you define
`Query`, `Path` and `Cookie` parameters.

Note: Cookie and Header is not properly working in the swagger
'''


@app.get("/hp/items/", tags=['Header Parameters'])
async def read_items_hp(
    user_agent: Annotated[str | None, Header()] = None,
    strange_header: Annotated[str | None, Header(convert_underscores=False)] = None,
    x_token: Annotated[list[str] | None, Header()] = None
):
    return {"User-Agent": user_agent, "strange_header": strange_header, 'x-token': x_token}


# Response Model - Return Type
'''
Declare the Return type with help of Annotation & Pydantic
'''


class ItemRMRT(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None
    tags: list[str] = []


@app.post("/rmrt/items", tags=['Response Model - Return Type'], response_model=ItemRMRT)
async def create_item_rmrt(item: ItemRMRT) -> Any:
    return item


@app.get(
        "/rmrt/items",
        tags=['Response Model - Return Type'],
        response_model=list[ItemRMRT],
        response_model_exclude_unset=True  # Exclude the unset values
    )
async def read_items_rmrt() -> Any:
    return [
        ItemRMRT(name="Portal Gun", price=42.0),
        ItemRMRT(name="plumbus", price=32.0)
    ]


class UserIn_RMRT(BaseModel):
    username: str
    password: str
    email: EmailStr
    full_name: str | None = None


class UserOut_RMRT(BaseModel):
    username: str
    email: EmailStr
    full_name: str | None = None


@app.post("/rmrt/user/", tags=['Response Model - Return Type'], response_model=UserOut_RMRT)
async def create_user(user: UserIn_RMRT) -> Any:
    return user


# Return Type and Data Filtering = RTDF
'''In the Previous example both the Model is differnt'''


class BaseUser_RTDF(BaseModel):
    username: str
    email: EmailStr
    full_name: str | None = None


class UserIn_RTDF(BaseUser_RTDF):  # Inherit the Above model
    password: str


@app.post("/rtdf/user", tags=['Return Type and Data Filtering'])
async def create_user_rtdf(user: UserIn_RTDF) -> BaseUser_RTDF:
    return user


# Extra Models
'''
Its common to have more than one model in a API like

- Input Model: Contians the Username & Password
- Output Model: It won't contians the Password
- DB Model: Store the password in Hash format
'''


class UserOut_EM(BaseModel):
    username: str
    email: str
    full_name: str


class UserIn_EM(UserOut_EM):
    password: str


class UserInDB(UserOut_EM):
    hashed_password: str


def fake_password_hasher(raw_password: str):
    return "supersecret" + raw_password


def fake_save_user(user_in: UserIn_EM):
    hashed_password = fake_password_hasher(user_in.password)
    user_in_db = UserInDB(**user_in.model_dump(), hashed_password=hashed_password)
    print(f"User saved! ..not really {user_in_db.model_dump()}")
    return user_in_db


@app.post("/em/user/", response_model=UserOut_EM, tags=['Extra Model'])
async def create_user_em(user_in: UserIn_EM):
    user_saved = fake_save_user(user_in)
    return user_saved


# Response Status Code

@app.post("/rsc/items/", status_code=status.HTTP_201_CREATED, tags=['Response Status Code'])
async def create_item_rsc(name: str):
    return {"name": name}


# Form Data

@app.post("/fm/login/", tags=['Form Data'])
async def login_fm(username: Annotated[str, Form()], password: Annotated[str, Form()]):
    return {"username": username}


# Request File

@app.post("/rf/files", tags=['Request File'])
async def create_file_rf(file: Annotated[bytes, File()]):
    return {"file_size": f"{round(len(file)/1024, 1)} KB"}


@app.post("/rf/uploadfile", tags=['Request File'])
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}


# Handling Errors


items = {
    'foo': "I'm foo",
    "Bar": "I'm Bar"
}


@app.get("/he/items/{item_id}", tags=['Handling Errors'])
async def get_items_he(item_id: Annotated[str, Path()]):
    if item_id not in items:
        raise HTTPException(
                status_code=404,
                detail="Item Not Found",
                headers={"X-Error": "There goes my error"}
            )
    return {"item": items[item_id]}


class UnicornException(Exception):
    def __init__(self, name: str):
        self.name = name


@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    return JSONResponse(
        status_code=418,
        content={"message": f"Oops! {exc.name} did something. There goes a rainbow..."},
    )


@app.get("/he/unicorns/{name}", tags=['Handling Errors'])
async def get_unicorns_he(name: Annotated[str, Path()]):
    if name == 'yolo':
        raise UnicornException(name=name)
    return {"unicorn_name": name}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=jsonable_encoder({"detail": exc.errors(), "body": exc.body}),
    )


@app.get("/he/v2/items/{item_id}", tags=['Handling Errors'])
async def read_item_he_v2(item_id: int):
    if item_id == 3:
        raise HTTPException(status_code=418, detail="Nope! I don't like 3.")
    return {"item_id": item_id}


class Item(BaseModel):
    title: str
    size: int


@app.post("/he/v3/items/", tags=['Handling Errors'])
async def create_item_he_v3(item: Item):
    return item


# JSON Compatible Encoder
'''
There are some cases we have to convert the Pydantic Model into the
`jsonable_encoder` 
'''

fake_db = {}


@app.put("/jce/items/{id}", tags=["JSON Compatible Encoder"])
def update_item_jec(id: str, item: Item):
    json_compatible_item_data = jsonable_encoder(item)
    fake_db[id] = json_compatible_item_data
    return fake_db


# Dependency Injection
'''
"Dependency Injection" means, in programming, that there is a way for
your code (in this case, your path operation functions) to declare
things that it requires to work and use: "dependencies"
'''


async def common_parameters(
  q: str | None = None,
  skip: int = 0,
  limit: int = 0
):
    return {"q": q, "skip": skip, "limit": limit}


CommonsDep = Annotated[dict, Depends(common_parameters)]


@app.get("/di/items", tags=['Dependency'])
async def read_items_di(commons: CommonsDep):
    return commons


@app.get("/di/users", tags=['Dependency'])
async def read_users_di(
    commons: CommonsDep
):
    return commons


# Classes as Dependencies


class CommonQueryParams:
    def __init__(
                    self,
                    q: str | None = None,
                    skip: int = 0,
                    limit: int = 0
                ):
        self.q = q
        self.skip = skip
        self.limit = limit


fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


@app.post("/cad/items/", tags=['Classes as Dependencies'])
async def get_items_cad(
    commons: Annotated[CommonQueryParams, Depends()]
):
    response: dict = {}
    if commons.q:
        response.update(q=commons.q)
    items = fake_item_db[commons.skip: commons.skip + commons.limit]
    response.update(items=items)
    return response


# Sub-dependencies

def query_extractor(q: str | None = None):
    return q


def query_or_cookie_extractor(
    q: Annotated[str, Depends(query_extractor)],
    last_query: Annotated[str | None, Cookie()] = None,
):
    if not q:
        return last_query
    return q


@app.get("/sd/items/", tags=['Sub-dependencies'])
async def read_query_sd(
    query_or_default: Annotated[str, Depends(query_or_cookie_extractor)],
):
    return {"q_or_cookie": query_or_default}


# Dependencies in path operation decorators

async def verify_token(x_token: Annotated[str, Header()]):
    if x_token != "fake-super-secret-token":
        raise HTTPException(status_code=400, detail="X-Token header invalid")


async def verify_key(x_key: Annotated[str, Header()]):
    if x_key != "fake-super-secret-key":
        raise HTTPException(status_code=400, detail="X-Key header invalid")
    return x_key


@app.get("/dpod/items/", dependencies=[Depends(verify_token), Depends(verify_key)], tags=['Dependencies in path operation decorators']) # noqa
async def read_items_dpod():
    return [{"item": "Foo"}, {"item": "Bar"}]


# Global Dependencies

'''py
app = FastAPI(dependencies=[Depends(verify_token), Depends(verify_key)])
'''

# Dependencies with yield
'''
FastAPI supports dependencies that do some extra steps after finishing.

To do this, use yield instead of return, and write the extra steps (code) after.

Use in the Database Dependency
'''


# Security

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@app.get("/security/items/", tags=["Security"])
async def read_items_security(token: Annotated[str, Depends(oauth2_scheme)]):
    return {"token": token}


#  SQL (Relational) Databases


class Hero(SQLModel, table=True):
    id: int | None = F(default=None, primary_key=True)
    name: str = F(index=True)
    age: int | None = F(default=None, index=True)
    secret_name: str


sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

connect_args = {"check_same_thread": False}
engine = create_engine(sqlite_url, connect_args=connect_args)


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


@app.on_event("startup")  # on_event Depricated
def on_startup():
    create_db_and_tables()


@app.post("/sqldb/heroes/", tags=['SQL DB'])
def create_hero(hero: Hero, session: SessionDep) -> Hero:
    session.add(hero)
    session.commit()
    session.refresh(hero)
    return hero


@app.get("/sqldb/heroes/", tags=['SQL DB'])
def read_heroes(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
) -> list[Hero]:
    heroes = session.exec(select(Hero).offset(offset).limit(limit)).all()
    return heroes


@app.get("/sqldb/heroes/{hero_id}", tags=['SQL DB'])
def read_hero(hero_id: int, session: SessionDep) -> Hero:
    hero = session.get(Hero, hero_id)
    if not hero:
        raise HTTPException(status_code=404, detail="Hero not found")
    return hero


@app.delete("/sqldb/heroes/{hero_id}", tags=['SQL DB'])
def delete_hero(hero_id: int, session: SessionDep):
    hero = session.get(Hero, hero_id)
    if not hero:
        raise HTTPException(status_code=404, detail="Hero not found")
    session.delete(hero)
    session.commit()
    return {"ok": True}
