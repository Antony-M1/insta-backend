# insta-backend
FastAPI, fastapi, redis


# Quick Start - Docker
```
docker compose up -d
```
For Force build add the `--build` argument in the above command

# [File Structure](https://fastapi.tiangolo.com/tutorial/bigger-applications/)

For a file structure we can follow these for a bigger applications

```
.
├── app
│   ├── __init__.py
│   ├── main.py
│   ├── dependencies.py
│   └── routers
│   │   ├── __init__.py
│   │   ├── items.py
│   │   └── users.py
│   └── internal
│       ├── __init__.py
│       └── admin.py
```
