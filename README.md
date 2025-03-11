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

**Folder Structure Reference**
- [Folder Structure for Scalable FastAPI Applications in Production](https://www.codersarts.com/post/structuring-your-fastapi-app-code)
- [How to Structure Your FastAPI Projects](https://medium.com/@amirm.lavasani/how-to-structure-your-fastapi-projects-0219a6600a8f)
- [What are the best practices for structuring a FastAPI project?](https://stackoverflow.com/questions/64943693/what-are-the-best-practices-for-structuring-a-fastapi-project)