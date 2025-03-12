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


# Migration

### Step 1
Ensure you have installed the `alembic` package

```
pip install alembic
```

### Step 2
Initialize the alembic, In the root of your project
```
alembic init alembic
```

### Step 3
Make Migrtion
```
alembic revision --autogenerate -m "YOUR MESSAGE"
```

### Step 4
Apply migration
```
alembic upgrde head
```

### Other Commands
<table data-start="2711" data-end="3363"><thead data-start="2711" data-end="2803"><tr data-start="2711" data-end="2803"><th data-start="2711" data-end="2755">Command</th><th data-start="2755" data-end="2803">Description</th></tr></thead><tbody data-start="2897" data-end="3363"><tr data-start="2897" data-end="2990"><td><code data-start="2899" data-end="2921">alembic init alembic</code></td><td>Initialize Alembic in the project</td></tr><tr data-start="2991" data-end="3084"><td><code data-start="2993" data-end="3035">alembic revision --autogenerate -m "msg"</code></td><td>Auto-generate a new migration with a message</td></tr><tr data-start="3085" data-end="3177"><td><code data-start="3087" data-end="3109">alembic upgrade head</code></td><td>Apply the latest migration</td></tr><tr data-start="3178" data-end="3270"><td><code data-start="3180" data-end="3202">alembic downgrade -1</code></td><td>Rollback the last migration</td></tr><tr data-start="3271" data-end="3363"><td><code data-start="3273" data-end="3290">alembic current</code></td><td>Show the current migration state</td></tr></tbody></table>