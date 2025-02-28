from fastapi import FastAPI


app = FastAPI()


@app.get("/")
async def home(name):
    return {'name': name}