from fastapi import FastAPI
from routers.optimize_router import optimize_router

app = FastAPI(title="Prompt Optimizer")
app.include_router(optimize_router)


@app.get("/")
def readroot():
    return {"message": "welcome to prompt optimizer backend"}
