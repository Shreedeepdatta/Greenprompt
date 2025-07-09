from fastapi import FastAPI

app = FastAPI(title="Prompt Optimizer")


@app.get("/")
def readroot():
    return {"message": "welcome to prompt optimizer backend"}
