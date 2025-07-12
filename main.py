from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.optimize_router import optimize_router

app = FastAPI(title="Prompt Optimizer")
app.include_router(optimize_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Need to Specify frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def readroot():
    return {"message": "welcome to prompt optimizer backend"}
