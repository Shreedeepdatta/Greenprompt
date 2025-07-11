from fastapi import APIRouter
from dto.optimize_prompt_dto import PromptRequest

optimize_router = APIRouter(tags="Prompt Optimize")


@optimize_router.post("/optimize")
def optimize_prompt()
