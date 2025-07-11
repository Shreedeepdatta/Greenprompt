from fastapi import APIRouter
from services.optimize import PromptOptimizer
from dto.optimize_prompt_dto import PromptRequest

optimize_router = APIRouter(tags=["Prompt Optimize"])


@optimize_router.post("/optimize")
def optimize_prompt(prompt: PromptRequest):
    prompt_optimizer = PromptOptimizer()
    response = prompt_optimizer.analyze_prompt(prompt=prompt.text)
    return response
