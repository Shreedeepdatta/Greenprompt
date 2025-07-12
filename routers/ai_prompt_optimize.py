from fastapi import APIRouter, Depends, HTTPException
from dto.ai_optimize_dto import HealthResponse, OptimizedPromptResponse, PromptRequest, KeywordResponse, SummaryResponse
from services.aiOptimise import PromptOptimizer
ai_optimize_router = APIRouter(
    prefix="/ai_prompt-optimizer",
    tags=["AI Prompt Optimization"],
)
optimizer = None


def get_optimizer() -> PromptOptimizer:
    """Dependency to get the optimizer instance."""
    global optimizer
    if optimizer is None:
        optimizer = PromptOptimizer()
    return optimizer

@ai_optimize_router.get("/health", response_model=HealthResponse)
async def health_check(opt: PromptOptimizer = Depends(get_optimizer)):
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if opt.models_loaded else "unhealthy",
        models_loaded=opt.models_loaded,
        version="1.0.0"
    )


@ai_optimize_router.post("/optimize", response_model=OptimizedPromptResponse)
async def optimize_prompt(
    request: PromptRequest,
    opt: PromptOptimizer = Depends(get_optimizer)
):
    """
    Optimize a prompt using both summarization and keyword extraction.

    - **text**: The input text to optimize
    - **max_length**: Maximum length for summarization (20-200)
    - **min_length**: Minimum length for summarization (10-100)
    - **top_keywords**: Number of keywords to extract (1-20)
    """
    if not opt.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    return opt.optimize_prompt(request)


@ai_optimize_router.post("/summarize", response_model=SummaryResponse)
async def summarize_text(
    request: PromptRequest,
    opt: PromptOptimizer = Depends(get_optimizer)
):
    """
    Summarize text using DistilBART.

    - **text**: The input text to summarize
    - **max_length**: Maximum length for summary
    - **min_length**: Minimum length for summary
    """
    if not opt.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    return opt.summarize_prompt(request.text, request.max_length,
                                request.min_length)


@ai_optimize_router.post("/keywords", response_model=KeywordResponse)
async def extract_keywords(
    request: PromptRequest,
    opt: PromptOptimizer = Depends(get_optimizer)
):
    """
    Extract keywords using KeyBERT with DistilBERT.

    - **text**: The input text to extract keywords from
    - **top_keywords**: Number of top keywords to extract
    """
    if not opt.models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    return opt.extract_keywords(request.text, request.top_keywords)
