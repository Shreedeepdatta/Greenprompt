from fastapi import APIRouter, HTTPException
from dto.energy_dto import TokenCountRequest, EnergyCalculationRequest, PromptRequest
from services.energy_calculation import TokenEnergyCalculator


energy_calculator_router = APIRouter(tags=["Energy calculation"],
                                     prefix="/energycalculation")
calculator = TokenEnergyCalculator()


@energy_calculator_router.post("/count-tokens")
async def count_tokens(request: TokenCountRequest):
    """Count tokens in the provided text."""
    try:
        token_count = calculator.count_tokens(request.text)
        return {
            "text_length": len(request.text),
            "token_count": token_count,
            "tokens_per_char": round(token_count / len(request.text), 3) if len(request.text) > 0 else 0
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error counting tokens: {str(e)}")


@energy_calculator_router.post("/")
def energy_calculation(request: EnergyCalculationRequest):
    """Calculate energy cost for a given number of tokens."""
    try:
        if request.model and request.model not in calculator.energy_costs and request.model != "default":
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' not supported. Use /models endpoint to see available models."
            )

        result = calculator.calculate_energy_cost(
            request.token_count, request.model)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error calculating energy: {str(e)}")


@energy_calculator_router.post("/calculate-energy")
async def calculate_energy(request: EnergyCalculationRequest):
    """Calculate energy cost for a given number of tokens."""
    try:
        if request.model and request.model not in calculator.energy_costs and request.model != "default":
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' not supported. Use /models endpoint to see available models."
            )

        result = calculator.calculate_energy_cost(
            request.token_count, request.model)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error calculating energy: {str(e)}")


energy_calculator_router.post("/analyze-prompt")


async def analyze_prompt(request: PromptRequest):
    """Analyze a prompt to get token count and energy cost."""
    try:
        if request.model and request.model not in calculator.energy_costs and request.model != "default":
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model}' not supported. Use /models endpoint to see available models."
            )

        result = calculator.analyze_prompt(request.prompt, request.model)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error analyzing prompt: {str(e)}")


@energy_calculator_router.post("/compare-models")
async def compare_models(request: PromptRequest):
    """Compare energy costs across different models for the same prompt."""
    try:
        result = calculator.compare_models(request.prompt)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error comparing models: {str(e)}")
