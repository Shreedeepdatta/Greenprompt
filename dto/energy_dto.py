from typing import Dict, Optional
from pydantic import BaseModel, Field


class TokenCountRequest(BaseModel):
    """
    Request model for token counting.
    """
    text: str = Field(...,
                      description="The text to count tokens for", min_length=1)


class EnergyCalculationRequest(BaseModel):
    """
    Request model for energy cost calculation.
    """
    token_count: int = Field(..., description="Number of tokens", ge=1)
    model: Optional[str] = Field(
        None, description="Model name for energy calculation")


class PromptRequest(BaseModel):
    """
    Request model for prompt analysis.
    """
    prompt: str = Field(...,
                        description="The prompt text to analyze", min_length=1)
    model: Optional[str] = Field(
        None, description="Model name for energy calculation")


class EnergyResponse(BaseModel):
    """
    Response model for energy cost calculations.
    """
    tokens: int
    model: str
    energy_wh: float
    energy_mwh: float
    co2_grams: float
    cost_per_1k_tokens_wh: float

    class Config:
        json_encoders = {
            float: lambda v: round(v, 6)
        }


class PromptAnalysisResponse(BaseModel):
    """
    Response model for complete prompt analysis.
    """
    prompt_length: int
    token_count: int
    tokens_per_char: float
    energy_cost: EnergyResponse


class ModelComparisonResponse(BaseModel):
    """
    Response model for model comparison analysis.
    """
    prompt_length: int
    token_count: int
    model_comparison: Dict[str, EnergyResponse]
