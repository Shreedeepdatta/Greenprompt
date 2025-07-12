from fastapi import APIRouter
from services.optimize import PromptOptimizer
from services.energy_calculation import TokenEnergyCalculator
from dto.optimize_prompt_dto import PromptRequest, EnergySavedResponse

optimize_router = APIRouter(tags=["Prompt Optimize"])


@optimize_router.post("/optimize")
def optimize_prompt(prompt: PromptRequest):
    energy_calculator = TokenEnergyCalculator()
    original_prompt_energy = energy_calculator.analyze_prompt(prompt=prompt.text, model=prompt.model_name)
    prompt_optimizer = PromptOptimizer()
    response = prompt_optimizer.analyze_prompt(prompt=prompt.text)
    balanced_prompt_energy = energy_calculator.analyze_prompt(prompt=response.balanced, model=prompt.model_name)
    aggresive_prompt_energy = energy_calculator.analyze_prompt(prompt=response.aggressive, model=prompt.model_name)
    conservative_prompt_energy = energy_calculator.analyze_prompt(prompt=response.conservative, model=prompt.model_name)
    energy_saved_balanced=original_prompt_energy.energy_mwh-balanced_prompt_energy.energy_mwh
    energy_saved_aggresive=original_prompt_energy.energy_mwh-aggresive_prompt_energy.energy_mwh
    energy_saved_conservative = original_prompt_energy.energy_mwh-conservative_prompt_energy.energy_mwh
    response_energy = EnergySavedResponse(
        original=response.original,
        conservative=response.conservative,
        aggressive=response.aggressive,
        balanced=response.balanced,
        removed_clauses=response.removed_clauses,
        text_after_clause_removal=response.text_after_clause_removal,
        pos_analysis=response.pos_analysis,
        stopwords_found=response.stopwords_found,
        important_words=response.important_words,
        energy_saved_balanced=energy_saved_balanced,
        energy_saved_aggressive=energy_saved_aggresive,
        energy_saved_conservative=energy_saved_conservative,
        original_energy=original_prompt_energy.energy_mwh,
        balanced_energy=balanced_prompt_energy.energy_mwh,
        aggressive_energy=aggresive_prompt_energy.energy_mwh,
        conservative_energy=conservative_prompt_energy.energy_mwh
    )
    return response_energy
