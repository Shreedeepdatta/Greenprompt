from pydantic import BaseModel


class PromptRequest(BaseModel):
    text: str
    model_name: str


class PromptResponse(BaseModel):
    original: str
    conservative: str
    aggressive: str
    balanced: str
    removed_clauses: str
    text_after_clause_removal: str
    pos_analysis: str
    stopwords_found: str
    important_words: str

class EnergySavedResponse(PromptResponse):
    energy_saved_balanced: float
    energy_saved_aggressive: float
    energy_saved_conservative: float
    original_energy: float
    balanced_energy: float
    aggressive_energy: float
    conservative_energy: float
