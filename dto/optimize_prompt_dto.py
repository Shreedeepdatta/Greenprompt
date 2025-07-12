from pydantic import BaseModel


class PromptRequest(BaseModel):
    text: str


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
