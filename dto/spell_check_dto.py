from fastapi import BaseModel


class SpellCheckRequest(BaseModel):
    text: str


class SpellCheckResponse(BaseModel):
    mispelled_word: str
    correction: str
    candidates: list[str]
