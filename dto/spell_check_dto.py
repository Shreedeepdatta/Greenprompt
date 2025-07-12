from fastapi import BaseModel
from typing import List, Optional


class SpellCheckRequest(BaseModel):
    text: str


class MisspelledWord(BaseModel):
    misspelledWord: str
    suggestions: List[str]


class SpellCheckData(BaseModel):
    originalSentence: str
    totalWords: int
    misspelledCount: int
    misspelledWords: List[MisspelledWord]


class SpellCheckResponse(BaseModel):
    status: str
    message: str
    data: Optional[SpellCheckData] = None
