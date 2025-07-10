from fastapi import APIRouter, HTTPException

from services import spellCheck

spellcheck = APIRouter(tags=["Spell Check"])


@spellcheck.post("/spell-check")
async def spell_check(text: str):
    result = await spellCheck.check_spelling(text)
    if result is None:
        raise HTTPException(
            status_code=500, detail="Spell check service failed")
    return {"corrected_text": result}
