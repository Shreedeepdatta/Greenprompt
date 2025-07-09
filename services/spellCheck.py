from spellchecker import SpellChecker
import re
import json

def spell_check_api(sentence):
    try:
        # Validate input
        if not sentence or not isinstance(sentence, str):
            return {
                "status": "error",
                "message": "Invalid input: sentence must be a non-empty string",
                "data": None
            }
        
        spell = SpellChecker()
        
        # Clean the sentence and split into words
        # Remove punctuation and convert to lowercase
        words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
        
        if not words:
            return {
                "status": "error",
                "message": "No valid words found in the sentence",
                "data": None
            }
        
        # Find misspelled words
        misspelled = spell.unknown(words)
        
        result = []
        
        for word in misspelled:
            # Get the best correction first
            best_correction = spell.correction(word)
            
            # Get all possible candidates
            candidates = spell.candidates(word)
            
            suggestions = []
            
            if best_correction:
                # Add the best correction first
                suggestions.append(best_correction)
            
            if candidates:
                # Add other candidates, excluding the best correction to avoid duplicates
                other_candidates = [c for c in candidates if c != best_correction]
                # Add up to 4 more suggestions (total of 5)
                suggestions.extend(other_candidates[:4])
            
            result.append({
                "misspelledWord": word,
                "suggestions": suggestions
            })
        
        return {
            "status": "success",
            "message": f"Found {len(misspelled)} misspelled word(s)" if misspelled else "No misspelled words found",
            "data": {
                "originalSentence": sentence,
                "totalWords": len(words),
                "misspelledCount": len(misspelled),
                "misspelledWords": result
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "data": None
        }
    

# {
#   "status": "success",
#   "message": "Found 1 misspelled word(s)",
#   "data": {
#     "originalSentence": "i Love dugs",
#     "totalWords": 3,
#     "misspelledCount": 1,
#     "misspelledWords": [
#       {
#         "misspelledWord": "dugs",
#         "suggestions": ["dogs", "digs", "bugs", "hugs", "pugs"]
#       }
#     ]
#   }
# }

# {
#   "status": "error",
#   "message": "Invalid input: sentence must be a non-empty string",
#   "data": null
# }