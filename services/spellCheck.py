from spellchecker import SpellChecker
import re


def find_misspelled_words(sentence):
    """
    Find misspelled words in a sentence and suggest corrections
    """
    spell = SpellChecker()

    # Clean the sentence and split into words
    # Remove punctuation and convert to lowercase
    words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())

    # Find misspelled words
    misspelled = spell.unknown(words)

    if not misspelled:
        print("No misspelled words found!")
        return

    print(f"Found {len(misspelled)} misspelled word(s):")
    print("-" * 40)

    for word in misspelled:
        print(f"Misspelled: '{word}'")

        # Get the most likely correction
        correction = spell.correction(word)
        print(f"Best suggestion: {correction}")

        # Get all possible candidates
        candidates = spell.candidates(word)
        if candidates:
            # Show top 5
            print(f"Other suggestions: {', '.join(list(candidates)[:50])}")
        else:
            print("No suggestions available")
        print("-" * 40)


# Example usage
find_misspelled_words("i Luve dugs")
