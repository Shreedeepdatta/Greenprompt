import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import re

# Download required NLTK data (run once)


def download_nltk_data():
    """Download all required NLTK data"""
    required_data = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('taggers/averaged_perceptron_tagger_eng',
         'averaged_perceptron_tagger_eng'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4')  # For wordnet lemmatizer
    ]

    for data_path, download_name in required_data:
        try:
            nltk.data.find(data_path)
            print(f"✓ {download_name} already available")
        except LookupError:
            print(f"Downloading {download_name}...")
            try:
                nltk.download(download_name)
                print(f"✓ {download_name} downloaded successfully")
            except Exception as e:
                print(f"⚠ Could not download {download_name}: {e}")
                print("Trying alternative method...")
                # Try without checking path first
                nltk.download(download_name, quiet=False)


# Download data
print("Checking and downloading required NLTK data...")
try:
    download_nltk_data()
    print("All NLTK data ready!\n")
except Exception as e:
    print(f"Warning: Some NLTK data may not be available: {e}")
    print("The script will use fallback methods where needed.\n")


class PromptOptimizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.custom_stop_words = {'english': {
            'please', 'kindly', 'a', 'an', 'the', 'such',
            'only', 'own', 'same', 'so', 'than', 'too',
            'very', 'about', 'after', 'all', 'also', 'any'
        }}
        self.lemmatizer = WordNetLemmatizer()

        # Request clauses and phrases to remove
        self.request_clauses = [
            # Direct requests
            r'\bhelp me\b',
            r'\bhelp me to\b',
            r'\bhelp me with\b',
            r'\bcan you help me\b',
            r'\bcould you help me\b',
            r'\bwould you help me\b',
            r'\bplease help me\b',

            # Polite requests
            r'\bi would like\b',
            r'\bi would like to\b',
            r'\bi want to\b',
            r'\bi need to\b',
            r'\bi need you to\b',
            r'\bi want you to\b',
            r'\bi would appreciate\b',
            r'\bi would be grateful\b',

            # Question starters
            r'\bcan you\b',
            r'\bcould you\b',
            r'\bwould you\b',
            r'\bwill you\b',
            r'\bare you able to\b',
            r'\bis it possible to\b',

            # Courtesy phrases
            r'\bplease\b',
            r'\bkindly\b',
            r'\bif you could\b',
            r'\bif possible\b',
            r'\bif you don\'t mind\b',

            # Uncertainty phrases
            r'\bi think\b',
            r'\bi believe\b',
            r'\bi guess\b',
            r'\bi suppose\b',
            r'\bmaybe\b',
            r'\bperhaps\b',

            # Redundant starters
            r'\bso\b',
            r'\bwell\b',
            r'\bokay\b',
            r'\balright\b',
            r'\bbasically\b',
            r'\bessentially\b',

            # Filler phrases
            r'\byou know\b',
            r'\bi mean\b',
            r'\blike\b(?=\s)',  # Only remove standalone 'like'
            r'\bactually\b',
            r'\breally\b',
            r'\bjust\b',
            r'\bsimply\b',

            # Conversational starters
            r'\bhi\b',
            r'\bhello\b',
            r'\bhey\b',
            r'\bexcuse me\b',
            r'\bsorry\b',
            r'\bpardon me\b',
        ]

        # Important POS tags to keep for prompt optimization
        self.important_pos_tags = {
            'NN', 'NNS', 'NNP', 'NNPS',  # Nouns
            'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Verbs
            'JJ', 'JJR', 'JJS',  # Adjectives
            'RB', 'RBR', 'RBS',  # Adverbs
            'CD',  # Numbers
            'FW'   # Foreign words
        }

        # Less important POS tags (can be filtered more aggressively)
        self.filter_pos_tags = {
            'DT',   # Determiners (the, a, an)
            'IN',   # Prepositions
            'CC',   # Coordinating conjunctions
            'TO',   # to
            'PRP',  # Personal pronouns
            'PRP$',  # Possessive pronouns
            'WDT',  # Wh-determiners
            'WP',   # Wh-pronouns
            'WP$',  # Possessive wh-pronouns
            'WRB'   # Wh-adverbs
        }

    def remove_request_clauses(self, text):
        """Remove common request clauses and phrases"""
        original_text = text
        removed_clauses = []

        # Convert to lowercase for matching but preserve original case
        text_lower = text.lower()

        for clause_pattern in self.request_clauses:
            matches = re.finditer(clause_pattern, text_lower, re.IGNORECASE)
            for match in matches:
                removed_clauses.append(match.group())

        # Remove the clauses (case-insensitive)
        for clause_pattern in self.request_clauses:
            text = re.sub(clause_pattern, '', text, flags=re.IGNORECASE)

        # Clean up extra spaces and punctuation
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        # Remove orphaned commas/semicolons
        text = re.sub(r'\s*[,;]\s*', ' ', text)
        # Remove leading punctuation
        text = re.sub(r'^\s*[,;.\-]\s*', '', text)
        text = re.sub(r'\s*[,;]\s*$', '', text)  # Remove trailing punctuation
        text = text.strip()

        return text, removed_clauses

    def preprocess_text(self, text):
        """Basic text preprocessing"""
        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove special characters but keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s\-\.\,\!\?\:\;]', '', text)

        return text

    def analyze_pos_tags(self, text):
        """Analyze POS tags in the text"""
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
        except Exception as e:
            print(f"Error with NLTK POS tagging: {e}")
            print("Falling back to simple tokenization...")
            # Fallback: simple tokenization without POS tagging
            tokens = text.split()
            pos_tags = [(token, 'NN') for token in tokens]  # Default to noun

        pos_analysis = defaultdict(list)
        for word, pos in pos_tags:
            pos_analysis[pos].append(word)

        return pos_tags, pos_analysis

    def find_stopwords_by_pos(self, pos_tagged_tokens):
        """Find stopwords considering POS tags"""
        stopwords_found = []
        important_words = []

        for word, pos in pos_tagged_tokens:
            # Check if word is a traditional stopword
            is_stopword = word.lower() in self.stop_words

            # Check if POS tag suggests it's less important
            is_filterable_pos = pos in self.filter_pos_tags

            if is_stopword or is_filterable_pos:
                stopwords_found.append(
                    (word, pos, 'stopword' if is_stopword
                        else 'filterable_pos'))
            elif pos in self.important_pos_tags:
                important_words.append((word, pos))

        return stopwords_found, important_words

    def optimize_prompt_conservative(self, prompt):
        """Conservative optimization - only remove clear stopwords and request clauses"""
        # First remove request clauses
        text_no_clauses, removed_clauses = self.remove_request_clauses(prompt)

        processed_text = self.preprocess_text(text_no_clauses)
        tokens = word_tokenize(processed_text)
        pos_tags = pos_tag(tokens)

        optimized_tokens = []
        for word, pos in pos_tags:
            # Keep word if it's not a traditional stopword
            if (word.lower() not in self.stop_words and
                    word.lower() not in self.custom_stop_words['english']):
                optimized_tokens.append(word)

        return ' '.join(optimized_tokens)

    def optimize_prompt_aggressive(self, prompt):
        """Aggressive optimization - remove stopwords, request clauses, and less important POS tags"""
        # First remove request clauses
        text_no_clauses, removed_clauses = self.remove_request_clauses(prompt)

        processed_text = self.preprocess_text(text_no_clauses)
        tokens = word_tokenize(processed_text)
        pos_tags = pos_tag(tokens)

        optimized_tokens = []
        for word, pos in pos_tags:
            # Keep word only if it's important based on POS and not a stopword
            if (word.lower() not in self.stop_words and
                    pos in self.important_pos_tags and
                    word.lower() not in self.custom_stop_words['english']):
                optimized_tokens.append(word)

        return ' '.join(optimized_tokens)

    def optimize_prompt_balanced(self, prompt):
        """Balanced optimization - smart filtering based on context"""
        # First remove request clauses
        text_no_clauses, removed_clauses = self.remove_request_clauses(prompt)

        processed_text = self.preprocess_text(text_no_clauses)
        tokens = word_tokenize(processed_text)
        pos_tags = pos_tag(tokens)

        optimized_tokens = []
        for i, (word, pos) in enumerate(pos_tags):
            # Always keep important words
            if pos in self.important_pos_tags:
                optimized_tokens.append(word)
            # Keep some function words if they're important for context
            elif pos in self.filter_pos_tags:
                # Keep prepositions that might be important for meaning
                if pos == 'IN' and len(optimized_tokens) > 0:
                    optimized_tokens.append(word)
                # Keep some determiners in specific contexts
                elif pos == 'DT' and i < len(pos_tags) - 1:
                    next_pos = pos_tags[i + 1][1]
                    if next_pos in self.important_pos_tags:
                        optimized_tokens.append(word)
            # Remove clear stopwords
            elif (word.lower() not in self.stop_words and
                    word.lower() not in self.custom_stop_words['english']):
                optimized_tokens.append(word)

        return ' '.join(optimized_tokens)

    def analyze_prompt(self, prompt):
        """Comprehensive analysis of the prompt"""
        print(f"Original prompt: {prompt}")
        print(f"Length: {len(prompt)} characters, {len(prompt.split())} words")
        print("-" * 50)

        # Request clause analysis
        text_no_clauses, removed_clauses = self.remove_request_clauses(prompt)
        if removed_clauses:
            print("Request clauses removed:")
            for clause in set(removed_clauses):  # Remove duplicates
                print(f"  '{clause}'")
            print(f"Text after clause removal: {text_no_clauses}")
            print()

        # POS analysis
        pos_tags, pos_analysis = self.analyze_pos_tags(text_no_clauses)
        print("POS Tag Analysis:")
        for pos, words in sorted(pos_analysis.items()):
            print(f"  {pos}: {words}")
        print()

        # Stopword analysis
        stopwords_found, important_words = self.find_stopwords_by_pos(pos_tags)
        print("Stopwords and filterable words found:")
        for word, pos, reason in stopwords_found:
            print(f"  '{word}' ({pos}) - {reason}")
        print()

        print("Important words to keep:")
        for word, pos in important_words:
            print(f"  '{word}' ({pos})")
        print()

        # Optimization results
        conservative = self.optimize_prompt_conservative(prompt)
        aggressive = self.optimize_prompt_aggressive(prompt)
        balanced = self.optimize_prompt_balanced(prompt)

        print("Optimization Results:")
        print(f"Conservative: {conservative}")
        print(
            f"  Length: {len(conservative)} chars, "
            f"{len(conservative.split())} words")
        print()
        print(f"Aggressive: {aggressive}")
        print(
            f"  Length: {len(aggressive)} chars, "
            f"{len(aggressive.split())} words")
        print()
        print(f"Balanced: {balanced}")
        print(
            f"  Length: {len(balanced)} chars, {len(balanced.split())} words")

        return {
            'original': prompt,
            'conservative': conservative,
            'aggressive': aggressive,
            'balanced': balanced,
            'removed_clauses': removed_clauses,
            'text_after_clause_removal': text_no_clauses,
            'pos_analysis': pos_analysis,
            'stopwords_found': stopwords_found,
            'important_words': important_words
        }


def main():
    optimizer = PromptOptimizer()

    # Example prompts
    example_prompts = [
        "Please write a detailed and comprehensive analysis of the current market trends in the technology sector.",
        "Can you help me understand the fundamental concepts of machine learning algorithms?",
        "I would like to know more about the benefits and drawbacks of renewable energy sources.",
        "Generate a creative story about a young adventurer who discovers a magical forest.",
        "Explain the process of photosynthesis in plants in simple terms that a child can understand."
    ]

    print("=== PROMPT OPTIMIZATION ANALYSIS ===\n")

    for i, prompt in enumerate(example_prompts, 1):
        print(f"EXAMPLE {i}:")
        optimizer.analyze_prompt(prompt)
        print("\n" + "="*80 + "\n")

    # Interactive mode
    print("Enter your own prompt to optimize (or 'quit' to exit):")
    while True:
        user_prompt = input("\nPrompt: ").strip()
        if user_prompt.lower() in ['quit', 'exit', 'q']:
            break
        if user_prompt:
            print()
            optimizer.analyze_prompt(user_prompt)
            print("\n" + "-"*50)


if __name__ == "__main__":
    main()
