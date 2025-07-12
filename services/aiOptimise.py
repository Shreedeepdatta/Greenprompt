from transformers import pipeline
from keybert import KeyBERT


class PromptOptimizer:
    def __init__(self):
        # Summarizer based on DistilBART (fine-tuned DistilBERT for summarization)
        self.summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
            tokenizer="sshleifer/distilbart-cnn-12-6"
        )
        # KeyBERT initialized with DistilBERT sentence embeddings
        self.keyword_extractor = KeyBERT('distilbert-base-nli-mean-tokens')

    def summarize_prompt(self, text: str, max_length: int = 50, min_length: int = 20) -> str:
        """Uses DistilBERT to summarize the prompt."""
        summary = self.summarizer(
            text, max_length=max_length, min_length=min_length, do_sample=False)
        return summary[0]['summary_text']

    def extract_keywords(self, text: str, top_n: int = 5) -> list:
        """Uses KeyBERT with DistilBERT to extract keywords."""
        keywords = self.keyword_extractor.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=top_n
        )
        return [kw[0] for kw in keywords]

    def compress_prompt(self, text: str) -> str:
        """Uses extracted keywords to compress the prompt."""
        keywords = self.extract_keywords(text)
        return ' '.join(keywords)

    def optimize_prompt(self, text: str) -> dict:
        """Returns both the summary and keyword-compressed versions of the prompt."""
        return {
            "original": text,
            "summarized": self.summarize_prompt(text),
            "keyword_based": self.compress_prompt(text)
        }


# 👇 Example usage
if __name__ == "__main__":
    prompt = """
    Hello, I hope you're doing well. I would be really grateful if you could help me find a good Python tutorial 
    that explains object-oriented programming in detail, especially with classes and inheritance. 
    I want to improve my skills and write cleaner code. Thank you so much!
    """

    optimizer = PromptOptimizer()
    result = optimizer.optimize_prompt(prompt)

    print("\nOriginal Prompt:\n", result["original"])
    print("\nSummarized Prompt:\n", result["summarized"])
    print("\nKeyword-based Compression:\n", result["keyword_based"])
