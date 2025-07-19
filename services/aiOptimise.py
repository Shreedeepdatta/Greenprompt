from fastapi import APIRouter, HTTPException, Depends
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from keybert import KeyBERT
from dto.ai_optimize_dto import (
    PromptRequest, KeywordResponse, SummaryResponse,
    OptimizedPromptResponse, OptimizedEnergyResponse
)
from services.energy_calculation import TokenEnergyCalculator


class PromptOptimizer:
    def __init__(self):
        try:
            # Summarizer based on DistilBART (fine-tuned DistilBERT for summarization)
            model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
            tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
            self.summarizer = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer
            )
            # KeyBERT initialized with DistilBERT sentence embeddings
            self.keyword_extractor = KeyBERT('distilbert-base-nli-mean-tokens')
            self.models_loaded = True
        except Exception as e:
            self.models_loaded = False
            raise

    def summarize_prompt(self, text: str, max_length: int = 50, min_length: int = 20) -> SummaryResponse:
        """Uses DistilBERT to summarize the prompt and returns structured response."""
        try:
            summary = self.summarizer(
                text, max_length=max_length, min_length=min_length,
                do_sample=False
            )
            summary_text = summary[0]['summary_text']

            return SummaryResponse(
                summary=summary_text,
                original_length=len(text),
                summary_length=len(summary_text),
                compression_ratio=len(summary_text) / len(text)
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Summarization failed: {str(e)}")

    def extract_keywords(self, text: str, top_n: int = 5) -> KeywordResponse:
        """Uses KeyBERT with DistilBERT to extract keywords
            and returns structured response."""
        try:
            keywords = self.keyword_extractor.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=top_n
            )
            keyword_list = [kw[0] for kw in keywords]

            return KeywordResponse(
                keywords=keyword_list,
                count=len(keyword_list)
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Keyword extraction failed: {str(e)}")

    def compress_prompt(self, text: str, top_n: int = 5) -> str:
        """Uses extracted keywords to compress the prompt."""
        keyword_response = self.extract_keywords(text, top_n)
        return ' '.join(keyword_response.keywords)

    def optimize_prompt(self, request: PromptRequest) -> OptimizedPromptResponse:
        """Returns comprehensive optimization results
            with structured response."""
        try:
            # Get summary
            summary_response = self.summarize_prompt(
                request.text,
                request.max_length,
                request.min_length
            )

            # Get keywords and compressed version
            keyword_response = self.extract_keywords(
                request.text, request.top_keywords)
            keyword_compressed = ' '.join(keyword_response.keywords)

            # Calculate compression ratios
            original_len = len(request.text)
            summary_len = len(summary_response.summary)
            keyword_len = len(keyword_compressed)
            energy_calculator = TokenEnergyCalculator()
            response_original = energy_calculator.analyze_prompt(request.text)
            response_summarized = energy_calculator.analyze_prompt(
                summary_response.summary)

            return OptimizedEnergyResponse(
                original=request.text,
                original_length=original_len,
                summarized=summary_response.summary,
                summary_length=summary_len,
                keyword_based=keyword_compressed,
                keywords_count=keyword_response.count,
                compression_ratios={
                    "summary": summary_len / original_len,
                    "keywords": keyword_len / original_len
                },
                energy_consumed_original=response_original.energy_wh,
                energy_consumed_summarized=response_summarized.energy_wh,
                carbon_footprint_original=response_original.co2_grams,
                carbon_footprint_summarized=response_summarized.co2_grams,
                energy_savings=response_original.energy_wh - response_summarized.energy_wh,
                carbon_savings=response_original.co2_grams - response_summarized.co2_grams
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Optimization failed: {str(e)}")


