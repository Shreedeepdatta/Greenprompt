import tiktoken
import math
from typing import Optional, Dict, Any


class TokenEnergyCalculator:
    """
    A class to calculate token count and energy cost for processing prompts.
    """

    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialize the calculator with a specific model.

        Args:
            model_name: The model to use for tokenization (default: "gpt-4")
        """
        self.model_name = model_name
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base encoding (used by GPT-4)
            print(
                f"Warning: Model {model_name} not found, using cl100k_base encoding")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        # Energy cost estimates (in Wh per 1000 tokens)
        # These are rough estimates based on research and may vary
        self.energy_costs = {
            "gpt-3.5-turbo": 0.002,  # 2 mWh per 1000 tokens
            "gpt-4": 0.008,          # 8 mWh per 1000 tokens
            "gpt-4-turbo": 0.006,    # 6 mWh per 1000 tokens
            "claude-3-sonnet": 0.005,  # 5 mWh per 1000 tokens
            "claude-3-opus": 0.012,   # 12 mWh per 1000 tokens
            "default": 0.005         # Default estimate
        }

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        Args:
            text: The input text to tokenize

        Returns:
            int: Number of tokens in the text
        """
        try:
            tokens = self.tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            # Fallback: rough estimate of 4 characters per token
            return math.ceil(len(text) / 4)

    def calculate_energy_cost(self, token_count: int, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate energy cost for processing the given number of tokens.

        Args:
            token_count: Number of tokens to process
            model: Model name (optional, uses instance default if not provided)

        Returns:
            dict: Energy cost information including Wh, mWh, and CO2 estimate
        """
        if model is None:
            model = self.model_name

        # Get energy cost per 1000 tokens
        cost_per_1k = self.energy_costs.get(
            model, self.energy_costs["default"])

        # Calculate total energy cost in Wh
        energy_wh = (token_count / 1000) * cost_per_1k
        energy_mwh = energy_wh * 1000  # Convert to mWh

        # Estimate CO2 emissions (using global average of ~0.5 kg CO2/kWh)
        co2_grams = (energy_wh / 1000) * 500  # Convert to grams of CO2

        return {
            "tokens": token_count,
            "model": model,
            "energy_wh": round(energy_wh, 6),
            "energy_mwh": round(energy_mwh, 3),
            "co2_grams": round(co2_grams, 6),
            "cost_per_1k_tokens_wh": cost_per_1k
        }

    def analyze_prompt(self, prompt: str, model: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze a prompt to get token count and energy cost.

        Args:
            prompt: The prompt text to analyze
            model: Model name (optional)

        Returns:
            dict: Complete analysis including token count and energy metrics
        """
        token_count = self.count_tokens(prompt)
        energy_info = self.calculate_energy_cost(token_count, model)

        return {
            "prompt_length": len(prompt),
            "token_count": token_count,
            "tokens_per_char": round(token_count / len(prompt), 3) if len(prompt) > 0 else 0,
            "energy_cost": energy_info
        }

    def compare_models(self, prompt: str) -> Dict[str, Any]:
        """
        Compare energy costs across different models for the same prompt.

        Args:
            prompt: The prompt text to analyze

        Returns:
            dict: Comparison of energy costs across models
        """
        token_count = self.count_tokens(prompt)
        results = {}

        for model in self.energy_costs:
            if model != "default":
                results[model] = self.calculate_energy_cost(token_count, model)

        return {
            "prompt_length": len(prompt),
            "token_count": token_count,
            "model_comparison": results
        }


def main():
    """
    Example usage of the TokenEnergyCalculator
    """
    # Initialize calculator
    calculator = TokenEnergyCalculator("gpt-4")

    # Example prompt
    sample_prompt = """
    You are a helpful AI assistant. Please analyze the following data and provide insights:
    
    Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    Please calculate the mean, median, and standard deviation of this dataset.
    Also, create a brief summary of what this data might represent.
    """

    print("=== Token and Energy Analysis ===")
    print(f"Prompt: {sample_prompt[:50]}...")
    print()

    # Analyze the prompt
    analysis = calculator.analyze_prompt(sample_prompt)

    print(f"Prompt Length: {analysis['prompt_length']} characters")
    print(f"Token Count: {analysis['token_count']} tokens")
    print(f"Tokens per Character: {analysis['tokens_per_char']}")
    print()

    energy = analysis['energy_cost']
    print(f"Model: {energy['model']}")
    print(
        f"Energy Cost: {energy['energy_wh']:.6f} Wh ({energy['energy_mwh']:.3f} mWh)")
    print(f"Estimated CO2: {energy['co2_grams']:.6f} grams")
    print(f"Cost per 1000 tokens: {energy['cost_per_1k_tokens_wh']} Wh")
    print()

    # Compare across models
    print("=== Model Comparison ===")
    comparison = calculator.compare_models(sample_prompt)

    for model, data in comparison['model_comparison'].items():
        print(
            f"{model}: {data['energy_wh']:.6f} Wh, {data['co2_grams']:.6f}g CO2")
