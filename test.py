from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass
import numpy as np
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import transformers
import tokenization_scorer
import os
import sys
from contextlib import contextmanager
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@contextmanager
def suppress_stdout():
    """
    Temporarily suppresses output to stdout.
    """
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


@dataclass
class EvaluationResult:
    """Container for evaluation metric results."""
    metric_name: str
    value: float
    description: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None


class TokenizerEvaluator:
    """Main class for evaluating tokenizer performance."""
    
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
    
    def __repr__(self):
        return f"TokenizerEvaluator(tokenizer={self.tokenizer.__class__.__name__})"
    
    def __str__(self):
        return f"TokenizerEvaluator for {self.tokenizer.__class__.__name__}"

    def print_results(self, results: Dict[str, Union[EvaluationResult, List[EvaluationResult]]], verbose: bool = False):
        """Print evaluation results."""
        for metric_name, result in results.items():
            print(f"\n{metric_name}:")
            if verbose:
                print(f"Description: {result.description}")
            print(f"Value: {result.value:.4f}")
            if verbose and result.additional_info:
                print(f"Additional Info: {result.additional_info}")
                if "parameters" in result.additional_info and result.additional_info["parameters"]:
                    print(f"Parameters: {result.additional_info['parameters']}")
        

    def _tokenize_text(self, text: str) -> List[str]:
        """Helper method to tokenize text and return token strings."""
        tokens = self.tokenizer.encode(text, add_special_tokens=False, return_tensors=None)
        return self.tokenizer.convert_ids_to_tokens(tokens)

    def evaluate_metric(self, text: Union[str, List[str]], metric: str, **kwargs) -> Union[EvaluationResult, List[EvaluationResult]]:
        """Generic method to evaluate any metric from tokenization_scorer."""
        if isinstance(text, str):
            return self._evaluate_single_metric(text, metric, **kwargs)
        return [self._evaluate_single_metric(t, metric, **kwargs) for t in text]

    def _evaluate_single_metric(self, text: str, metric: str, **kwargs) -> EvaluationResult:
        """Evaluate a single metric for a single text."""
        tokens = self._tokenize_text(text)
        formatted_text = [tokens]
        
        with suppress_stdout():
            value = tokenization_scorer.score(formatted_text, metric=metric, **kwargs)
        
        description = self._get_metric_description(metric)
        
        return EvaluationResult(
            metric_name=metric,
            value=value,
            description=description,
            additional_info={
                "num_tokens": len(tokens),
                "num_chars": len(text),
                "parameters": kwargs
            }
        )

    def _get_metric_description(self, metric: str) -> str:
        """Return a description of the given metric."""
        descriptions = {
            "shannon_entropy": "Measures the uncertainty or information content of the token sequence.",
            "shannon_efficiency": "Indicates how efficiently the token sequence encodes information relative to its length.",
            "renyi_entropy": "A generalization of Shannon entropy that measures diversity with an adjustable parameter (power).",
            "renyi_efficiency": "Similar to Renyi entropy, but normalized by token sequence length.",
            "seq_len": "Measures the number of tokens in the sequence.",
            "percentile_freq": "Evaluates the frequency of tokens between specified percentiles of occurrence.",
            "bits": "Represents the size of the encoded sequence in bits."
        }
        return descriptions.get(metric, "No description available for this metric.")

    def evaluate_all_metrics(self, text: Union[str, List[str]], **kwargs) -> Dict[str, Union[EvaluationResult, List[EvaluationResult]]]:
        """Evaluate all available metrics for the given text(s)."""
        metrics = {
            "shannon_entropy": {"metric": "shannon_entropy"},
            "shannon_efficiency": {"metric": "shannon_efficiency"},
            "renyi_entropy": {"metric": "renyi_entropy", "power": kwargs.get("power", 2.5)},
            "renyi_efficiency": {"metric": "renyi_efficiency", "power": kwargs.get("power", 2.5)},
            "sequence_length": {"metric": "seq_len"},
            "percentile_freq": {
                "metric": "percentile_freq",
                "perc_start": kwargs.get("perc_start", 0.03),
                "perc_end": kwargs.get("perc_end", 0.83)
            },
            "bits": {"metric": "bits"}
        }
        
        results = {metric_name: self.evaluate_metric(text, **params) for metric_name, params in metrics.items()}
        results.update({
            "compression_ratio": self.evaluate_compression(text),
            "speed": self.evaluate_speed(text),
            "average_token_length_vocab": self.evaluate_average_token_length_vocab(),
            "average_token_length_text": self.evaluate_average_token_length_text(text),
            "word_initial_tokens": self.evaluate_word_initial_tokens(text)
        })
        
        return results

    def evaluate_compression(self, text: Union[str, List[str]]) -> EvaluationResult:
        """Evaluate compression ratio of the tokenizer."""
        return self._evaluate_generic(text, self._evaluate_single_compression)

    def evaluate_speed(self, text: Union[str, List[str]]) -> EvaluationResult:
        """Evaluate the speed of the tokenizer."""
        return self._evaluate_generic(text, self._evaluate_single_speed)

    def evaluate_average_token_length_text(self, text: Union[str, List[str]]) -> EvaluationResult:
        """Evaluate the average token length in the text."""
        return self._evaluate_generic(text, self._evaluate_single_average_token_length_text)

    def _evaluate_generic(self, text: Union[str, List[str]], method) -> Union[EvaluationResult, List[EvaluationResult]]:
        """Generic evaluation method to handle both single and list inputs."""
        if isinstance(text, str):
            return method(text)
        return [method(t) for t in text]
    
    def _evaluate_single_compression(self, text: str) -> EvaluationResult:
        tokens = self._tokenize_text(text)
        compression_ratio = len(tokens) / len(text)
        
        return EvaluationResult(
            metric_name="compression_ratio",
            value=compression_ratio,
            description="Measures the compression ratio between the number of tokens and the number of characters in the text.",
            additional_info={
                "num_tokens": len(tokens),
                "num_chars": len(text),
                "tokens": tokens,
                "mean_token_length": len(text) / len(tokens)
            }
        )
    
    def _evaluate_single_speed(self, text: str) -> EvaluationResult:
        tokens = self._tokenize_text(text)
        start_time = time.time()
        _ = self._tokenize_text(text)
        time_taken = time.time() - start_time
        tokens_per_second = len(tokens) / time_taken
        
        return EvaluationResult(
            metric_name="speed",
            value=tokens_per_second,
            description="Measures the speed of the tokenizer in tokens processed per second.",
            additional_info={
                "num_tokens": len(tokens),
                "num_chars": len(text),
                "time_taken": time_taken,
                "tokens_per_second": tokens_per_second
            }
        )

    def evaluate_average_token_length_vocab(self) -> EvaluationResult:
        """Evaluate the average token length in the vocabulary."""
        vocab = self.tokenizer.get_vocab()
        avg_token_length = np.mean([len(token) for token in vocab.keys()])
        
        return EvaluationResult(
            metric_name="average_token_length_vocab",
            value=avg_token_length,
            description="Measures the average length of tokens in the tokenizer's vocabulary.",
            additional_info={
                "vocab_size": len(vocab)
            }
        )

    def _evaluate_single_average_token_length_text(self, text: str) -> EvaluationResult:
        tokens = self._tokenize_text(text)
        avg_token_length = len(text) / len(tokens)
        
        return EvaluationResult(
            metric_name="average_token_length_text",
            value=avg_token_length,
            description="Measures the average length of tokens in the given text.",
            additional_info={
                "num_tokens": len(tokens),
                "num_chars": len(text)
            }
        )
    
    # Now we implement number/proportion of word-initial tokens (number of tokens without ##/_/Gwith dot as the first character)
    def evaluate_word_initial_tokens(self, text: Union[str, List[str]]) -> EvaluationResult:
        return self._evaluate_generic(text, self._evaluate_single_word_initial_tokens)
    
    def _evaluate_single_word_initial_tokens(self, text: str) -> EvaluationResult:
        tokens = self._tokenize_text(text)
        word_initial_tokens = [token for token in tokens if not token[0] in ['_', '#', '.', 'Ġ']]
        proportion = len(word_initial_tokens) / len(tokens)
        
        return EvaluationResult(
            metric_name="word_initial_tokens",
            value=proportion,
            description="Measures the proportion of word-initial tokens in the token sequence.",
            additional_info={
                "num_tokens": len(tokens),
                "num_chars": len(text),
                "word_initial_tokens": word_initial_tokens
            }
        )
    

# Example usage
def main():
    sample_texts = [
        "This is a sample text to evaluate tokenizer metrics.",
        "Here's another example with different characters and lengths!",
    ]
    tokenizer_path = '/Users/Mattia/Desktop/tokenizer'
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    evaluator = TokenizerEvaluator(tokenizer)
    
    # Evaluate all metrics
    results = evaluator.evaluate_all_metrics(sample_texts[0])
    verbose = True
    # Print results
    evaluator.print_results(results, verbose=verbose)
if __name__ == "__main__":
    main()