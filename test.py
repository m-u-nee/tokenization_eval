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
    additional_info: Optional[Dict[str, Any]] = None

class TokenizerEvaluator:
    """Main class for evaluating tokenizer performance."""
    
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Helper method to tokenize text and return token strings."""
        tokens = self.tokenizer.encode(
            text,
            add_special_tokens=False,
            return_tensors=None
        )
        return self.tokenizer.convert_ids_to_tokens(tokens)

    def evaluate_metric(self, text: Union[str, List[str]], metric: str, **kwargs) -> Union[EvaluationResult, List[EvaluationResult]]:
        """
        Generic method to evaluate any metric from tokenization_scorer.
        
        Args:
            text: Input text or list of texts to evaluate
            metric: Name of the metric to evaluate (e.g., 'shannon', 'renyi', 'bits')
            **kwargs: Additional arguments passed to the metric function
            
        Returns:
            EvaluationResult or list of EvaluationResults containing metric values
        """
        if isinstance(text, str):
            return self._evaluate_single_metric(text, metric, **kwargs)
        else:
            return [self._evaluate_single_metric(t, metric, **kwargs) for t in text]

    def _evaluate_single_metric(self, text: str, metric: str, **kwargs) -> EvaluationResult:
        """Evaluate a single metric for a single text."""
        tokens = self._tokenize_text(text)
        # Prepare text in the format expected by tokenization_scorer (list of lists of tokens)
        formatted_text = [tokens]
        
        with suppress_stdout():
            value = tokenization_scorer.score(formatted_text, metric=metric, **kwargs)
        
        return EvaluationResult(
            metric_name=metric,
            value=value,
            additional_info={
                "num_tokens": len(tokens),
                "num_chars": len(text),
                "parameters": kwargs
            }
        )

    def evaluate_all_metrics(self, text: Union[str, List[str]], **kwargs) -> Dict[str, Union[EvaluationResult, List[EvaluationResult]]]:
        """
        Evaluate all available metrics for the given text(s).
        
        Args:
            text: Input text or list of texts to evaluate
            **kwargs: Additional arguments for specific metrics
            
        Returns:
            Dictionary mapping metric names to their EvaluationResults
        """
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
        
        results = {}
        for metric_name, metric_params in metrics.items():
            results[metric_name] = self.evaluate_metric(text, **metric_params)
        
        compress_results = self.evaluate_compression(text)
        compress_results.additional_info["parameters"] = {}
        results["compression_ratio"] = compress_results

        speed_results = self.evaluate_speed(text)
        speed_results.additional_info["parameters"] = {}
        results["speed"] = speed_results

        avg_token_length_vocab = self.evaluate_average_token_length_vocab()
        avg_token_length_vocab.additional_info["parameters"] = {}
        results["average_token_length_vocab"] = avg_token_length_vocab

        avg_token_length_text = self.evaluate_average_token_length_text(text)
        avg_token_length_text.additional_info["parameters"] = {}
        results["average_token_length_text"] = avg_token_length_text
        
        return results

    # Keep existing methods (evaluate_compression, etc.)
    def evaluate_compression(self, text: Union[str, List[str]]) -> Union[EvaluationResult, List[EvaluationResult]]:
        """Evaluate compression ratio of the tokenizer."""
        if isinstance(text, str):
            return self._evaluate_single_compression(text)
        else:
            return [self._evaluate_single_compression(t) for t in text]
    
    def _evaluate_single_compression(self, text: str) -> EvaluationResult:
        tokens = self._tokenize_text(text)
        num_tokens = len(tokens)
        num_chars = len(text)
        
        compression_ratio = num_tokens / num_chars
        
        return EvaluationResult(
            metric_name="compression_ratio",
            value=compression_ratio,
            additional_info={
                "num_tokens": num_tokens,
                "num_chars": num_chars,
                "tokens": tokens,
                "mean_token_length": num_chars / num_tokens
            }
        )
    
    def evaluate_speed(self, text: Union[str, List[str]]) -> EvaluationResult:
        """Evaluate the speed of the tokenizer."""
        if isinstance(text, str):
            return self._evaluate_single_speed(text)
        else:
            return [self._evaluate_single_speed(t) for t in text]
        
    def _evaluate_single_speed(self, text: str) -> EvaluationResult:
        tokens = self._tokenize_text(text)
        num_tokens = len(tokens)
        num_chars = len(text)
        
        # Time the tokenization process
        with suppress_stdout():
            start_time = time.time()
            _ = self._tokenize_text(text)
            end_time = time.time()
        
        time_taken = end_time - start_time
        tokens_per_second = num_tokens / time_taken
        
        return EvaluationResult(
            metric_name="speed",
            value=tokens_per_second,
            additional_info={
                "num_tokens": num_tokens,
                "num_chars": num_chars,
                "time_taken": time_taken,
                "tokens_per_second": tokens_per_second
            }
        )

    def evaluate_average_token_length_vocab(self) -> EvaluationResult:
        """Evaluate the average token length in the vocabulary."""
        vocab = self.tokenizer.get_vocab()
        token_lengths = [len(token) for token in vocab.keys()]
        avg_token_length = np.mean(token_lengths)
        
        return EvaluationResult(
            metric_name="average_token_length_vocab",
            value=avg_token_length,
            additional_info={
                "vocab_size": len(vocab),
                "token_lengths": token_lengths
            }
        )
    def evaluate_average_token_length_text(self, text: Union[str, List[str]]) -> Union[EvaluationResult, List[EvaluationResult]]:
        """Evaluate the average token length in the text."""
        if isinstance(text, str):
            return self._evaluate_single_average_token_length_text(text)
        else:
            return [self._evaluate_single_average_token_length_text(t) for t in text]
        
    def _evaluate_single_average_token_length_text(self, text: str) -> EvaluationResult:
        tokens = self._tokenize_text(text)
        num_tokens = len(tokens)
        num_chars = len(text)
        
        avg_token_length = num_chars / num_tokens
        
        return EvaluationResult(
            metric_name="average_token_length_text",
            value=avg_token_length,
            additional_info={
                "num_tokens": num_tokens,
                "num_chars": num_chars,
                "avg_token_length": avg_token_length
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
    
    # Print results
    for metric_name, result in results.items():
        print(f"\n{metric_name}:")
        print(f"Value: {result.value:.4f}")
        print(f"Parameters: {result.additional_info['parameters']}")

if __name__ == "__main__":
    main()