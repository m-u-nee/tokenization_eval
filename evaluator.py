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
import functools
import logging
from transformers import logging as transformers_logging
from tqdm import tqdm
tqdm.tqdm = lambda *args, **kwargs: args[0]  # Return the iterable without progress bar

# Override the tqdm progress bar to disable it completely
transformers_logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class EvaluationResult:
    """Container for evaluation metric results."""
    metric_name: str
    value: float
    description: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None

def accepts_list(func):
    """Decorator to allow methods to accept either a string or a list of strings."""
    @functools.wraps(func)
    def wrapper(self, text, *args, **kwargs):
        if isinstance(text, str):
            return func(self, text, *args, **kwargs)
        return [func(self, t, *args, **kwargs) for t in text]
    return wrapper
class TokenizerEvaluator:
    """Main class for evaluating tokenizer performance.

    This class provides methods to evaluate various metrics of a given tokenizer,
    including compression ratio, speed, and average token lengths. It can handle
    both single strings and lists of strings for evaluation.
    """

    def __init__(self, tokenizer: Any):
        """Initialize the TokenizerEvaluator with a tokenizer.

        Args:
            tokenizer (Any): An instance of a tokenizer (e.g., from Hugging Face Transformers).
        """
        self.tokenizer = tokenizer

    def __repr__(self):
        """Return a string representation of the TokenizerEvaluator instance."""
        return f"TokenizerEvaluator(tokenizer={self.tokenizer.__class__.__name__})"

    def __str__(self):
        """Return a user-friendly string description of the TokenizerEvaluator instance."""
        return f"TokenizerEvaluator for {self.tokenizer.__class__.__name__}"

    def print_results(self, results: Dict[str, Union[EvaluationResult, List[EvaluationResult]]], verbose: bool = False):
        """Print evaluation results in a readable format.

        Args:
            results (Dict[str, Union[EvaluationResult, List[EvaluationResult]]]): 
                A dictionary containing metric names and their corresponding evaluation results.
            verbose (bool): If True, include additional information in the printed results.
        """
        for metric_name, result in results.items():
            print(f"\n{metric_name}:")
            if isinstance(result, list):
                for res in result:
                    self._print_single_result(res, verbose)
            else:
                self._print_single_result(result, verbose)

    def _print_single_result(self, result: EvaluationResult, verbose: bool):
        """Print a single evaluation result.

        Args:
            result (EvaluationResult): The evaluation result to print.
            verbose (bool): If True, include additional information in the printed result.
        """
        print(f"Value: {result.value:.4f}")
        if verbose:
            if result.description:
                print(f"Description: {result.description}")
            if result.additional_info:
                print(f"Additional Info: {result.additional_info}")

    def _tokenize_text(self, text: str) -> List[str]:
        """Helper method to tokenize text and return token strings.

        Args:
            text (str): The input text to tokenize.

        Returns:
            List[str]: A list of token strings obtained from the input text.
        """
        tokens = self.tokenizer.encode(text, add_special_tokens=False, return_tensors=None)
        return self.tokenizer.convert_ids_to_tokens(tokens)

    @accepts_list
    def evaluate_metric(self, text: str, metric: str, **kwargs) -> EvaluationResult:
        """Generic method to evaluate any metric from tokenization_scorer.

        Args:
            text (str): The input text for evaluation.
            metric (str): The name of the metric to evaluate.
            **kwargs: Additional parameters for metric evaluation.

        Returns:
            EvaluationResult: The result of the metric evaluation.
        """
        tokens = self._tokenize_text(text)
        formatted_text = [tokens]
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
        """Return a description of the given metric.

        Args:
            metric (str): The name of the metric.

        Returns:
            str: A description of the metric or a default message if not found.
        """
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
        """Evaluate all available metrics for the given text(s).

        Args:
            text (Union[str, List[str]]): The input text or a list of texts for evaluation.
            **kwargs: Additional parameters for specific metrics.

        Returns:
            Dict[str, Union[EvaluationResult, List[EvaluationResult]]]: 
                A dictionary containing metric names and their corresponding evaluation results.
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

        results = {metric_name: self.evaluate_metric(text, **params) for metric_name, params in metrics.items()}

        # Methods that require text as input
        text_metrics = {
            "compression_ratio": self.evaluate_compression,
            "speed": self.evaluate_speed,
            "average_token_length_text": self.evaluate_average_token_length_text,
            "word_initial_tokens": self.evaluate_word_initial_tokens
        }

        for metric_name, method in text_metrics.items():
            results[metric_name] = method(text)

        # Methods that do not require text as input
        non_text_metrics = {
            "average_token_length_vocab": self.evaluate_average_token_length_vocab
        }

        for metric_name, method in non_text_metrics.items():
            results[metric_name] = method()

        return results

    @accepts_list
    def evaluate_compression(self, text: str) -> EvaluationResult:
        """Evaluate compression ratio of the tokenizer.

        Args:
            text (str): The input text to evaluate for compression.

        Returns:
            EvaluationResult: The result of the compression ratio evaluation.
        """
        tokens = self._tokenize_text(text)
        compression_ratio = len(tokens) / len(text) if len(text) > 0 else 0
        return EvaluationResult(
            metric_name="compression_ratio",
            value=compression_ratio,
            description="Measures the compression ratio between the number of tokens and the number of characters in the text.",
            additional_info={
                "num_tokens": len(tokens),
                "num_chars": len(text),
                "mean_token_length": len(text) / len(tokens) if len(tokens) > 0 else 0
            }
        )

    @accepts_list
    def evaluate_speed(self, text: str) -> EvaluationResult:
        """Evaluate the speed of the tokenizer.

        Args:
            text (str): The input text to evaluate for speed.

        Returns:
            EvaluationResult: The result of the speed evaluation.
        """
        start_time = time.time()
        tokens = self._tokenize_text(text)
        time_taken = time.time() - start_time
        tokens_per_second = len(tokens) / time_taken if time_taken > 0 else 0
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
        """Evaluate the average token length in the vocabulary.

        Returns:
            EvaluationResult: The result of the average token length evaluation in the vocabulary.
        """
        vocab = self.tokenizer.get_vocab()
        avg_token_length = np.mean([len(token) for token in vocab.keys()]) if vocab else 0
        return EvaluationResult(
            metric_name="average_token_length_vocab",
            value=avg_token_length,
            description="Measures the average length of tokens in the tokenizer's vocabulary.",
            additional_info={
                "vocab_size": len(vocab)
            }
        )

    @accepts_list
    def evaluate_average_token_length_text(self, text: str) -> EvaluationResult:
        """Evaluate the average token length in the given text.

        Args:
            text (str): The input text to evaluate for average token length.

        Returns:
            EvaluationResult: The result of the average token length evaluation.
        """
        tokens = self._tokenize_text(text)
        avg_token_length = len(text) / len(tokens) if len(tokens) > 0 else 0
        return EvaluationResult(
            metric_name="average_token_length_text",
            value=avg_token_length,
            description="Measures the average length of tokens in the given text.",
            additional_info={
                "num_tokens": len(tokens),
                "num_chars": len(text)
            }
        )

    @accepts_list
    def evaluate_word_initial_tokens(self, text: str) -> EvaluationResult:
        """Evaluate the proportion of word-initial tokens in the token sequence.

        Args:
            text (str): The input text to evaluate for word-initial tokens.

        Returns:
            EvaluationResult: The result of the word-initial tokens evaluation.
        """
        tokens = self._tokenize_text(text)
        word_initial_tokens = [token for token in tokens if not token.startswith(('_', '#', '.', 'Ġ'))]
        proportion = len(word_initial_tokens) / len(tokens) if len(tokens) > 0 else 0
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
    
    def evaluate_compression_over_languages(self) -> EvaluationResult:
        """Evaluate the compression ratio over different languages.

        Returns:
            EvaluationResult: A dictionary of compression results for different languages.
        """
        languages = {
            "english": 'datasets/evaluation/english.txt',
            "french": 'datasets/evaluation/french.txt',
            "german": 'datasets/evaluation/german.txt',
            "spanish": 'datasets/evaluation/spanish.txt',
            "italian": 'datasets/evaluation/italian.txt',
            "python": 'datasets/evaluation/python.txt',
            "c": 'datasets/evaluation/c.txt'
        }
        results = {}
        for lang, path in languages.items():
            with open(path, 'r') as f:
                text = f.read()
            results[lang] = self.evaluate_compression(text)
        return results
    
    def evaluate_compression_over_custom_files(self, paths: List[str]) -> EvaluationResult:
        """Evaluate the compression ratio over custom files.

        Args:
            paths (List[str]): A list of file paths to evaluate for compression.

        Returns:
            EvaluationResult: A dictionary of compression results for the specified custom files.
        """
        results = {}
        for path in paths:
            with open(path, 'r') as f:
                text = f.read()
            results[path] = self.evaluate_compression(text)
        return results

# Example usage
def main():
    sample_texts = [
        "This is a sample text to evaluate tokenizer metrics.",
        "Here's another example with different characters and lengths!",
        " This is one with many strange characters: !@#$%^&*()_+{}|:\"<>?`-=[]\;',./~",
    ]
    tokenizer_path = '/Users/Mattia/Desktop/tokenizer'
    tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(tokenizer_path, tqdm_disable=True)
    evaluator = TokenizerEvaluator(tokenizer)
    
    # Evaluate all metrics
    results = evaluator.evaluate_all_metrics(sample_texts)
    # results = evaluator.evaluate_compression_over_languages()
    # results = evaluator.evaluate_compression_over_custom_files(['datasets/evaluation/english.txt', 'datasets/evaluation/french.txt'])
    # Print results
    evaluator.print_results(results, verbose = True)

if __name__ == "__main__":
    main()