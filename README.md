# Tokenizer Evaluator

The **Tokenizer Evaluator** is a Python tool for evaluating the performance of various tokenizers, particularly those used in natural language processing (NLP). This library provides metrics such as compression ratio, speed, average token lengths, and more, enabling users to assess the efficiency of different tokenization strategies.

## Table of Contents
- [Features](#features)
- [Available Functions](#available-functions)
- [Usage](#usage)
- [License](#license)

## Features
- Evaluate multiple tokenizer metrics:
  - **Compression Ratio**: Measures the ratio of tokens to characters in the input text.
  - **Speed**: Assesses how many tokens can be processed per second.
  - **Average Token Length**: Calculates the average length of tokens in both the input text and tokenizer vocabulary.
  - **Word-Initial Tokens**: Evaluates the proportion of tokens that start with a word character.
  - **Language-Specific Evaluations**: Perform evaluations on texts from various languages.

## Available Functions

### 1. `evaluate_metric(text: str, metric: str, **kwargs) -> EvaluationResult`
Evaluates a specified metric from the `tokenization_scorer` for the given text.

- **Parameters**:
  - `text`: The input text for evaluation.
  - `metric`: The name of the metric to evaluate (e.g., `shannon_entropy`, `compression_ratio`).
  - `**kwargs`: Additional parameters specific to the metric being evaluated.
  
- **Returns**: An `EvaluationResult` object containing the metric name, value, description, and additional information.

### 2. `evaluate_all_metrics(text: Union[str, List[str]], **kwargs) -> Dict[str, Union[EvaluationResult, List[EvaluationResult]]`
Evaluates all available metrics for the given text or list of texts.

- **Parameters**:
  - `text`: A single input text or a list of texts for evaluation.
  - `**kwargs`: Additional parameters that can be used in specific metrics.
  
- **Returns**: A dictionary where keys are metric names and values are their corresponding `EvaluationResult` objects.

### 3. `evaluate_compression(text: str) -> EvaluationResult`
Evaluates the compression ratio of the tokenizer for the specified text.

- **Parameters**:
  - `text`: The input text to evaluate for compression.
  
- **Returns**: An `EvaluationResult` object with details about the compression ratio, number of tokens, number of characters, and mean token length.

### 4. `evaluate_speed(text: str) -> EvaluationResult`
Measures the speed of the tokenizer by calculating how many tokens can be processed per second.

- **Parameters**:
  - `text`: The input text to evaluate for speed.
  
- **Returns**: An `EvaluationResult` object that includes the speed of token processing, number of tokens, time taken, and characters in the text.

### 5. `evaluate_average_token_length_text(text: str) -> EvaluationResult`
Calculates the average token length of the given text.

- **Parameters**:
  - `text`: The input text to evaluate for average token length.
  
- **Returns**: An `EvaluationResult` object that contains the average token length, number of tokens, and number of characters.

### 6. `evaluate_average_token_length_vocab() -> EvaluationResult`
Evaluates the average token length in the tokenizer's vocabulary.

- **Returns**: An `EvaluationResult` object that includes the average token length and the vocabulary size.

### 7. `evaluate_word_initial_tokens(text: str) -> EvaluationResult`
Measures the proportion of word-initial tokens in the token sequence.

- **Parameters**:
  - `text`: The input text to evaluate for word-initial tokens.
  
- **Returns**: An `EvaluationResult` object that details the proportion of word-initial tokens, along with the total number of tokens and characters.

### 8. `evaluate_compression_over_languages() -> Dict[str, EvaluationResult]`
Evaluates the compression ratio over predefined texts in multiple languages.

- **Returns**: A dictionary where the keys are language names and the values are their corresponding `EvaluationResult` objects.

### 9. `evaluate_compression_over_custom_files(paths: List[str]) -> Dict[str, EvaluationResult]`
Evaluates the compression ratio over custom files provided by the user.

- **Parameters**:
  - `paths`: A list of file paths to evaluate for compression.
  
- **Returns**: A dictionary where the keys are file paths and the values are their corresponding `EvaluationResult` objects.

## Usage

Here's an example of how to use the `TokenizerEvaluator` class:

```python
from transformers import PreTrainedTokenizerFast
from tokenizer_evaluator import TokenizerEvaluator

# Load your tokenizer
tokenizer_path = '/path/to/your/tokenizer'
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

# Create an instance of the TokenizerEvaluator
evaluator = TokenizerEvaluator(tokenizer)

# Sample texts for evaluation
sample_texts = [
    "This is a sample text to evaluate tokenizer metrics.",
    "Here's another example with different characters and lengths!",
    " This is one with many strange characters: !@#$%^&*()_+{}|:\"<>?`-=[]\;',./~",
]

# Evaluate all metrics
results = evaluator.evaluate_all_metrics(sample_texts)

# Print the results
evaluator.print_results(results, verbose=True)
