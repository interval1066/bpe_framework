# bpe_framework
Large Language Model for Agentic AI

Build: cmake -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..

#### The test_bpe application does the following:
 1. Includes necessary headers and defines the main function.
 2. Creates an instance of the BPETokenizer.
 3. Defines a training corpus (a vector of strings).
 4. Trains the tokenizer on the corpus with a specified vocabulary size (500 in this case).
 5. Tests the tokenizer by encoding a sample string ("the quick brown fox").
 6. Decodes the tokens back to a string and prints the original, tokens, and decoded string.
 7. Saves the tokenizer to a file ("bpe_model.txt").
 8. Loads the tokenizer from the file and verifies the loaded tokenizer's vocabulary size.
 The purpose of this test is to verify that the BPE tokenizer can be trained, encode, decode, and serialize/deserialize correctly.
 
#### The test_bpe application is a comprehensive test program that validates the functionality of the BPE tokenizer implementation in the LM Framework. Here's how it works:
1. Initialization

    Creates an instance of BPETokenizer

    Defines a training corpus with sample English text

2. Training Process

    Calls tokenizer.train(corpus, 500) to train the tokenizer

    The training process:

        Initializes with byte-level vocabulary (0-255)

        Analyzes word frequencies in the corpus

        Iteratively merges the most frequent character pairs

        Builds a vocabulary of 500 tokens (as specified)

3. Encoding Test

    Encodes the test string "the quick brown fox"

    The encoding process:

        Splits text into words

        Converts each character to its initial token ID

        Applies learned BPE merges to combine tokens

        Returns a sequence of integer token IDs

4. Decoding Test

    Decodes the token IDs back to text

    The decoding process:

        Converts each token ID back to its string representation

        Concatenates the strings to reconstruct the original text

5. Serialization Test

    Saves the trained tokenizer to "bpe_model.txt"

    The serialization process:

        Writes vocabulary size and token-ID mappings

        Records all learned merge rules

6. Deserialization Test

    Loads the tokenizer from "bpe_model.txt"

    Verifies the loaded tokenizer has the same vocabulary size

    Confirms the tokenizer can perform encoding/decoding

Expected Output
text

Training tokenizer...
Vocabulary size: 500
Original: the quick brown fox
Tokens: [list of token IDs]
Decoded: the quick brown fox
Successfully loaded tokenizer
Loaded vocabulary size: 500

Key Validations

    Training Completes without errors

    Encoding/Decoding Round-Trip preserves the original text

    Serialization/Deserialization maintains tokenizer state

    Vocabulary Size matches the specified target (500)

    Token IDs are consistent between sessions

# BPE Tokenizer Performance Test Suite

## Overview

This performance test application is a comprehensive benchmarking tool designed to evaluate the efficiency and scalability of the Byte Pair Encoding (BPE) tokenizer implementation. The test suite measures critical performance metrics including training time, memory usage, encoding/decoding speed, and serialization performance across various configurations.

## Key Features

### 1. Corpus Generation
- Automatically generates realistic test corpora using common AI/ML terminology
- Configurable sentence count and word range parameters
- Creates diverse text samples that mimic real-world language patterns

### 2. Multi-Dimensional Testing
- Tests multiple corpus sizes (100, 1000, 5000 sentences)
- Evaluates different vocabulary sizes (500, 1000, 2000 tokens)
- Measures performance across various workload scenarios

### 3. Comprehensive Performance Metrics
- **Training Time**: Measures how long it takes to build the BPE vocabulary from a corpus
- **Memory Usage**: Tracks peak memory consumption during training (Linux-specific)
- **Encoding Speed**: Calculates processing time per token during text encoding
- **Round-Trip Verification**: Ensures encoding/decoding preserves original content
- **Serialization Performance**: Measures model save/load operations

### 4. Validation Checks
- Verifies encoding/decoding consistency
- Detects potential data corruption issues
- Validates vocabulary construction

## Usage Scenarios

This performance test is ideal for:
- Benchmarking different BPE implementations
- Evaluating hardware suitability for language processing tasks
- Identifying performance bottlenecks in tokenization pipelines
- Testing scalability of tokenizer implementations
- Comparing optimization techniques

## Technical Implementation

The test suite utilizes:
- High-resolution timing with `<chrono>` for precise measurements
- Linux-specific memory tracking via `/proc/self/status`
- Randomized corpus generation with configurable parameters
- Exception handling for robust testing
- Automatic cleanup of temporary files

## Output Metrics

The application provides detailed performance reports including:
- Training duration in milliseconds
- Peak memory usage in megabytes
- Encoding speed in microseconds per token
- Serialization/deserialization times
- Vocabulary size validation
- Round-trip integrity verification

This test framework serves as an essential tool for developers and researchers working with BPE tokenizers, providing quantitative data to guide optimization efforts and implementation choices.

Based on the research of Timothy O'Neil, Frederick Warren, et. al.
