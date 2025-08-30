# bpe_framework
## Byte Pair Encoding Framework
Large Language Model for Agentic AI

Fully internationalized framework for Agentic AI research

Requires:
1. Dr. Neils Lohmann’s Json for C++
	(https://github.com/nlohmann/json)
	sudo apt install nlohmann-json3-dev
2. Internationalzation library for Unicode by Frederick Roubert
 	(https://github.com/unicode-org/icu) sudo apt install libicu-dev
3. OpenNMT Tokenizer by Thuc Pham (https://github.com/OpenNMT/Tokenize)
	(Must be installed from source on Debian as far as I know)
4. Eigen Library for Linear Math
   (https://github.com/PX4/eigen)
	sudo apt install libeigen3-dev
6. BLAS (Basic Linear Algebra Subprograms) support (https://www.netlib.org/blas/)
	sudo apt install libblas3
7. The Parallel Hashmap Library (https://github.com/greg7mdp/parallel-hashmap)
	sudo apt-get install libparallel-hashmap-dev

### What’s here:
 A 100% C++ 17/STL implementation of a Byte Pair Encoding (Tokenization) AI Engine with speed at the foremost of the designer's minds, fully internationalized. Future plans include hooks for expansion and additional functionality with Python, other languages.

#### To Build:
Create a build directory in the top level bpe_framework 
Also contains a Code::Blocks project file, other IDEs coming.
#### The test_bpe application is a comprehensive test program that validates the functionality of the BPE tokenizer implementation in the LM Framework. Here's how it works:
1. Initialization:
    Creates an instance of BPETokenizer
    Defines a training corpus with sample English text

2. Training Process:
    Calls tokenizer.train(corpus, 500) to train the tokenizer
    The training process:
        Initializes with byte-level vocabulary (0-255)
        Analyzes word frequencies in the corpus
        Iteratively merges the most frequent character pairs
        Builds a vocabulary of 500 tokens (as specified)

3. Encoding Test:
    Encodes the test string "the quick brown fox"
    The encoding process:
        Splits text into words
        Converts each character to its initial token ID
        Applies learned BPE merges to combine tokens
        Returns a sequence of integer token IDs

4. Decoding Test:
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

## test_unicode.cpp

### Lower-level Unicode-specific tests:

    Unicode normalization functions

    Character boundary detection

    Grapheme cluster handling

    Encoding conversion utilities

    Validation of Unicode compliance


## BPE Tokenizer Performance Test Suite

### Overview

The performance test application is a comprehensive benchmarking tool designed to evaluate the efficiency and scalability of the Byte Pair Encoding (BPE) tokenizer implementation. The test suite measures critical performance metrics including training time, memory usage, encoding/decoding speed, and serialization performance across various configurations.

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

## Technical Summary: BPE Framework
### Overview

The BPE Framework is a C++-based neural network framework designed for building and training language models with Byte Pair Encoding (BPE) tokenization. It implements a complete deep learning stack with automatic differentiation, optimization, and model serialization capabilities.
Core Components
#### 1. Tensor Operations with Autograd

    Header-only Tensor class with Eigen backend for efficient linear algebra

    Automatic differentiation with backward propagation

    Comprehensive operator support: element-wise operations, matrix multiplication, reductions

    Activation functions: ReLU, GELU, Softmax, Sigmoid with gradient support

    Memory-efficient implementation with shape-aware operations

#### 2. BPE Tokenizer

    PIMPL pattern implementation for API stability

    Efficient vocabulary management with merge operations

    Encoding/decoding support for text processing

    Non-copyable design (uses unique_ptr) for proper resource management

#### 3. Neural Network Architecture

    Transformer-based language model implementation

    Configurable dimensions: embedding size, hidden layers, attention heads

    Parameter management with named parameters for serialization

    Training/inference modes support

#### 4. Training Infrastructure

    Adam optimizer with configurable hyperparameters

    Gradient accumulation and moment estimation

    Batch processing with sequence padding

    Loss computation (cross-entropy) with masking support

#### 5. Model Serialization

    Binary format with versioning and magic number validation

    Parameter-by-name storage and retrieval

    Shape preservation and data integrity checks

    Error handling for file operations and format validation

### Key Technical Features
#### Memory Management

    Eigen integration for optimized matrix operations

    Shape-aware memory allocation preventing unnecessary copies

    RAII principles for resource management

#### Performance Considerations

    Header-only design for Tensor class enabling compiler optimizations

    Batch processing for efficient training

    In-place operations where possible to reduce memory overhead

#### Extensibility

    Modular architecture allowing component replacement

    Clear interfaces between tokenizer, model, and training components

    Parameter naming convention supporting complex architectures

#### Architecture Patterns

    PIMPL Idiom: Used in tokenizer for stable ABI

    RAII: Comprehensive resource management throughout

    Builder Pattern: Model configuration through constructor parameters

    Strategy Pattern: Optimizer implementation allowing algorithm changes

#### Current Capabilities

    * Automatic differentiation with reverse-mode autograd

    * BPE tokenization with vocabulary learning

    * Transformer language model training

    * Adam optimization with moment estimation

    * Model serialization/deserialization

    * Configurable network architectures

    * Batch processing with padding

### Technical Stack

    C++17 with standard library components

    Eigen for linear algebra operations

    CMake for build system management

    Header-only design for core components

#### Usage Example

// Initialize components
BPETokenizer tokenizer(corpus);
LanguageModel model(tokenizer.vocab_size(), 512, 2048, 8);
LanguageModelTrainer trainer(tokenizer, 512, 2048, 8);

// Train model
trainer.train(training_corpus, 10, 32, 256);

trainer.save_model("language_model.bin");


Based on the research of Timothy O'Neil, Frederick Warren, et. al.

