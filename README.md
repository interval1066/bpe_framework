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
 Let's break down the code step by step.
test_bpe Application Overview

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

This test application provides a comprehensive validation of the BPE tokenizer's core functionality, ensuring it works correctly for text processing tasks in the LM Framework.



