#include "lm/models/transformer.hpp"
#include <iostream>
#include <random>
#include <cmath>
#include <immintrin.h>  // For AVX intrinsics

namespace lm {

Transformer::Transformer(size_t vocab_size, size_t d_model, size_t num_heads, 
    size_t d_ff, size_t num_layers, size_t max_seq_len, float dropout)
    : vocab_size_(vocab_size), d_model_(d_model), num_heads_(num_heads),
    d_ff_(d_ff), num_layers_(num_layers), max_seq_len_(max_seq_len), 
    dropout_(dropout), training_(false) {

    // Initialize embedding layer
    embedding_ = Tensor::randn({vocab_size_, d_model_}, 0.0, 0.02);
    embedding_.requires_grad(true);

    // Initialize positional encoding - use explicit vector
    positional_encoding_ = Tensor(std::vector<size_t>{max_seq_len_, d_model_});
    
    // Precompute constants for positional encoding
    const float log10000 = std::log(10000.0f);
    
    // Vectorized positional encoding initialization
    for (size_t pos = 0; pos < max_seq_len_; ++pos) {
        float* pos_data = &positional_encoding_.data()(pos, 0);
        
        #ifdef __AVX2__
        // Process 8 elements at a time with AVX
        for (size_t i = 0; i + 7 < d_model_; i += 8) {
            __m256i indices = _mm256_setr_epi32(i, i+1, i+2, i+3, i+4, i+5, i+6, i+7);
            __m256 even_mask = _mm256_cmp_ps(_mm256_cvtepi32_ps(_mm256_and_si256(indices, _mm256_set1_epi32(1))), 
                                            _mm256_setzero_ps(), _CMP_EQ_OQ);
            
            // Calculate denominator: 10000^(2*i/d_model)
            __m256 divisor = _mm256_div_ps(_mm256_mul_ps(_mm256_set1_ps(2.0f), 
                                          _mm256_cvtepi32_ps(indices)), 
                                          _mm256_set1_ps(static_cast<float>(d_model_)));
            __m256 denominator = _mm256_exp_ps(_mm256_mul_ps(divisor, _mm256_set1_ps(log10000)));
            
            // Calculate position / denominator
            __m256 pos_div = _mm256_div_ps(_mm256_set1_ps(static_cast<float>(pos)), denominator);
            
            // Apply sin to even indices, cos to odd indices
            __m256 sin_val = _mm256_sin_ps(pos_div);
            __m256 cos_val = _mm256_cos_ps(pos_div);
            
            // Blend based on even/odd
            __m256 result = _mm256_blendv_ps(cos_val, sin_val, even_mask);
            
            // Store result
            _mm256_storeu_ps(pos_data + i, result);
        }
        #else
        // Fallback for non-AVX
        for (size_t i = 0; i < d_model_; ++i) {
            if (i % 2 == 0) {
                pos_data[i] = std::sin(pos / std::pow(10000, 2.0 * i / d_model_));
            } else {
                pos_data[i] = std::cos(pos / std::pow(10000, 2.0 * (i - 1) / d_model_));
            }
        }
        #endif
    }
    positional_encoding_.requires_grad(true);

    // Initialize transformer blocks
    for (size_t i = 0; i < num_layers_; ++i) {
        transformer_blocks_.push_back(std::make_unique<TransformerBlock>(d_model_, num_heads_, d_ff_, dropout_));
    }

    // Initialize output layer
    output_layer_ = Tensor::randn({d_model_, vocab_size_}, 0.0, 0.02);
    output_layer_.requires_grad(true);

    std::cout << "Initialized Transformer with:\n";
    std::cout << "  vocab_size: " << vocab_size_ << "\n";
    std::cout << "  d_model: " << d_model_ << "\n";
    std::cout << "  num_heads: " << num_heads_ << "\n";
    std::cout << "  d_ff: " << d_ff_ << "\n";
    std::cout << "  num_layers: " << num_layers_ << "\n";
    std::cout << "  max_seq_len: " << max_seq_len_ << "\n";
    std::cout << "  dropout: " << dropout_ << "\n";
}

std::vector<Tensor> Transformer::parameters() const {
    std::vector<Tensor> params;

    // Add embedding parameters
    params.push_back(embedding_);

    // Add positional encoding parameters
    params.push_back(positional_encoding_);

    // Add transformer block parameters
    for (const auto& block : transformer_blocks_) {
        auto block_params = block->parameters();
        params.insert(params.end(), block_params.begin(), block_params.end());
    }

    // Add output layer parameters
    params.push_back(output_layer_);

    return params;
}

void Transformer::set_training(bool training) {
    training_ = training;

    // Set training mode for all transformer blocks
    for (auto& block : transformer_blocks_) {
        block->set_training(training);
    }

    std::cout << "Set training mode to: " << (training ? "true" : "false") << "\n";
}

Tensor Transformer::forward(const Tensor& input, const Tensor& mask) {
    // Get input dimensions
    size_t batch_size = input.shape()[0];
    size_t seq_len = input.shape()[1];

    // Convert token IDs to embeddings - use explicit vector
    Tensor embeddings(std::vector<size_t>{batch_size, seq_len, d_model_});

    // Vectorized embedding lookup
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            size_t token_id = static_cast<size_t>(input(b, t));
            if (token_id >= vocab_size_) continue;
            
            float* emb_ptr = &embeddings.data()(b * seq_len * d_model_ + t * d_model_);
            const float* token_emb_ptr = &embedding_.data()(token_id, 0);
            
            #ifdef __AVX2__
            // Process 8 elements at a time with AVX
            size_t i = 0;
            for (; i + 7 < d_model_; i += 8) {
                __m256 emb_vec = _mm256_loadu_ps(token_emb_ptr + i);
                _mm256_storeu_ps(emb_ptr + i, emb_vec);
            }
            // Process remaining elements
            for (; i < d_model_; ++i) {
                emb_ptr[i] = token_emb_ptr[i];
            }
            #else
            // Fallback for non-AVX
            for (size_t i = 0; i < d_model_; ++i) {
                emb_ptr[i] = token_emb_ptr[i];
            }
            #endif
        }
    }

    // Add positional encoding - vectorized
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            float* emb_ptr = &embeddings.data()(b * seq_len * d_model_ + t * d_model_);
            const float* pos_ptr = &positional_encoding_.data()(t, 0);
            
            #ifdef __AVX2__
            // Process 8 elements at a time with AVX
            size_t i = 0;
            for (; i + 7 < d_model_; i += 8) {
                __m256 emb_vec = _mm256_loadu_ps(emb_ptr + i);
                __m256 pos_vec = _mm256_loadu_ps(pos_ptr + i);
                __m256 result = _mm256_add_ps(emb_vec, pos_vec);
                _mm256_storeu_ps(emb_ptr + i, result);
            }
            // Process remaining elements
            for (; i < d_model_; ++i) {
                emb_ptr[i] += pos_ptr[i];
            }
            #else
            // Fallback for non-AVX
            for (size_t i = 0; i < d_model_; ++i) {
                emb_ptr[i] += pos_ptr[i];
            }
            #endif
        }
    }

    // Apply dropout during training
    if (training_) {
        embeddings = apply_dropout(embeddings, dropout_);
    }

    // Pass through transformer blocks
    Tensor hidden_states = embeddings;
    for (auto& block : transformer_blocks_) {
        hidden_states = block->forward(hidden_states, mask);
    }

    // Apply output layer - vectorized matrix multiplication
    Tensor logits(std::vector<size_t>{batch_size, seq_len, vocab_size_});
    
    // Precompute pointers for efficient access
    const float* hidden_data = hidden_states.data().data();
    const float* output_weight_data = output_layer_.data().data();
    float* logits_data = logits.data().data();
    
    // Vectorized matrix multiplication
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t t = 0; t < seq_len; ++t) {
            const float* hidden_ptr = hidden_data + (b * seq_len + t) * d_model_;
            float* logits_ptr = logits_data + (b * seq_len + t) * vocab_size_;
            
            #ifdef __AVX2__
            // Process output vocabulary in chunks of 8
            for (size_t v = 0; v < vocab_size_; v += 8) {
                __m256 sum = _mm256_setzero_ps();
                size_t v_end = std::min(v + 8, vocab_size_);
                
                // Dot product between hidden state and output weights
                for (size_t d = 0; d < d_model_; ++d) {
                    __m256 hidden_broadcast = _mm256_set1_ps(hidden_ptr[d]);
                    __m256 weight_vec = _mm256_loadu_ps(output_weight_data + d * vocab_size_ + v);
                    
                    // Handle the case when we're at the end and need to avoid reading past vocab_size
                    if (v_end < v + 8) {
                        // Create a mask for the valid elements
                        __m256i mask = _mm256_setr_epi32(
                            v < vocab_size_ ? -1 : 0,
                            v+1 < vocab_size_ ? -1 : 0,
                            v+2 < vocab_size_ ? -1 : 0,
                            v+3 < vocab_size_ ? -1 : 0,
                            v+4 < vocab_size_ ? -1 : 0,
                            v+5 < vocab_size_ ? -1 : 0,
                            v+6 < vocab_size_ ? -1 : 0,
                            v+7 < vocab_size_ ? -1 : 0
                        );
                        weight_vec = _mm256_maskload_ps(output_weight_data + d * vocab_size_ + v, mask);
                    }
                    
                    sum = _mm256_fmadd_ps(hidden_broadcast, weight_vec, sum);
                }
                
                // Store the result
                if (v_end == v + 8) {
                    _mm256_storeu_ps(logits_ptr + v, sum);
                } else {
                    // Handle the tail elements
                    float temp[8];
                    _mm256_storeu_ps(temp, sum);
                    for (size_t i = v; i < v_end; ++i) {
                        logits_ptr[i] = temp[i - v];
                    }
                }
            }
            #else
            // Fallback for non-AVX
            for (size_t v = 0; v < vocab_size_; ++v) {
                float sum = 0.0f;
                for (size_t d = 0; d < d_model_; ++d) {
                    sum += hidden_ptr[d] * output_weight_data[d * vocab_size_ + v];
                }
                logits_ptr[v] = sum;
            }
            #endif
        }
    }

    return logits;
}

Tensor Transformer::forward(const Tensor& input) {
    // Create an empty mask tensor
    Tensor mask;
    return forward(input, mask);
}

Tensor Transformer::apply_dropout(const Tensor& input, float dropout_rate) {
    if (dropout_rate <= 0.0) return input;
    
    Tensor output = input;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(1.0 - dropout_rate);
    
    #ifdef __AVX2__
    // Vectorized dropout
    float* data = output.data().data();
    size_t size = output.size();
    const float scale = 1.0f / (1.0f - dropout_rate);
    const __m256 scale_vec = _mm256_set1_ps(scale);
    
    // Generate random numbers in batches
    alignas(32) uint32_t random_bits[8];
    std::uniform_int_distribution<uint32_t> int_dist(0, UINT32_MAX);
    
    for (size_t i = 0; i < size; i += 8) {
        // Generate 8 random bits
        for (int j = 0; j < 8; ++j) {
            random_bits[j] = int_dist(gen);
        }
        
        __m256i mask_bits = _mm256_load_si256(reinterpret_cast<__m256i*>(random_bits));
        __m256 threshold = _mm256_set1_ps(1.0f - dropout_rate);
        __m256 rand_vals = _mm256_cvtepi32_ps(mask_bits);
        rand_vals = _mm256_div_ps(rand_vals, _mm256_set1_ps(static_cast<float>(UINT32_MAX)));
        
        // Create mask
        __m256 mask = _mm256_cmp_ps(rand_vals, threshold, _CMP_LE_OQ);
        
        // Load data
        __m256 data_vec = _mm256_loadu_ps(data + i);
        
        // Apply mask and scale
        data_vec = _mm256_and_ps(data_vec, mask);
        data_vec = _mm256_mul_ps(data_vec, scale_vec);
        
        // Store result
        _mm256_storeu_ps(data + i, data_vec);
    }
    #else
    // Fallback for non-AVX
    for (size_t i = 0; i < output.size(); ++i) {
        if (!dist(gen)) {
            output(i) = 0.0;
        } else {
            output(i) /= (1.0 - dropout_rate);
        }
    }
    #endif
    
    return output;
}

} // namespace lm
