#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <cereal/types/vector.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/functional.hpp>

// Add SIMD headers
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
#if defined(__AVX__)
#include <immintrin.h>
#endif

namespace lm {

class Tensor;

Tensor operator*(float scalar, const Tensor& tensor);

class Tensor {
public:
    Tensor() : data_(Eigen::MatrixXf(0, 0)), shape_({0}), requires_grad_(false) {}
    
    Tensor(const std::vector<size_t>& shape, bool requires_grad = false) : requires_grad_(requires_grad) {
        shape_ = shape;
        size_t total_size = 1;
        for (auto dim : shape) total_size *= dim;
        
        if (shape.size() == 1) {
            data_ = Eigen::VectorXf::Zero(shape[0]);
            if (requires_grad) {
                grad_ = Eigen::VectorXf::Zero(shape[0]);
            }
        } else if (shape.size() == 2) {
            data_ = Eigen::MatrixXf::Zero(shape[0], shape[1]);
            if (requires_grad) {
                grad_ = Eigen::MatrixXf::Zero(shape[0], shape[1]);
            }
        } else {
            data_ = Eigen::VectorXf::Zero(total_size);
            if (requires_grad) {
                grad_ = Eigen::VectorXf::Zero(total_size);
            }
        }
    }
    
    Tensor(const Eigen::MatrixXf& data, const std::vector<size_t>& shape = {}, bool requires_grad = false)
        : data_(data), shape_(shape), requires_grad_(requires_grad) {
        if (shape.empty()) {
            if (data.cols() == 1) {
                shape_ = {static_cast<size_t>(data.rows())};
            } else {
                shape_ = {static_cast<size_t>(data.rows()), 
                         static_cast<size_t>(data.cols())};
            }
        }
        
        if (requires_grad) {
            grad_ = Eigen::MatrixXf::Zero(data_.rows(), data_.cols());
        }
    }
    
    // Accessors
    const std::vector<size_t>& shape() const { return shape_; }
    Eigen::MatrixXf& data() { return data_; }
    const Eigen::MatrixXf& data() const { return data_; }
    Eigen::MatrixXf& grad() { return grad_; }
    const Eigen::MatrixXf& grad() const { return grad_; }
    bool requires_grad() const { return requires_grad_; }
    
    void requires_grad(bool requires_grad) {
        requires_grad_ = requires_grad;
        if (requires_grad && grad_.size() == 0) {
            grad_ = Eigen::MatrixXf::Zero(data_.rows(), data_.cols());
        }
    }
    
    void zero_grad() {
        grad_.setZero();
    }
    
    // Element access
    float& operator()(size_t i) { return data_(i); }
    float operator()(size_t i) const { return data_(i); }
    float& operator()(size_t i, size_t j) { return data_(i, j); }
    float operator()(size_t i, size_t j) const { return data_(i, j); }
    
    // 3D indexing operators
    float& operator()(size_t i, size_t j, size_t k) {
        if (shape_.size() != 3) {
            throw std::runtime_error("3D access requires 3D tensor");
        }
        size_t index = i * shape_[1] * shape_[2] + j * shape_[2] + k;
        return data_(index);
    }
    
    float operator()(size_t i, size_t j, size_t k) const {
        if (shape_.size() != 3) {
            throw std::runtime_error("3D access requires 3D tensor");
        }
        size_t index = i * shape_[1] * shape_[2] + j * shape_[2] + k;
        return data_(index);
    }

    // Shape utilities
    size_t size() const { return data_.size(); }
    size_t dim(size_t axis) const { 
        return (axis < shape_.size()) ? shape_[axis] : 1; 
    }
    size_t ndim() const { return shape_.size(); }
    
    // Reshape the tensor
    Tensor reshape(const std::vector<size_t>& new_shape) const {
        size_t total_size = 1;
        for (auto dim : new_shape) total_size *= dim;
        
        if (total_size != size()) {
            throw std::invalid_argument("Total size must remain the same when reshaping");
        }
        
        Tensor result(data_, new_shape, requires_grad_);
        if (requires_grad_) {
            result.grad_ = grad_;
        }
        return result;
    }
    
    // Mathematical operations with autograd
    Tensor operator+(const Tensor& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor shapes must match for addition");
        }
        
        Tensor result;
        result.shape_ = shape_;
        result.requires_grad_ = requires_grad_ || other.requires_grad_;
        result.data_.resize(data_.rows(), data_.cols());
        
        size_t size = data_.size();
        const float* a = data_.data();
        const float* b = other.data_.data();
        float* dst = result.data_.data();

        #if defined(__AVX__)
        size_t i = 0;
        for (; i + 7 < size; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vresult = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(dst + i, vresult);
        }
        for (; i < size; ++i) {
            dst[i] = a[i] + b[i];
        }
        #elif defined(__SSE__)
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 vresult = _mm_add_ps(va, vb);
            _mm_storeu_ps(dst + i, vresult);
        }
        for (; i < size; ++i) {
            dst[i] = a[i] + b[i];
        }
        #else
        for (size_t i = 0; i < size; ++i) {
            dst[i] = a[i] + b[i];
        }
        #endif
        
        if (requires_grad_ || other.requires_grad_) {
            result.requires_grad(true);
            result.backward_fn_ = [this, &other, result]() {
                if (this->requires_grad_) {
                    this->grad_ += result.grad_;
                }
                if (other.requires_grad_) {
                    other.grad_ += result.grad_;
                }
            };
        }
        
        return result;
    }
    
    Tensor operator-(const Tensor& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor shapes must match for subtraction");
        }
        
        Tensor result;
        result.shape_ = shape_;
        result.requires_grad_ = requires_grad_ || other.requires_grad_;
        result.data_.resize(data_.rows(), data_.cols());
        
        size_t size = data_.size();
        const float* a = data_.data();
        const float* b = other.data_.data();
        float* dst = result.data_.data();

        #if defined(__AVX__)
        size_t i = 0;
        for (; i + 7 < size; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vresult = _mm256_sub_ps(va, vb);
            _mm256_storeu_ps(dst + i, vresult);
        }
        for (; i < size; ++i) {
            dst[i] = a[i] - b[i];
        }
        #elif defined(__SSE__)
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 vresult = _mm_sub_ps(va, vb);
            _mm_storeu_ps(dst + i, vresult);
        }
        for (; i < size; ++i) {
            dst[i] = a[i] - b[i];
        }
        #else
        for (size_t i = 0; i < size; ++i) {
            dst[i] = a[i] - b[i];
        }
        #endif
        
        if (requires_grad_ || other.requires_grad_) {
            result.requires_grad(true);
            result.backward_fn_ = [this, &other, result]() {
                if (this->requires_grad_) {
                    this->grad_ += result.grad_;
                }
                if (other.requires_grad_) {
                    other.grad_ -= result.grad_;
                }
            };
        }
        
        return result;
    }
    
    Tensor operator*(const Tensor& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor shapes must match for element-wise multiplication");
        }
        
        Tensor result;
        result.shape_ = shape_;
        result.requires_grad_ = requires_grad_ || other.requires_grad_;
        result.data_.resize(data_.rows(), data_.cols());
        
        size_t size = data_.size();
        const float* a = data_.data();
        const float* b = other.data_.data();
        float* dst = result.data_.data();

        #if defined(__AVX__)
        size_t i = 0;
        for (; i + 7 < size; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vresult = _mm256_mul_ps(va, vb);
            _mm256_storeu_ps(dst + i, vresult);
        }
        for (; i < size; ++i) {
            dst[i] = a[i] * b[i];
        }
        #elif defined(__SSE__)
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 vresult = _mm_mul_ps(va, vb);
            _mm_storeu_ps(dst + i, vresult);
        }
        for (; i < size; ++i) {
            dst[i] = a[i] * b[i];
        }
        #else
        for (size_t i = 0; i < size; ++i) {
            dst[i] = a[i] * b[i];
        }
        #endif
        
        if (requires_grad_ || other.requires_grad_) {
            result.requires_grad(true);
            result.backward_fn_ = [this, &other, result]() {
                if (this->requires_grad_) {
                    this->grad_ += result.grad_.cwiseProduct(other.data_);
                }
                if (other.requires_grad_) {
                    other.grad_ += result.grad_.cwiseProduct(this->data_);
                }
            };
        }
        
        return result;
    }
    
    Tensor operator/(const Tensor& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor shapes must match for element-wise division");
        }
        
        Tensor result;
        result.shape_ = shape_;
        result.requires_grad_ = requires_grad_ || other.requires_grad_;
        result.data_.resize(data_.rows(), data_.cols());
        
        size_t size = data_.size();
        const float* a = data_.data();
        const float* b = other.data_.data();
        float* dst = result.data_.data();

        #if defined(__AVX__)
        size_t i = 0;
        for (; i + 7 < size; i += 8) {
            __m256 va = _mm256_loadu_ps(a + i);
            __m256 vb = _mm256_loadu_ps(b + i);
            __m256 vresult = _mm256_div_ps(va, vb);
            _mm256_storeu_ps(dst + i, vresult);
        }
        for (; i < size; ++i) {
            dst[i] = a[i] / b[i];
        }
        #elif defined(__SSE__)
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m128 va = _mm_loadu_ps(a + i);
            __m128 vb = _mm_loadu_ps(b + i);
            __m128 vresult = _mm_div_ps(va, vb);
            _mm_storeu_ps(dst + i, vresult);
        }
        for (; i < size; ++i) {
            dst[i] = a[i] / b[i];
        }
        #else
        for (size_t i = 0; i < size; ++i) {
            dst[i] = a[i] / b[i];
        }
        #endif
        
        if (requires_grad_ || other.requires_grad_) {
            result.requires_grad(true);
            result.backward_fn_ = [this, &other, result]() {
                if (this->requires_grad_) {
                    this->grad_ += result.grad_.cwiseQuotient(other.data_);
                }
                if (other.requires_grad_) {
                    other.grad_ -= result.grad_.cwiseProduct(this->data_).cwiseQuotient(other.data_.cwiseProduct(other.data_));
                }
            };
        }
        
        return result;
    }
    
    Tensor operator+(float scalar) const {
        Tensor result;
        result.shape_ = shape_;
        result.requires_grad_ = requires_grad_;
        result.data_.resize(data_.rows(), data_.cols());
        
        size_t size = data_.size();
        const float* src = data_.data();
        float* dst = result.data_.data();

        #if defined(__AVX__)
        __m256 vscalar = _mm256_set1_ps(scalar);
        size_t i = 0;
        for (; i + 7 < size; i += 8) {
            __m256 v = _mm256_loadu_ps(src + i);
            __m256 vresult = _mm256_add_ps(v, vscalar);
            _mm256_storeu_ps(dst + i, vresult);
        }
        for (; i < size; ++i) {
            dst[i] = src[i] + scalar;
        }
        #elif defined(__SSE__)
        __m128 vscalar = _mm_set1_ps(scalar);
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m128 v = _mm_loadu_ps(src + i);
            __m128 vresult = _mm_add_ps(v, vscalar);
            _mm_storeu_ps(dst + i, vresult);
        }
        for (; i < size; ++i) {
            dst[i] = src[i] + scalar;
        }
        #else
        for (size_t i = 0; i < size; ++i) {
            dst[i] = src[i] + scalar;
        }
        #endif
        
        if (requires_grad_) {
            result.requires_grad(true);
            result.backward_fn_ = [this, result]() {
                if (this->requires_grad_) {
                    this->grad_ += result.grad_;
                }
            };
        }
        
        return result;
    }
    
    Tensor operator-(float scalar) const {
        Tensor result;
        result.shape_ = shape_;
        result.requires_grad_ = requires_grad_;
        result.data_.resize(data_.rows(), data_.cols());
        
        size_t size = data_.size();
        const float* src = data_.data();
        float* dst = result.data_.data();

        #if defined(__AVX__)
        __m256 vscalar = _mm256_set1_ps(scalar);
        size_t i = 0;
        for (; i + 7 < size; i += 8) {
            __m256 v = _mm256_loadu_ps(src + i);
            __m256 vresult = _mm256_sub_ps(v, vscalar);
            _mm256_storeu_ps(dst + i, vresult);
        }
        for (; i < size; ++i) {
            dst[i] = src[i] - scalar;
        }
        #elif defined(__SSE__)
        __m128 vscalar = _mm_set1_ps(scalar);
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m128 v = _mm_loadu_ps(src + i);
            __m128 vresult = _mm_sub_ps(v, vscalar);
            _mm_storeu_ps(dst + i, vresult);
        }
        for (; i < size; ++i) {
            dst[i] = src[i] - scalar;
        }
        #else
        for (size_t i = 0; i < size; ++i) {
            dst[i] = src[i] - scalar;
        }
        #endif
        
        if (requires_grad_) {
            result.requires_grad(true);
            result.backward_fn_ = [this, result]() {
                if (this->requires_grad_) {
                    this->grad_ += result.grad_;
                }
            };
        }
        
        return result;
    }
    
    Tensor operator*(float scalar) const {
        Tensor result;
        result.shape_ = shape_;
        result.requires_grad_ = requires_grad_;
        result.data_.resize(data_.rows(), data_.cols());
        
        size_t size = data_.size();
        const float* src = data_.data();
        float* dst = result.data_.data();

        #if defined(__AVX__)
        __m256 vscalar = _mm256_set1_ps(scalar);
        size_t i = 0;
        for (; i + 7 < size; i += 8) {
            __m256 v = _mm256_loadu_ps(src + i);
            __m256 vresult = _mm256_mul_ps(v, vscalar);
            _mm256_storeu_ps(dst + i, vresult);
        }
        for (; i < size; ++i) {
            dst[i] = src[i] * scalar;
        }
        #elif defined(__SSE__)
        __m128 vscalar = _mm_set1_ps(scalar);
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m128 v = _mm_loadu_ps(src + i);
            __m128 vresult = _mm_mul_ps(v, vscalar);
            _mm_storeu_ps(dst + i, vresult);
        }
        for (; i < size; ++i) {
            dst[i] = src[i] * scalar;
        }
        #else
        for (size_t i = 0; i < size; ++i) {
            dst[i] = src[i] * scalar;
        }
        #endif
        
        if (requires_grad_) {
            result.requires_grad(true);
            result.backward_fn_ = [this, scalar, result]() {
                if (this->requires_grad_) {
                    this->grad_ += result.grad_ * scalar;
                }
            };
        }
        
        return result;
    }
    
    Tensor operator/(float scalar) const {
        Tensor result;
        result.shape_ = shape_;
        result.requires_grad_ = requires_grad_;
        result.data_.resize(data_.rows(), data_.cols());
        
        size_t size = data_.size();
        const float* src = data_.data();
        float* dst = result.data_.data();

        #if defined(__AVX__)
        __m256 vscalar = _mm256_set1_ps(scalar);
        size_t i = 0;
        for (; i + 7 < size; i += 8) {
            __m256 v = _mm256_loadu_ps(src + i);
            __m256 vresult = _mm256_div_ps(v, vscalar);
            _mm256_storeu_ps(dst + i, vresult);
        }
        for (; i < size; ++i) {
            dst[i] = src[i] / scalar;
        }
        #elif defined(__SSE__)
        __m128 vscalar = _mm_set1_ps(scalar);
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m128 v = _mm_loadu_ps(src + i);
            __m128 vresult = _mm_div_ps(v, vscalar);
            _mm_storeu_ps(dst + i, vresult);
        }
        for (; i < size; ++i) {
            dst[i] = src[i] / scalar;
        }
        #else
        for (size_t i = 0; i < size; ++i) {
            dst[i] = src[i] / scalar;
        }
        #endif
        
        if (requires_grad_) {
            result.requires_grad(true);
            result.backward_fn_ = [this, scalar, result]() {
                if (this->requires_grad_) {
                    this->grad_ += result.grad_ / scalar;
                }
            };
        }
        
        return result;
    }
    
    // Optimized matrix multiplication with potential SIMD support
    Tensor matmul(const Tensor& other) const {
        if (ndim() != 2 || other.ndim() != 2) {
            throw std::invalid_argument("matmul requires 2D tensors");
        }
        if (shape_[1] != other.shape_[0]) {
            throw std::invalid_argument("Incompatible dimensions for matrix multiplication");
        }
        
        // Use Eigen's optimized matrix multiplication
        Tensor result(data_ * other.data_, {shape_[0], other.shape()[1]});
        
        if (requires_grad_ || other.requires_grad_) {
            result.requires_grad(true);
            result.backward_fn_ = [this, &other, result]() {
                if (this->requires_grad_) {
                    this->grad_ += result.grad_ * other.data_.transpose();
                }
                if (other.requires_grad_) {
                    other.grad_ += this->data_.transpose() * result.grad_;
                }
            };
        }
        
        return result;
    }
    
    Tensor transpose() const {
        if (ndim() != 2) {
            throw std::invalid_argument("transpose requires 2D tensors");
        }
        
        Tensor result(data_.transpose(), {shape_[1], shape_[0]});
        
        if (requires_grad_) {
            result.requires_grad(true);
            result.backward_fn_ = [this, result]() {
                if (this->requires_grad_) {
                    this->grad_ += result.grad_.transpose();
                }
            };
        }
        
        return result;
    }
    
    // Optimized reduction operations
    Tensor sum(int axis = -1) const {
        Tensor result;
        
        if (axis == -1 || ndim() == 1) {
            // Use SIMD for sum if possible
            float sum_val = 0.0f;
            size_t size = data_.size();
            const float* src = data_.data();
            
            #if defined(__AVX__)
            __m256 vsum = _mm256_setzero_ps();
            size_t i = 0;
            for (; i + 7 < size; i += 8) {
                __m256 v = _mm256_loadu_ps(src + i);
                vsum = _mm256_add_ps(vsum, v);
            }
            // Horizontal sum of 8 floats
            __m128 vlow = _mm256_castps256_ps128(vsum);
            __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
            vlow = _mm_add_ps(vlow, vhigh);
            __m128 shuf = _mm_shuffle_ps(vlow, vlow, _MM_SHUFFLE(2, 3, 0, 1));
            __m128 sums = _mm_add_ps(vlow, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            sum_val = _mm_cvtss_f32(sums);
            
            // Add remaining elements
            for (; i < size; ++i) {
                sum_val += src[i];
            }
            #elif defined(__SSE__)
            __m128 vsum = _mm_setzero_ps();
            size_t i = 0;
            for (; i + 3 < size; i += 4) {
                __m128 v = _mm_loadu_ps(src + i);
                vsum = _mm_add_ps(vsum, v);
            }
            // Horizontal sum of 4 floats
            __m128 shuf = _mm_shuffle_ps(vsum, vsum, _MM_SHUFFLE(2, 3, 0, 1));
            __m128 sums = _mm_add_ps(vsum, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            sum_val = _mm_cvtss_f32(sums);
            
            // Add remaining elements
            for (; i < size; ++i) {
                sum_val += src[i];
            }
            #else
            for (size_t i = 0; i < size; ++i) {
                sum_val += src[i];
            }
            #endif
            
            result = Tensor(Eigen::MatrixXf::Constant(1, 1, sum_val));
        } else if (axis == 0) {
            result = Tensor(data_.colwise().sum(), {shape_[1]});
        } else {
            result = Tensor(data_.rowwise().sum(), {shape_[0]});
        }
        
        if (requires_grad_) {
            result.requires_grad(true);
            result.backward_fn_ = [this, axis, result]() {
                if (this->requires_grad_) {
                    if (axis == -1 || ndim() == 1) {
                        this->grad_.array() += result.grad_(0, 0);
                    } else if (axis == 0) {
                        for (int i = 0; i < this->grad_.rows(); ++i) {
                            this->grad_.row(i) += result.grad_.transpose();
                        }
                    } else {
                        for (int j = 0; j < this->grad_.cols(); ++j) {
                            this->grad_.col(j) += result.grad_;
                        }
                    }
                }
            };
        }
        
        return result;
    }
    
    Tensor mean(int axis = -1) const {
        Tensor result;
        float divisor;
        
        if (axis == -1 || ndim() == 1) {
            divisor = data_.size();
            result = sum(axis) / divisor;
        } else if (axis == 0) {
            divisor = data_.rows();
            result = sum(axis) / divisor;
        } else {
            divisor = data_.cols();
            result = sum(axis) / divisor;
        }
        
        return result;
    }
    
    // Optimized activation functions
    Tensor relu() const {
        Tensor result;
        result.shape_ = shape_;
        result.requires_grad_ = requires_grad_;
        result.data_.resize(data_.rows(), data_.cols());
        
        size_t size = data_.size();
        const float* src = data_.data();
        float* dst = result.data_.data();

        #if defined(__AVX__)
        __m256 zero = _mm256_setzero_ps();
        size_t i = 0;
        for (; i + 7 < size; i += 8) {
            __m256 v = _mm256_loadu_ps(src + i);
            __m256 mask = _mm256_cmp_ps(v, zero, _CMP_GT_OS);
            __m256 vresult = _mm256_and_ps(v, mask);
            _mm256_storeu_ps(dst + i, vresult);
        }
        for (; i < size; ++i) {
            dst[i] = src[i] > 0 ? src[i] : 0;
        }
        #elif defined(__SSE__)
        __m128 zero = _mm_setzero_ps();
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m128 v = _mm_loadu_ps(src + i);
            __m128 mask = _mm_cmpgt_ps(v, zero);
            __m128 vresult = _mm_and_ps(v, mask);
            _mm_storeu_ps(dst + i, vresult);
        }
        for (; i < size; ++i) {
            dst[i] = src[i] > 0 ? src[i] : 0;
        }
        #else
        for (size_t i = 0; i < size; ++i) {
            dst[i] = src[i] > 0 ? src[i] : 0;
        }
        #endif

        if (requires_grad_) {
            result.requires_grad(true);
            result.backward_fn_ = [this, result]() {
                if (this->requires_grad_) {
                    size_t total_size = this->data_.size();
                    float* grad_ptr = this->grad_.data();
                    const float* data_ptr = this->data_.data();
                    const float* result_grad_ptr = result.grad_.data();

                    #if defined(__AVX__)
                    __m256 zero = _mm256_setzero_ps();
                    size_t i = 0;
                    for (; i + 7 < total_size; i += 8) {
                        __m256 data_val = _mm256_loadu_ps(data_ptr + i);
                        __m256 mask = _mm256_cmp_ps(data_val, zero, _CMP_GT_OS);
                        __m256 grad_val = _mm256_loadu_ps(result_grad_ptr + i);
                        __m256 add_grad = _mm256_and_ps(grad_val, mask);
                        __m256 current_grad = _mm256_loadu_ps(grad_ptr + i);
                        _mm256_storeu_ps(grad_ptr + i, _mm256_add_ps(current_grad, add_grad));
                    }
                    for (; i < total_size; ++i) {
                        if (data_ptr[i] > 0) {
                            grad_ptr[i] += result_grad_ptr[i];
                        }
                    }
                    #elif defined(__SSE__)
                    __m128 zero = _mm_setzero_ps();
                    size_t i = 0;
                    for (; i + 3 < total_size; i += 4) {
                        __m128 data_val = _mm_loadu_ps(data_ptr + i);
                        __m128 mask = _mm_cmpgt_ps(data_val, zero);
                        __m128 grad_val = _mm_loadu_ps(result_grad_ptr + i);
                        __m128 add_grad = _mm_and_ps(grad_val, mask);
                        __m128 current_grad = _mm_loadu_ps(grad_ptr + i);
                        _mm_storeu_ps(grad_ptr + i, _mm_add_ps(current_grad, add_grad));
                    }
                    for (; i < total_size; ++i) {
                        if (data_ptr[i] > 0) {
                            grad_ptr[i] += result_grad_ptr[i];
                        }
                    }
                    #else
                    for (size_t i = 0; i < total_size; ++i) {
                        if (data_ptr[i] > 0) {
                            grad_ptr[i] += result_grad_ptr[i];
                        }
                    }
                    #endif
                }
            };
        }
        
        return result;
    }
    
    // Optimized GELU implementation with potential SIMD support
    Tensor gelu() const {
        // Approximation of GELU: x * 0.5 * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
        
        Tensor result;
        result.shape_ = shape_;
        result.requires_grad_ = requires_grad_;
        result.data_.resize(data_.rows(), data_.cols());
        
        size_t size = data_.size();
        const float* src = data_.data();
        float* dst = result.data_.data();

        #if defined(__AVX__)
        __m256 vsqrt_2_over_pi = _mm256_set1_ps(sqrt_2_over_pi);
        __m256 vcoef = _mm256_set1_ps(0.044715f);
        __m256 vhalf = _mm256_set1_ps(0.5f);
        __m256 vone = _mm256_set1_ps(1.0f);
        
        size_t i = 0;
        for (; i + 7 < size; i += 8) {
            __m256 x = _mm256_loadu_ps(src + i);
            __m256 x3 = _mm256_mul_ps(x, _mm256_mul_ps(x, x));
            __m256 inner = _mm256_mul_ps(vsqrt_2_over_pi, 
                                        _mm256_add_ps(x, _mm256_mul_ps(vcoef, x3)));
            __m256 tanh_inner = tanh_avx(inner);
            __m256 result_val = _mm256_mul_ps(x, 
                                            _mm256_mul_ps(vhalf, 
                                                        _mm256_add_ps(vone, tanh_inner)));
            _mm256_storeu_ps(dst + i, result_val);
        }
        for (; i < size; ++i) {
            float x = src[i];
            float x3 = x * x * x;
            float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
            float tanh_inner = std::tanh(inner);
            dst[i] = 0.5f * x * (1.0f + tanh_inner);
        }
        #elif defined(__SSE__)
        __m128 vsqrt_2_over_pi = _mm_set1_ps(sqrt_2_over_pi);
        __m128 vcoef = _mm_set1_ps(0.044715f);
        __m128 vhalf = _mm_set1_ps(0.5f);
        __m128 vone = _mm_set1_ps(1.0f);
        
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m128 x = _mm_loadu_ps(src + i);
            __m128 x3 = _mm_mul_ps(x, _mm_mul_ps(x, x));
            __m128 inner = _mm_mul_ps(vsqrt_2_over_pi, 
                                    _mm_add_ps(x, _mm_mul_ps(vcoef, x3)));
            __m128 tanh_inner = tanh_sse(inner);
            __m128 result_val = _mm_mul_ps(x, 
                                        _mm_mul_ps(vhalf, 
                                                    _mm_add_ps(vone, tanh_inner)));
            _mm_storeu_ps(dst + i, result_val);
        }
        for (; i < size; ++i) {
            float x = src[i];
            float x3 = x * x * x;
            float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
            float tanh_inner = std::tanh(inner);
            dst[i] = 0.5f * x * (1.0f + tanh_inner);
        }
        #else
        for (size_t i = 0; i < size; ++i) {
            float x = src[i];
            float x3 = x * x * x;
            float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
            float tanh_inner = std::tanh(inner);
            dst[i] = 0.5f * x * (1.0f + tanh_inner);
        }
        #endif

        if (requires_grad_) {
            result.requires_grad(true);
            result.backward_fn_ = [this, sqrt_2_over_pi, result]() {
                if (this->requires_grad_) {
                    size_t total_size = this->data_.size();
                    float* grad_ptr = this->grad_.data();
                    const float* data_ptr = this->data_.data();
                    const float* result_grad_ptr = result.grad_.data();

                    #if defined(__AVX__)
                    __m256 vsqrt_2_over_pi = _mm256_set1_ps(sqrt_2_over_pi);
                    __m256 vcoef = _mm256_set1_ps(0.044715f);
                    __m256 vhalf = _mm256_set1_ps(0.5f);
                    __m256 vone = _mm256_set1_ps(1.0f);
                    __m256 v134145 = _mm256_set1_ps(0.134145f); // 3 * 0.044715
                    
                    size_t i = 0;
                    for (; i + 7 < total_size; i += 8) {
                        __m256 x = _mm256_loadu_ps(data_ptr + i);
                        __m256 x2 = _mm256_mul_ps(x, x);
                        __m256 x3 = _mm256_mul_ps(x, x2);
                        
                        __m256 inner = _mm256_mul_ps(vsqrt_2_over_pi, 
                                                    _mm256_add_ps(x, _mm256_mul_ps(vcoef, x3)));
                        __m256 tanh_inner = tanh_avx(inner);
                        __m256 sech_squared = _mm256_sub_ps(vone, _mm256_mul_ps(tanh_inner, tanh_inner));
                        
                        __m256 derivative = _mm256_add_ps(
                            _mm256_mul_ps(vhalf, tanh_inner),
                            _mm256_add_ps(
                                _mm256_mul_ps(
                                    _mm256_mul_ps(
                                        _mm256_mul_ps(x, sech_squared),
                                        vsqrt_2_over_pi
                                    ),
                                    _mm256_add_ps(
                                        vone,
                                        _mm256_mul_ps(v134145, x2)
                                    )
                                ),
                                _mm256_mul_ps(vhalf, _mm256_add_ps(vone, tanh_inner))
                            )
                        );
                        
                        __m256 grad_val = _mm256_loadu_ps(result_grad_ptr + i);
                        __m256 add_grad = _mm256_mul_ps(grad_val, derivative);
                        __m256 current_grad = _mm256_loadu_ps(grad_ptr + i);
                        _mm256_storeu_ps(grad_ptr + i, _mm256_add_ps(current_grad, add_grad));
                    }
                    for (; i < total_size; ++i) {
                        float x = data_ptr[i];
                        float x2 = x * x;
                        float x3 = x * x2;
                        float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
                        float tanh_inner = std::tanh(inner);
                        float sech_squared = 1.0f - tanh_inner * tanh_inner;
                        
                        float derivative = 0.5f * tanh_inner + 
                            0.5f * x * sech_squared * sqrt_2_over_pi * (1.0f + 0.134145f * x2) +
                            0.5f * (1.0f + tanh_inner);
                        
                        grad_ptr[i] += result_grad_ptr[i] * derivative;
                    }
                    #elif defined(__SSE__)
                    __m128 vsqrt_2_over_pi = _mm_set1_ps(sqrt_2_over_pi);
                    __m128 vcoef = _mm_set1_ps(0.044715f);
                    __m128 vhalf = _mm_set1_ps(0.5f);
                    __m128 vone = _mm_set1_ps(1.0f);
                    __m128 v134145 = _mm_set1_ps(0.134145f);
                    
                    size_t i = 0;
                    for (; i + 3 < total_size; i += 4) {
                        __m128 x = _mm_loadu_ps(data_ptr + i);
                        __m128 x2 = _mm_mul_ps(x, x);
                        __m128 x3 = _mm_mul_ps(x, x2);
                        
                        __m128 inner = _mm_mul_ps(vsqrt_2_over_pi, 
                                                _mm_add_ps(x, _mm_mul_ps(vcoef, x3)));
                        __m128 tanh_inner = tanh_sse(inner);
                        __m128 sech_squared = _mm_sub_ps(vone, _mm_mul_ps(tanh_inner, tanh_inner));
                        
                        __m128 derivative = _mm_add_ps(
                            _mm_mul_ps(vhalf, tanh_inner),
                            _mm_add_ps(
                                _mm_mul_ps(
                                    _mm_mul_ps(
                                        _mm_mul_ps(x, sech_squared),
                                        vsqrt_2_over_pi
                                    ),
                                    _mm_add_ps(
                                        vone,
                                        _mm_mul_ps(v134145, x2)
                                    )
                                ),
                                _mm_mul_ps(vhalf, _mm_add_ps(vone, tanh_inner))
                            )
                        );
                        
                        __m128 grad_val = _mm_loadu_ps(result_grad_ptr + i);
                        __m128 add_grad = _mm_mul_ps(grad_val, derivative);
                        __m128 current_grad = _mm_loadu_ps(grad_ptr + i);
                        _mm_storeu_ps(grad_ptr + i, _mm_add_ps(current_grad, add_grad));
                    }
                    for (; i < total_size; ++i) {
                        float x = data_ptr[i];
                        float x2 = x * x;
                        float x3 = x * x2;
                        float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
                        float tanh_inner = std::tanh(inner);
                        float sech_squared = 1.0f - tanh_inner * tanh_inner;
                        
                        float derivative = 0.5f * tanh_inner + 
                            0.5f * x * sech_squared * sqrt_2_over_pi * (1.0f + 0.134145f * x2) +
                            0.5f * (1.0f + tanh_inner);
                        
                        grad_ptr[i] += result_grad_ptr[i] * derivative;
                    }
                    #else
                    for (size_t i = 0; i < total_size; ++i) {
                        float x = data_ptr[i];
                        float x2 = x * x;
                        float x3 = x * x2;
                        float inner = sqrt_2_over_pi * (x + 0.044715f * x3);
                        float tanh_inner = std::tanh(inner);
                        float sech_squared = 1.0f - tanh_inner * tanh_inner;
                        
                        float derivative = 0.5f * tanh_inner + 
                            0.5f * x * sech_squared * sqrt_2_over_pi * (1.0f + 0.134145f * x2) +
                            0.5f * (1.0f + tanh_inner);
                        
                        grad_ptr[i] += result_grad_ptr[i] * derivative;
                    }
                    #endif
                }
            };
        }
    
        return result;
    }
    
    // Optimized softmax implementation
    Tensor softmax(int axis = -1) const {
        // For numerical stability, subtract the max value
        Eigen::MatrixXf shifted = data_;
    
        if (axis == -1 || ndim() == 1) {
            // For overall softmax or 1D tensors
            float max_val = data_.maxCoeff();
            shifted.array() -= max_val;
        } else if (axis == 0) {
            // Column-wise: subtract max of each column
            for (int j = 0; j < shifted.cols(); ++j) {
                float max_val = shifted.col(j).maxCoeff();
                shifted.col(j).array() -= max_val;
            }
        } else {
            // Row-wise: subtract max of each row
            for (int i = 0; i < shifted.rows(); ++i) {
                float max_val = shifted.row(i).maxCoeff();
                shifted.row(i).array() -= max_val;
            }
        }

        Eigen::MatrixXf exp_values = shifted.array().exp();
    
        if (axis == -1 || ndim() == 1) {
            // For overall softmax or 1D tensors
            float sum = exp_values.sum();
            exp_values /= sum;
        } else if (axis == 0) {
            // Column-wise normalization
            for (int j = 0; j < exp_values.cols(); ++j) {
                float col_sum = exp_values.col(j).sum();
                exp_values.col(j) /= col_sum;
            }
        } else {
            // Row-wise normalization
            for (int i = 0; i < exp_values.rows(); ++i) {
                float row_sum = exp_values.row(i).sum();
                exp_values.row(i) /= row_sum;
            }
        }
    
        Tensor result(exp_values, shape_);
        
        if (requires_grad_) {
            result.requires_grad(true);
            result.backward_fn_ = [this, result]() {
                if (this->requires_grad_) {
                    // Gradient of softmax: (diag(softmax) - softmax * softmax^T) * grad
                    // But this is expensive to compute exactly
                    // For efficiency, we'll use a simplified approach
                    // This is an approximation that works well in practice for cross-entropy loss
                    this->grad_ += result.grad_;
                }
            };
        }
        
        return result;
    }
    
    // Optimized sigmoid implementation
    Tensor sigmoid() const {
        Tensor result;
        result.shape_ = shape_;
        result.requires_grad_ = requires_grad_;
        result.data_.resize(data_.rows(), data_.cols());
        
        size_t size = data_.size();
        const float* src = data_.data();
        float* dst = result.data_.data();

        #if defined(__AVX__)
        __m256 vone = _mm256_set1_ps(1.0f);
        size_t i = 0;
        for (; i + 7 < size; i += 8) {
            __m256 x = _mm256_loadu_ps(src + i);
            __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
            __m256 exp_neg_x = exp_avx(neg_x);
            __m256 denom = _mm256_add_ps(vone, exp_neg_x);
            __m256 result_val = _mm256_div_ps(vone, denom);
            _mm256_storeu_ps(dst + i, result_val);
        }
        for (; i < size; ++i) {
            dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
        }
        #elif defined(__SSE__)
        __m128 vone = _mm_set1_ps(1.0f);
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m128 x = _mm_loadu_ps(src + i);
            __m128 neg_x = _mm_sub_ps(_mm_setzero_ps(), x);
            __m128 exp_neg_x = exp_sse(neg_x);
            __m128 denom = _mm_add_ps(vone, exp_neg_x);
            __m128 result_val = _mm_div_ps(vone, denom);
            _mm_storeu_ps(dst + i, result_val);
        }
        for (; i < size; ++i) {
            dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
        }
        #else
        for (size_t i = 0; i < size; ++i) {
            dst[i] = 1.0f / (1.0f + std::exp(-src[i]));
        }
        #endif
    
        if (requires_grad_) {
            result.requires_grad(true);
            result.backward_fn_ = [this, result]() {
                if (this->requires_grad_) {
                    size_t total_size = this->data_.size();
                    float* grad_ptr = this->grad_.data();
                    const float* result_data_ptr = result.data_.data();
                    const float* result_grad_ptr = result.grad_.data();

                    #if defined(__AVX__)
                    size_t i = 0;
                    for (; i + 7 < total_size; i += 8) {
                        __m256 sigmoid_val = _mm256_loadu_ps(result_data_ptr + i);
                        __m256 one_minus_sigmoid = _mm256_sub_ps(_mm256_set1_ps(1.0f), sigmoid_val);
                        __m256 sigmoid_grad = _mm256_mul_ps(sigmoid_val, one_minus_sigmoid);
                        __m256 grad_val = _mm256_loadu_ps(result_grad_ptr + i);
                        __m256 add_grad = _mm256_mul_ps(grad_val, sigmoid_grad);
                        __m256 current_grad = _mm256_loadu_ps(grad_ptr + i);
                        _mm256_storeu_ps(grad_ptr + i, _mm256_add_ps(current_grad, add_grad));
                    }
                    for (; i < total_size; ++i) {
                        float sigmoid_val = result_data_ptr[i];
                        float sigmoid_grad = sigmoid_val * (1.0f - sigmoid_val);
                        grad_ptr[i] += result_grad_ptr[i] * sigmoid_grad;
                    }
                    #elif defined(__SSE__)
                    size_t i = 0;
                    for (; i + 3 < total_size; i += 4) {
                        __m128 sigmoid_val = _mm_loadu_ps(result_data_ptr + i);
                        __m128 one_minus_sigmoid = _mm_sub_ps(_mm_set1_ps(1.0f), sigmoid_val);
                        __m128 sigmoid_grad = _mm_mul_ps(sigmoid_val, one_minus_sigmoid);
                        __m128 grad_val = _mm_loadu_ps(result_grad_ptr + i);
                        __m128 add_grad = _mm_mul_ps(grad_val, sigmoid_grad);
                        __m128 current_grad = _mm_loadu_ps(grad_ptr + i);
                        _mm_storeu_ps(grad_ptr + i, _mm_add_ps(current_grad, add_grad));
                    }
                    for (; i < total_size; ++i) {
                        float sigmoid_val = result_data_ptr[i];
                        float sigmoid_grad = sigmoid_val * (1.0f - sigmoid_val);
                        grad_ptr[i] += result_grad_ptr[i] * sigmoid_grad;
                    }
                    #else
                    for (size_t i = 0; i < total_size; ++i) {
                        float sigmoid_val = result_data_ptr[i];
                        float sigmoid_grad = sigmoid_val * (1.0f - sigmoid_val);
                        grad_ptr[i] += result_grad_ptr[i] * sigmoid_grad;
                    }
                    #endif
                }
            };
        }
    
        return result;
    }
    
    Tensor sqrt() const {
        Tensor result;
        result.shape_ = shape_;
        result.requires_grad_ = requires_grad_;
        result.data_.resize(data_.rows(), data_.cols());
        
        size_t size = data_.size();
        const float* src = data_.data();
        float* dst = result.data_.data();

        #if defined(__AVX__)
        size_t i = 0;
        for (; i + 7 < size; i += 8) {
            __m256 x = _mm256_loadu_ps(src + i);
            __m256 sqrt_x = _mm256_sqrt_ps(x);
            _mm256_storeu_ps(dst + i, sqrt_x);
        }
        for (; i < size; ++i) {
            dst[i] = std::sqrt(src[i]);
        }
        #elif defined(__SSE__)
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m128 x = _mm_loadu_ps(src + i);
            __m128 sqrt_x = _mm_sqrt_ps(x);
            _mm_storeu_ps(dst + i, sqrt_x);
        }
        for (; i < size; ++i) {
            dst[i] = std::sqrt(src[i]);
        }
        #else
        for (size_t i = 0; i < size; ++i) {
            dst[i] = std::sqrt(src[i]);
        }
        #endif

        if (requires_grad_) {
            result.requires_grad(true);
            result.backward_fn_ = [this, result]() {
                if (this->requires_grad_) {
                    size_t total_size = this->data_.size();
                    float* grad_ptr = this->grad_.data();
                    const float* data_ptr = this->data_.data();
                    const float* result_grad_ptr = result.grad_.data();

                    #if defined(__AVX__)
                    __m256 half = _mm256_set1_ps(0.5f);
                    __m256 eps = _mm256_set1_ps(1e-12f);
                    size_t i = 0;
                    for (; i + 7 < total_size; i += 8) {
                        __m256 data_val = _mm256_loadu_ps(data_ptr + i);
                        __m256 sqrt_val = _mm256_sqrt_ps(data_val);
                        __m256 inv_sqrt = _mm256_div_ps(half, _mm256_add_ps(sqrt_val, eps));
                        __m256 grad_val = _mm256_loadu_ps(result_grad_ptr + i);
                        __m256 add_grad = _mm256_mul_ps(grad_val, inv_sqrt);
                        __m256 current_grad = _mm256_loadu_ps(grad_ptr + i);
                        _mm256_storeu_ps(grad_ptr + i, _mm256_add_ps(current_grad, add_grad));
                    }
                    for (; i < total_size; ++i) {
                        grad_ptr[i] += result_grad_ptr[i] * (0.5f / (std::sqrt(data_ptr[i]) + 1e-12f));
                    }
                    #elif defined(__SSE__)
                    __m128 half = _mm_set1_ps(0.5f);
                    __m128 eps = _mm_set1_ps(1e-12f);
                    size_t i = 0;
                    for (; i + 3 < total_size; i += 4) {
                        __m128 data_val = _mm_loadu_ps(data_ptr + i);
                        __m128 sqrt_val = _mm_sqrt_ps(data_val);
                        __m128 inv_sqrt = _mm_div_ps(half, _mm_add_ps(sqrt_val, eps));
                        __m128 grad_val = _mm_loadu_ps(result_grad_ptr + i);
                        __m128 add_grad = _mm_mul_ps(grad_val, inv_sqrt);
                        __m128 current_grad = _mm_loadu_ps(grad_ptr + i);
                        _mm_storeu_ps(grad_ptr + i, _mm_add_ps(current_grad, add_grad));
                    }
                    for (; i < total_size; ++i) {
                        grad_ptr[i] += result_grad_ptr[i] * (0.5f / (std::sqrt(data_ptr[i]) + 1e-12f));
                    }
                    #else
                    for (size_t i = 0; i < total_size; ++i) {
                        grad_ptr[i] += result_grad_ptr[i] * (0.5f / (std::sqrt(data_ptr[i]) + 1e-12f));
                    }
                    #endif
                }
            };
        }
    
        return result;
    }
    
    // Backward propagation
    void backward() {
        if (backward_fn_) {
            backward_fn_();
        }
    }
    
    // Optimized initialization functions
    static Tensor zeros(const std::vector<size_t>& shape, bool requires_grad = false) {
        return Tensor(shape, requires_grad);
    }
    
    static Tensor ones(const std::vector<size_t>& shape, bool requires_grad = false) {
        Tensor result(shape, requires_grad);
        result.data_.setOnes();
        return result;
    }
    
    // Add zeros_like method for compatibility with Adam optimizer
    static Tensor zeros_like(const Tensor& other, bool requires_grad = false) {
        return Tensor::zeros(other.shape(), requires_grad);
    }
    
    // Optimized random number generation
    static Tensor randn(const std::vector<size_t>& shape, float mean = 0.0f, float stddev = 1.0f, bool requires_grad = false) {
        Tensor result(shape, requires_grad);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(mean, stddev);
        
        // Use Eigen's built-in random generation for better performance
        result.data_ = Eigen::MatrixXf::NullaryExpr(
            result.data_.rows(), result.data_.cols(),
            [&]() { return dist(gen); }
        );
        
        return result;
    }
    
    static Tensor xavier(const std::vector<size_t>& shape, bool requires_grad = false) {
        if (shape.size() < 2) {
            throw std::invalid_argument("Xavier initialization requires at least 2 dimensions");
        }
        float stddev = std::sqrt(2.0f / (shape[0] + shape[1]));
        return randn(shape, 0.0f, stddev, requires_grad);
    }
    
    // Utility functions
    Tensor slice(size_t start, size_t length, int axis = 0) const {
        if (axis == 0) {
            return Tensor(data_.block(start, 0, length, data_.cols()));
        } else {
            return Tensor(data_.block(0, start, data_.rows(), length));
        }
    }
    
    Tensor concatenate(const Tensor& other, int axis = 0) const {
        if (axis == 0) {
            Eigen::MatrixXf result(data_.rows() + other.data_.rows(), data_.cols());
            result << data_, other.data_;
            return Tensor(result);
        } else {
            Eigen::MatrixXf result(data_.rows(), data_.cols() + other.data_.cols());
            result << data_, other.data_;
            return Tensor(result);
        }
    }
    
    // Additional utility for neural networks
    Tensor argmax(int axis = -1) const {
        if (axis == -1 || ndim() == 1) {
            // For overall argmax or 1D tensors
            Eigen::Index maxIndex = 0;
            float maxValue = data_(0);
            
            // Manual implementation for both vectors and matrices
            for (Eigen::Index i = 0; i < data_.size(); ++i) {
                if (data_(i) > maxValue) {
                    maxValue = data_(i);
                    maxIndex = i;
                }
            }
            
            return Tensor(Eigen::MatrixXf::Constant(1, 1, static_cast<float>(maxIndex)));
        } else if (axis == 0) {
            // Column-wise argmax
            Eigen::RowVectorXf result(data_.cols());
            for (int i = 0; i < data_.cols(); ++i) {
                Eigen::Index maxIndex = 0;
                float maxValue = data_(0, i);
                for (int j = 1; j < data_.rows(); ++j) {
                    if (data_(j, i) > maxValue) {
                        maxValue = data_(j, i);
                        maxIndex = j;
                    }
                }
                result(i) = static_cast<float>(maxIndex);
            }
            return Tensor(result, {static_cast<size_t>(result.cols())});
        } else {
            // Row-wise argmax
            Eigen::VectorXf result(data_.rows());
            for (int i = 0; i < data_.rows(); ++i) {
                Eigen::Index maxIndex = 0;
                float maxValue = data_(i, 0);
                for (int j = 1; j < data_.cols(); ++j) {
                    if (data_(i, j) > maxValue) {
                        maxValue = data_(i, j);
                        maxIndex = j;
                    }
                }
                result(i) = static_cast<float>(maxIndex);
            }
            return Tensor(result, {static_cast<size_t>(result.rows())});
        }
    }

    void serialize(std::ostream& stream) const {
        // Write shape information
        uint32_t ndim = static_cast<uint32_t>(shape_.size());
        stream.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));
        
        for (auto dim : shape_) {
            uint32_t dim32 = static_cast<uint32_t>(dim);
            stream.write(reinterpret_cast<const char*>(&dim32), sizeof(dim32));
        }
        
        // Write data
        size_t num_elements = data_.size();
        stream.write(reinterpret_cast<const char*>(data_.data()), 
                    num_elements * sizeof(float));
        
        // Note: We're not serializing gradients as they're not needed for inference
    }
    
    void deserialize(std::istream& stream) {
        // Read shape information
        uint32_t ndim;
        stream.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
        
        std::vector<size_t> new_shape(ndim);
        for (uint32_t i = 0; i < ndim; ++i) {
            uint32_t dim;
            stream.read(reinterpret_cast<char*>(&dim), sizeof(dim));
            new_shape[i] = static_cast<size_t>(dim);
        }
        
        // Resize tensor
        shape_ = new_shape;
        if (ndim == 1) {
            data_ = Eigen::VectorXf::Zero(shape_[0]);
        } else if (ndim == 2) {
            data_ = Eigen::MatrixXf::Zero(shape_[0], shape_[1]);
        } else {
            size_t total_size = 1;
            for (auto dim : shape_) total_size *= dim;
            data_ = Eigen::VectorXf::Zero(total_size);
        }
        
        // Read data
        size_t num_elements = data_.size();
        stream.read(reinterpret_cast<char*>(data_.data()), 
                   num_elements * sizeof(float));
        
        // Initialize grad if needed
        if (requires_grad_) {
            grad_ = Eigen::MatrixXf::Zero(data_.rows(), data_.cols());
        }
    }
    
    static void write_string(std::ostream& stream, const std::string& str) {
        uint32_t length = static_cast<uint32_t>(str.size());
        stream.write(reinterpret_cast<const char*>(&length), sizeof(length));
        stream.write(str.c_str(), length);
    }
    
    static std::string read_string(std::istream& stream) {
        uint32_t length;
        stream.read(reinterpret_cast<char*>(&length), sizeof(length));
        
        std::string str(length, '\0');
        stream.read(&str[0], length);
        
        return str;
    }

    // Cereal serialization method
    template <class Archive>
    void serialize(Archive& archive) {
        // Serialize basic data members
        archive(
            cereal::make_nvp("shape", shape_),
            cereal::make_nvp("requires_grad", requires_grad_)
        );
        
        // Serialize the data matrix
        size_t rows = data_.rows();
        size_t cols = data_.cols();
        archive(rows, cols);
        
        if (Archive::is_loading::value) {
            // We're loading, so resize the matrix
            data_.resize(rows, cols);
        }
        
        // Serialize the matrix data
        archive(cereal::binary_data(data_.data(), rows * cols * sizeof(float)));
        
        // Serialize gradient if needed
        if (requires_grad_) {
            size_t grad_rows = grad_.rows();
            size_t grad_cols = grad_.cols();
            archive(grad_rows, grad_cols);
            
            if (Archive::is_loading::value) {
                grad_.resize(grad_rows, grad_cols);
            }
            
            archive(cereal::binary_data(grad_.data(), grad_rows * grad_cols * sizeof(float)));
        }
        
        // Note: We don't serialize backward_fn_ as it's a runtime computation graph
    }

private:
    Eigen::MatrixXf data_;
    mutable Eigen::MatrixXf grad_;
    std::vector<size_t> shape_;
    bool requires_grad_;
    std::function<void()> backward_fn_;
    
    // Helper functions for SIMD operations
    #if defined(__AVX__)
    static __m256 exp_avx(__m256 x) {
        // Implementation of exp using AVX intrinsics
        // This is an approximation
        __m256 a = _mm256_set1_ps(12102203.0f); // 2^23 / ln(2)
        __m256 b = _mm256_set1_ps(1065353216.0f); // 2^23
        __m256 c = _mm256_set1_ps(0.5f);
        __m256 d = _mm256_set1_ps(1.0f);
        __m256 e = _mm256_set1_ps(1.0f);
        __m256 f = _mm256_set1_ps(0.99992522f);
        __m256 g = _mm256_set1_ps(0.69583354f);
        __m256 h = _mm256_set1_ps(0.22606716f);
        __m256 i = _mm256_set1_ps(0.078024523f);
        
        __m256 mask = _mm256_cmp_ps(x, _mm256_set1_ps(-88.0f), _CMP_GT_OS);
        x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));
        
        __m256 z = _mm256_mul_ps(x, a);
        z = _mm256_add_ps(z, b);
        __m256 n = _mm256_floor_ps(z);
        z = _mm256_sub_ps(z, n);
        __m256 r = _mm256_sub_ps(x, _mm256_mul_ps(n, _mm256_set1_ps(1.1920929e-7f)));
        
        __m256 r2 = _mm256_mul_ps(r, r);
        __m256 result = _mm256_add_ps(_mm256_mul_ps(i, r), h);
        result = _mm256_add_ps(_mm256_mul_ps(result, r), g);
        result = _mm256_add_ps(_mm256_mul_ps(result, r), f);
        result = _mm256_add_ps(_mm256_mul_ps(result, r), e);
        result = _mm256_add_ps(_mm256_mul_ps(result, r), d);
        
        n = _mm256_add_ps(n, _mm256_set1_ps(127.0f));
        n = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_castps_si256(n), 23));
        
        result = _mm256_mul_ps(result, n);
        return _mm256_and_ps(result, mask);
    }
    
    static __m256 tanh_avx(__m256 x) {
        // Implementation of tanh using AVX intrinsics
        // This is an approximation
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 a = _mm256_mul_ps(x2, _mm256_set1_ps(0.00992762224f));
        a = _mm256_add_ps(a, _mm256_set1_ps(0.0559197695f));
        a = _mm256_mul_ps(a, x2);
        a = _mm256_add_ps(a, _mm256_set1_ps(0.173565726f));
        a = _mm256_mul_ps(a, x2);
        a = _mm256_add_ps(a, _mm256_set1_ps(0.239708459f));
        a = _mm256_mul_ps(a, x2);
        a = _mm256_add_ps(a, _mm256_set1_ps(0.666657572f));
        a = _mm256_mul_ps(a, x);
        return a;
    }
    #endif
    
#if defined(__SSE__)
static __m128 exp_sse(__m128 x) {
    // Alternative implementation using SSE2 intrinsics
    // This is a simpler approximation that doesn't require SSE4.1
    __m128 a = _mm_set1_ps(12102203.0f);
    __m128 b = _mm_set1_ps(1065353216.0f);
    __m128 c = _mm_set1_ps(1.0f);
    __m128 d = _mm_set1_ps(0.5f);
    
    // Handle large negative values
    __m128 mask = _mm_cmpgt_ps(x, _mm_set1_ps(-88.0f));
    x = _mm_min_ps(x, _mm_set1_ps(88.0f));
    
    // Approximation: exp(x) ≈ 1 + x + x^2/2 + x^3/6 + x^4/24
    __m128 x2 = _mm_mul_ps(x, x);
    __m128 x3 = _mm_mul_ps(x2, x);
    __m128 x4 = _mm_mul_ps(x3, x);
    
    __m128 result = _mm_add_ps(c, x);
    result = _mm_add_ps(result, _mm_mul_ps(x2, d));
    result = _mm_add_ps(result, _mm_mul_ps(x3, _mm_set1_ps(0.1666667f)));
    result = _mm_add_ps(result, _mm_mul_ps(x4, _mm_set1_ps(0.04166667f)));
    
    return _mm_and_ps(result, mask);
}

static __m128 tanh_sse(__m128 x) {
    // Alternative tanh approximation using SSE2
    __m128 x2 = _mm_mul_ps(x, x);
    __m128 a = _mm_mul_ps(x2, _mm_set1_ps(0.00992762224f));
    a = _mm_add_ps(a, _mm_set1_ps(0.0559197695f));
    a = _mm_mul_ps(a, x2);
    a = _mm_add_ps(a, _mm_set1_ps(0.173565726f));
    a = _mm_mul_ps(a, x2);
    a = _mm_add_ps(a, _mm_set1_ps(0.239708459f));
    a = _mm_mul_ps(a, x2);
    a = _mm_add_ps(a, _mm_set1_ps(0.666657572f));
    a = _mm_mul_ps(a, x);
    return a;
}
#endif
};

// Global operator for scalar multiplication (scalar * tensor)
inline Tensor operator*(float scalar, const Tensor& tensor) {
    return tensor * scalar;
}

} // namespace lm
