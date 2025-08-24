#pragma once

#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>

namespace lm {

class Tensor {
public:
    Tensor() : data_(Eigen::MatrixXf(0, 0)), shape_({0}) {}
    
    Tensor(const std::vector<size_t>& shape) {
        shape_ = shape;
        if (shape.size() == 1) {
            data_ = Eigen::VectorXf::Zero(shape[0]);
        } else if (shape.size() == 2) {
            data_ = Eigen::MatrixXf::Zero(shape[0], shape[1]);
        } else {
            // For higher dimensions, we'll flatten and handle with care
            size_t total_size = 1;
            for (auto dim : shape) total_size *= dim;
            data_ = Eigen::VectorXf::Zero(total_size);
        }
    }
    
    Tensor(const Eigen::MatrixXf& data, const std::vector<size_t>& shape = {})
        : data_(data), shape_(shape) {
        if (shape.empty()) {
            if (data.cols() == 1) {
                shape_ = {static_cast<size_t>(data.rows())};
            } else {
                shape_ = {static_cast<size_t>(data.rows()), 
                         static_cast<size_t>(data.cols())};
            }
        }
    }
    
    // Accessors
    const std::vector<size_t>& shape() const { return shape_; }
    Eigen::MatrixXf& data() { return data_; }
    const Eigen::MatrixXf& data() const { return data_; }
    
    // Element access
    float& operator()(size_t i) { return data_(i); }
    float operator()(size_t i) const { return data_(i); }
    float& operator()(size_t i, size_t j) { return data_(i, j); }
    float operator()(size_t i, size_t j) const { return data_(i, j); }

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
        
        return Tensor(data_, new_shape);
    }
    
    // Mathematical operations
    Tensor operator+(const Tensor& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor shapes must match for addition");
        }
        return Tensor(data_ + other.data_, shape_);
    }
    
    Tensor operator-(const Tensor& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor shapes must match for subtraction");
        }
        return Tensor(data_ - other.data_, shape_);
    }
    
    Tensor operator*(const Tensor& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor shapes must match for element-wise multiplication");
        }
        return Tensor(data_.cwiseProduct(other.data_), shape_);
    }
    
    Tensor operator/(const Tensor& other) const {
        if (shape_ != other.shape_) {
            throw std::invalid_argument("Tensor shapes must match for element-wise division");
        }
        return Tensor(data_.cwiseQuotient(other.data_), shape_);
    }
    
    Tensor operator+(float scalar) const {
        return Tensor(data_.array() + scalar, shape_);
    }
    
    Tensor operator-(float scalar) const {
        return Tensor(data_.array() - scalar, shape_);
    }
    
    Tensor operator*(float scalar) const {
        return Tensor(data_ * scalar, shape_);
    }
    
    Tensor operator/(float scalar) const {
        return Tensor(data_ / scalar, shape_);
    }
    
    Tensor matmul(const Tensor& other) const {
        if (ndim() != 2 || other.ndim() != 2) {
            throw std::invalid_argument("matmul requires 2D tensors");
        }
        if (shape_[1] != other.shape_[0]) {
            throw std::invalid_argument("Incompatible dimensions for matrix multiplication");
        }
        
        return Tensor(data_ * other.data_, {shape_[0], other.shape()[1]});
    }
    
    Tensor transpose() const {
        if (ndim() != 2) {
            throw std::invalid_argument("transpose requires 2D tensors");
        }
        return Tensor(data_.transpose(), {shape_[1], shape_[0]});
    }
    
    // Reduction operations
    Tensor sum(int axis = -1) const {
        if (axis == -1 || ndim() == 1) {
            return Tensor(Eigen::MatrixXf::Constant(1, 1, data_.sum()));
        } else if (axis == 0) {
            return Tensor(data_.colwise().sum(), {shape_[1]});
        } else {
            return Tensor(data_.rowwise().sum(), {shape_[0]});
        }
    }
    
    Tensor mean(int axis = -1) const {
        if (axis == -1 || ndim() == 1) {
            return Tensor(Eigen::MatrixXf::Constant(1, 1, data_.mean()));
        } else if (axis == 0) {
            return Tensor(data_.colwise().mean(), {shape_[1]});
        } else {
            return Tensor(data_.rowwise().mean(), {shape_[0]});
        }
    }
    
    Tensor max(int axis = -1) const {
        if (axis == -1 || ndim() == 1) {
            return Tensor(Eigen::MatrixXf::Constant(1, 1, data_.maxCoeff()));
        } else if (axis == 0) {
            Eigen::RowVectorXf result = data_.colwise().maxCoeff();
            return Tensor(result, {static_cast<size_t>(result.cols())});
        } else {
            Eigen::VectorXf result = data_.rowwise().maxCoeff();
            return Tensor(result, {static_cast<size_t>(result.rows())});
        }
    }
    
    Tensor min(int axis = -1) const {
        if (axis == -1 || ndim() == 1) {
            return Tensor(Eigen::MatrixXf::Constant(1, 1, data_.minCoeff()));
        } else if (axis == 0) {
            Eigen::RowVectorXf result = data_.colwise().minCoeff();
            return Tensor(result, {static_cast<size_t>(result.cols())});
        } else {
            Eigen::VectorXf result = data_.rowwise().minCoeff();
            return Tensor(result, {static_cast<size_t>(result.rows())});
        }
    }
    
    // Activation functions
    Tensor relu() const {
        return Tensor(data_.cwiseMax(0.0f), shape_);
    }
    
    Tensor gelu() const {
        // Approximation of GELU: x * 0.5 * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float sqrt_2_over_pi = std::sqrt(2.0f / M_PI);
        Eigen::ArrayXf x_array = data_.array();
        Eigen::ArrayXf result = 0.5f * x_array * 
            (1.0f + (sqrt_2_over_pi * (x_array + 0.044715f * x_array.pow(3))).tanh());
        return Tensor(Eigen::MatrixXf(result), shape_);
    }
    
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
    
        return Tensor(exp_values, shape_);
    }
    
    Tensor sigmoid() const {
        Eigen::ArrayXf x_array = data_.array();
        Eigen::ArrayXf result = 1.0f / (1.0f + (-x_array).exp());
        return Tensor(Eigen::MatrixXf(result), shape_);
    }
    
    // Initialization
    static Tensor zeros(const std::vector<size_t>& shape) {
        return Tensor(shape);
    }
    
    static Tensor ones(const std::vector<size_t>& shape) {
        Tensor result(shape);
        result.data_.setOnes();
        return result;
    }
    
    static Tensor randn(const std::vector<size_t>& shape, float mean = 0.0f, float stddev = 1.0f) {
        Tensor result(shape);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(mean, stddev);
        
        for (int i = 0; i < result.data_.rows(); ++i) {
            for (int j = 0; j < result.data_.cols(); ++j) {
                result.data_(i, j) = dist(gen);
            }
        }
        
        return result;
    }
    
    static Tensor xavier(const std::vector<size_t>& shape) {
        if (shape.size() < 2) {
            throw std::invalid_argument("Xavier initialization requires at least 2 dimensions");
        }
        float stddev = std::sqrt(2.0f / (shape[0] + shape[1]));
        return randn(shape, 0.0f, stddev);
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

private:
    Eigen::MatrixXf data_;
    std::vector<size_t> shape_;
};

} // namespace lm

