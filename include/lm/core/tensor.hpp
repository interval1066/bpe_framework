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
        Eigen::MatrixXf exp_values = data_.array().exp();
        if (axis == 0 || ndim() == 1) {
            exp_values.array().colwise() /= exp_values.colwise().sum().array();
        } else {
            exp_values.array().rowwise() /= exp_values.rowwise().sum().array();
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
            Eigen::Index maxIndex;
            if (ndim() == 1) {
                // This is a vector, so we can use maxCoeff with index
                data_.maxCoeff(&maxIndex);
            } else {
                // For matrices, we need to flatten first
                Eigen::VectorXf flattened = Eigen::Map<const Eigen::VectorXf>(data_.data(), data_.size());
                flattened.maxCoeff(&maxIndex);
            }
            return Tensor(Eigen::MatrixXf::Constant(1, 1, static_cast<float>(maxIndex)));
        } else if (axis == 0) {
            // Column-wise argmax
            Eigen::RowVectorXf result(data_.cols());
            for (int i = 0; i < data_.cols(); ++i) {
                Eigen::Index maxIndex;
                data_.col(i).maxCoeff(&maxIndex);
                result(i) = static_cast<float>(maxIndex);
            }
            return Tensor(result, {static_cast<size_t>(result.cols())});
        } else {
            // Row-wise argmax
            Eigen::VectorXf result(data_.rows());
            for (int i = 0; i < data_.rows(); ++i) {
                Eigen::Index maxIndex;
                data_.row(i).maxCoeff(&maxIndex);
                result(i) = static_cast<float>(maxIndex);
            }
            return Tensor(result, {static_cast<size_t>(result.rows())});
        }
    }

	// 3D element access
	float& operator()(size_t i, size_t j, size_t k) {
	    // Calculate the index in the flattened data
	    size_t index = i * dim(1) * dim(2) + j * dim(2) + k;
	    return data()(index);
	}

	float operator()(size_t i, size_t j, size_t k) const {
	    // Calculate the index in the flattened data
	    size_t index = i * dim(1) * dim(2) + j * dim(2) + k;
	    return data()(index);
	}

private:
    Eigen::MatrixXf data_;
    std::vector<size_t> shape_;
};

} // namespace lm

