#pragma once

#include <Eigen/Dense>
#include <cereal/cereal.hpp>

namespace cereal {

// Serialization for Eigen matrices
template <class Archive, typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
void serialize(Archive& archive, Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& matrix) {
    int rows = matrix.rows();
    int cols = matrix.cols();
    archive(rows, cols);
    
    if (rows * cols != matrix.size()) {
        matrix.resize(rows, cols);
    }
    
    archive(binary_data(matrix.data(), rows * cols * sizeof(Scalar)));
}

// Serialization for Eigen vectors
template <class Archive, typename Scalar, int Size, int Options, int MaxSize>
void serialize(Archive& archive, Eigen::Matrix<Scalar, Size, 1, Options, MaxSize, 1>& vector) {
    int size = vector.size();
    archive(size);
    
    if (size != vector.size()) {
        vector.resize(size);
    }
    
    archive(binary_data(vector.data(), size * sizeof(Scalar)));
}

} // namespace cereal

