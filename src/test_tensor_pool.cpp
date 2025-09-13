// src/test_tensor_pool.cpp
#include <lm/core/tensor_pool.hpp>
#include <lm/core/tensor.hpp>
#include <iostream>
#include <vector>
#include <memory>

int main() {
    std::cout << "Testing TensorPool functionality..." << std::endl;
    
    // Create a tensor pool
    lm::TensorPool pool;
    
    std::cout << "Initial pool size: " << pool.size() << std::endl;
    
    // Test 1: Acquire a tensor and use it
    std::cout << "\n=== Test 1: Acquire and use a tensor ===" << std::endl;
    auto tensor1 = pool.acquire({128, 128}, true);
    std::cout << "Acquired tensor with shape: [";
    for (auto dim : tensor1->shape()) {
        std::cout << dim << ", ";
    }
    std::cout << "], requires_grad: " << tensor1->requires_grad() << std::endl;
    
    // Use the tensor
    tensor1->data().setConstant(5.0f);
    std::cout << "Tensor data[0][0]: " << tensor1->data()(0, 0) << std::endl;
    
    // Test 2: Release the tensor back to the pool
    std::cout << "\n=== Test 2: Release tensor back to pool ===" << std::endl;
    pool.release(std::move(tensor1));
    std::cout << "Pool size after release: " << pool.size() << std::endl;
    
    // Test 3: Acquire another tensor with the same specs (should reuse)
    std::cout << "\n=== Test 3: Acquire tensor with same specs (should reuse) ===" << std::endl;
    auto tensor2 = pool.acquire({128, 128}, true);
    std::cout << "Acquired tensor with shape: [";
    for (auto dim : tensor2->shape()) {
        std::cout << dim << ", ";
    }
    std::cout << "], requires_grad: " << tensor2->requires_grad() << std::endl;
    std::cout << "Pool size after acquisition: " << pool.size() << std::endl;
    
    // Test 4: Verify the tensor was reset (should be zeros)
    std::cout << "\n=== Test 4: Verify tensor was reset ===" << std::endl;
    std::cout << "Tensor data[0][0] (should be 0): " << tensor2->data()(0, 0) << std::endl;
    
    // Test 5: Acquire a tensor with different specs (should create new)
    std::cout << "\n=== Test 5: Acquire tensor with different specs (should create new) ===" << std::endl;
    auto tensor3 = pool.acquire({64, 64}, false);
    std::cout << "Acquired tensor with shape: [";
    for (auto dim : tensor3->shape()) {
        std::cout << dim << ", ";
    }
    std::cout << "], requires_grad: " << tensor3->requires_grad() << std::endl;
    std::cout << "Pool size after acquisition: " << pool.size() << std::endl;
    
    // Test 6: Release both tensors
    std::cout << "\n=== Test 6: Release both tensors ===" << std::endl;
    pool.release(std::move(tensor2));
    pool.release(std::move(tensor3));
    std::cout << "Pool size after releasing both: " << pool.size() << std::endl;
    
    // Test 7: Clear the pool
    std::cout << "\n=== Test 7: Clear the pool ===" << std::endl;
    pool.clear();
    std::cout << "Pool size after clear: " << pool.size() << std::endl;
    
    // Test 8: Test with multiple tensors
    std::cout << "\n=== Test 8: Test with multiple tensors ===" << std::endl;
    std::vector<std::unique_ptr<lm::Tensor>> tensors;
    for (int i = 0; i < 5; i++) {
        tensors.push_back(pool.acquire({32, 32}, true));
        std::cout << "Acquired tensor " << i+1 << ", pool size: " << pool.size() << std::endl;
    }
    
    // Release all tensors
    for (auto& tensor : tensors) {
        pool.release(std::move(tensor));
    }
    std::cout << "Released all tensors, pool size: " << pool.size() << std::endl;
    
    std::cout << "\n=== All tests completed successfully! ===" << std::endl;
    
    return 0;
}
