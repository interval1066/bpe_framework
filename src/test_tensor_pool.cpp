#include <lm/core/tensor_pool.hpp>
#include <iostream>

int main() {
    lm::TensorPool pool;
    
    // Test acquiring and releasing tensors
    auto tensor1 = pool.acquire({128, 128}, true);
    std::cout << "Acquired tensor with shape: [";
    for (auto dim : tensor1->shape()) {
        std::cout << dim << ", ";
    }
    std::cout << "]" << std::endl;
    
    pool.release(std::move(tensor1));
    
    // Test reusing the same tensor
    auto tensor2 = pool.acquire({128, 128}, true);
    std::cout << "Reused tensor from pool" << std::endl;
    
    std::cout << "Pool size: " << pool.size() << std::endl;
    
    return 0;
}

