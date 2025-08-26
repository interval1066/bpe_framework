// test_transformer.cpp
#include "lm/models/transformer.hpp"

int main() {
    lm::Transformer transformer(1000, 512, 8, 2048, 6, 512);
    return 0;
}

