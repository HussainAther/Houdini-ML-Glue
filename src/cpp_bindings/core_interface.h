#pragma once
#include <torch/torch.h>
#include <map>
#include <vector>
#include <string>

// The Registry: A global map to keep tensors alive between Houdini nodes
// Key: Tensor ID (int) | Value: The actual Torch Tensor
extern std::map<int, torch::Tensor> g_tensor_registry;
extern int g_next_tensor_id;

namespace deepvex {
    // Core Functions
    int register_tensor(torch::Tensor t);
    torch::Tensor get_tensor(int id);
    bool remove_tensor(int id);
    
    // Data Conversion
    int create_from_numpy(std::vector<std::vector<float>> data_list);
}
