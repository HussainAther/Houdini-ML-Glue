#include "core_interface.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

std::map<int, torch::Tensor> g_tensor_registry;
int g_next_tensor_id = 0;

int deepvex::register_tensor(torch::Tensor t) {
    int id = g_next_tensor_id++;
    g_tensor_registry[id] = t;
    return id;
}

torch::Tensor deepvex::get_tensor(int id) {
    if (g_tensor_registry.count(id)) {
        return g_tensor_registry[id];
    }
    throw std::runtime_error("Tensor ID not found in Registry!");
}

bool deepvex::remove_tensor(int id) {
    return g_tensor_registry.erase(id) > 0;
}

// THE PYBIND11 MODULE - This makes the functions visible in Houdini's Python
PYBIND11_MODULE(deepvex_bindings, m) {
    m.doc() = "DeepVEX C++ Backend for Houdini";
    
    m.def("create_tensor_from_arrays", [](std::vector<std::vector<float>> data) {
        // Simple example: convert flat vector list to a 2D Tensor
        // In a real PR, someone could optimize this for CUDA memory directly!
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        
        // This is a placeholder for the logic that stacks your Houdini attributes
        torch::Tensor t = torch::ones({(long)data.size(), (long)data[0].size()}, options); 
        
        return deepvex::register_tensor(t);
    }, "Converts Houdini attribute arrays into a registered PyTorch Tensor");

    m.def("get_tensor_by_id", &deepvex::get_tensor);
    m.def("delete_tensor_by_id", &deepvex::remove_tensor);
}

// Add to core_interface.cpp

int run_feedback_inference(int current_tensor_id, int previous_state_id) {
    // 1. Retrieve both tensors from the Registry
    auto& current_t = g_tensor_registry[current_tensor_id];
    auto& previous_t = g_tensor_registry[previous_state_id];

    // 2. Prepare for Inference
    torch::NoGradGuard no_grad;
    
    // 3. Concatenate Tensors (Temporal Stacking)
    // We stack them along the feature dimension (dim 1)
    // If 'P' is (N, 3), the combined tensor becomes (N, 6)
    torch::Tensor combined_input = torch::cat({current_t, previous_t}, 1);

    // 4. Get the Model (Assuming we use the last loaded model for simplicity)
    // In a complex sim, you'd pass a model_id here too
    auto& model = g_model_registry.rbegin()->second; 

    // 5. Forward Pass
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(combined_input.to(torch::kCUDA)); // Move to GPU for speed
    
    torch::Tensor result = model.forward(inputs).toTensor();

    // 6. Return the new state ID
    return deepvex::register_tensor(result.to(torch::kCPU));
}

// Add this to your PYBIND11_MODULE block:
m.def("run_feedback_inference", &run_feedback_inference, "Runs inference using current and temporal tensors");
