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
