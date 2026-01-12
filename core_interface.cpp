// Add to your core_interface.cpp

int run_inference(int model_id, int input_tensor_id) {
    // 1. Get the model and tensor from our registries
    auto& model = g_model_registry[model_id];
    auto& input_tensor = g_tensor_registry[input_tensor_id];

    // 2. Execute the model
    // No-grad is important for performance during inference
    torch::NoGradGuard no_grad;
    
    // We wrap the input in a vector because LibTorch expects a list of inputs
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    // Run forward pass
    torch::Tensor output = model.forward(inputs).toTensor();

    // 3. Register the new output tensor and return its ID
    return deepvex::register_tensor(output);
}

// Add this to your PYBIND11_MODULE block:
m.def("run_inference", &run_inference, "Executes a model forward pass using registered IDs");

// Add to your core_interface.cpp

// NEW: Temporal Registry (Memory that survives across frames)
std::map<std::string, int> g_temporal_registry;

void set_temporal_state(std::string name, int tensor_id) {
    g_temporal_registry[name] = tensor_id;
}

int get_temporal_state(std::string name) {
    if (g_temporal_registry.count(name)) {
        return g_temporal_registry[name];
    }
    return -1; // Not found
}

// Add these to your PYBIND11_MODULE block:
m.def("set_temporal_state", &set_temporal_state);
m.def("get_temporal_state", &get_temporal_state);
