# Houdini ML Glue

```
/Houdini-ML-Glue
|
├── /docs                         # Technical documentation and contributor guides
|   └── CONTRIBUTION_GUIDE.md     # CRITICAL for high PR volume
|
├── /src                          # Core C++/CUDA backend logic
|   ├── /cuda_kernels             # Highly optimized CUDA implementations
|   └── /cpp_bindings             # C++/Python/PyTorch interfaces
|
├── /hdas                         # All Houdini Digital Assets (HDAs)
|   ├── /ML_Input_SOP             # Step 1: Data Transfer Node
|   |   ├── ML_Input_SOP.hda      # The HDA file
|   |   ├── vex                   # VEX snippets and helpers
|   |   └── python                # Python wrapper logic
|   |
|   ├── /Sim_Denoise_SOP          # Step 2: Denoising Node
|   |   ├── Sim_Denoise_SOP.hda
|   |   ├── /tests                # Example scenes and unit tests for *this* HDA
|   |   └── README.md             # Specific documentation for *this* node
|   |
|   └── /ML_Solver_Template_DOP   # Future modular HDAs go here
|
├── .gitignore                    # Ignore binaries, build artifacts, and Houdini temp files
├── setup.py                      # For installing the Python/PyTorch wrappers
├── requirements.txt              # PyTorch/TensorFlow, etc.
└── README.md                     # Project overview and installation
```

## 🌟 Example: Neural Smoke Denoising
Using the DeepVEX nodes, you can take a low-resolution, noisy Pyro simulation and pass it through a pre-trained UNet to recover high-frequency details.

### The Pipeline:
1. **ML_Model_Loader**: Load `smoke_denoiser_v01.pt`.
2. **ML_Input**: Capture `density` and `vel` attributes as Tensors.
3. **ML_Feedback_Solver**: Compute the next state using the temporal registry.
4. **Result**: High-fidelity simulation at a fraction of the computation time.

> **Note:** Example `.hip` files and pre-trained models are located in the `/examples` directory (Coming Soon - PRs welcome!).
