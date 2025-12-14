# Houdini ML Glue

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
