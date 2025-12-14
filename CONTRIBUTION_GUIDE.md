# DeepVEX: Houdini-ML-Glue
## ⚡ Performance-Optimized ML Integration for VFX and Scientific Simulation

DeepVEX provides a highly modular, low-latency bridge between **Houdini** and core **Machine Learning** frameworks like PyTorch and TensorFlow. Our goal is to expose fundamental ML operations as native, easy-to-use Houdini Digital Assets (HDAs) and VEX/Python wrappers.

This project is optimized for performance and contribution.

---

### Key Features
* **Zero-Copy Data Transfer:** Efficiently convert Houdini geometry attributes (points, vectors) into PyTorch Tensors and back using the **ML_Input_SOP**.
* **Modular HDAs:** Every feature (denoising, attribute mapping, etc.) is a separate HDA, allowing for rapid integration and easy updates.
* **CUDA/VEX Optimization:** Low-level bindings ensure simulation-level performance.

### 🚀 Quick Start: Installing DeepVEX

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YourUsername/Houdini-ML-Glue.git](https://github.com/YourUsername/Houdini-ML-Glue.git)
    cd Houdini-ML-Glue
    ```
2.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    python setup.py install # This builds C++ and Python bindings
    ```
3.  **Configure Houdini:**
    Add the repository's main directory and the `/hdas` directory to your `HOUDINI_PATH` environment variable.

### Core HDAs Included

| Node Name | Type | Description |
| :--- | :--- | :--- |
| **ML_Input_SOP** | Utility | Converts specified geometry attributes (e.g., `P`, `v`) into a PyTorch Tensor object, storing the reference as a detail attribute. |
| **Sim_Denoise_SOP** | Feature | Applies a pre-trained ML denoiser model to a point attribute (typically velocity or noise) to stabilize simulations. |

---

### 👋 Contributing
DeepVEX is designed for high-volume community contributions! Check out our detailed **[CONTRIBUTION_GUIDE.md](CONTRIBUTION_GUIDE.md)** to see how you can get started, whether you're optimizing VEX snippets or adding a new ML node.
````

## 2\. 📝 `CONTRIBUTION_GUIDE.md` (The PR Engine)

This file is your PR blueprint, explicitly inviting small, high-quality contributions.

```markdown
# DeepVEX Contribution Guide: The PR Engine

Welcome to the DeepVEX project! This repository is intrinsically modular, making it easy to generate high-impact contributions (Pull Requests) quickly.

**Primary Goal:** Every feature, optimization, or documentation fix should be a self-contained Pull Request.

### 🛠️ How to Contribute: The Three Tiers

| Tier | Focus | Examples (Easy PRs!) | Location |
| :--- | :--- | :--- | :--- |
| **Tier 1: Documentation/Style** | Fixing typos, improving clarity, VEX comments. | Fix grammar in `README.md`. Write clear VEX comments in `/hdas/ML_Input_SOP/vex`. | Anywhere |
| **Tier 2: Optimization/Testing** | Minor VEX/Python logic improvements or test scene creation. | Optimize a single line of VEX. Add a simple test `.hipnc` file to `/hdas/Sim_Denoise_SOP/tests`. | `/hdas`, `/src/cpp_bindings` |
| **Tier 3: New Feature (New HDA)** | Adding a brand new ML function as a standalone HDA. | Implement `ML_Attribute_Transfer_SOP` (New Folder Required). | `/hdas/New_Feature_SOP` |

---

### 🚀 Tier 3: Adding a New HDA (Feature PR Template)

If you are adding a new Machine Learning node, follow this structure precisely. **Each new node must be its own PR.**

#### 1. Create a New Module Folder
Navigate to the `/hdas` directory and create a new folder named after your HDA (e.g., `/hdas/ML_Lookup_SOP`).

#### 2. Essential Files for a New HDA

Your new folder must contain, at a minimum:

* `ML_Lookup_SOP.hda`: The actual Digital Asset file.
* `README.md`: Explaining the parameters and function of *this specific node*.
* `/tests/`: A folder containing at least one `.hipnc` file demonstrating its functionality.

#### 3. Standard Pull Request Template

Please use the following template for your PR description:

```

**Title:** [HDA/FEAT] Add new node: ML\_Lookup\_SOP for tensor indexing
**Description:**
This PR introduces the ML\_Lookup\_SOP. It allows users to take a PyTorch tensor reference (from ML\_Input\_SOP) and use point attribute data (e.g., point ID) to index and sample the tensor, returning the result as a new attribute.

  * Implements core logic in `/src/cpp_bindings/lookup_kernels.cpp`.
  * Adds `lookup_tensor` function to the Python interface.

**Checklist (Must be complete for merge):**

  - [ ] New module folder created under `/hdas/`
  - [ ] HDA saved with versioning and locked interface
  - [ ] Node-specific `README.md` created
  - [ ] At least one `.hipnc` test scene included in the `/tests` folder

<!-- end list -->

