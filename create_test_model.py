import torch
import torch.nn as nn

class HelloHoudini(nn.Module):
    def __init__(self):
        super(HelloHoudini, self).__init__()
        # Takes 3 inputs (X, Y, Z) and outputs 3 (new X, Y, Z)
        self.layer = nn.Linear(3, 3)
        
        # Initialize with an identity-like matrix so it doesn't 
        # just explode the geometry instantly.
        with torch.no_grad():
            self.layer.weight.copy_(torch.eye(3))
            self.layer.bias.fill_(0.1) # Add a tiny nudge

    def forward(self, x):
        return self.layer(x)

# 1. Initialize Model
model = HelloHoudini()

# 2. Convert to TorchScript (This is what LibTorch/Houdini needs)
scripted_model = torch.jit.script(model)

# 3. Save it
scripted_model.save("hello_world_model.pt")
print("Model saved as hello_world_model.pt")
