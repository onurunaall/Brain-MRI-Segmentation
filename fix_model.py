import torch
from collections import OrderedDict

# Load the "broken" weights
state_dict = torch.load("./checkpoints/best_model.pt", map_location="cpu")
new_state_dict = OrderedDict()

# Remove the '_orig_mod.' prefix from every key
for k, v in state_dict.items():
    name = k.replace("_orig_mod.", "") 
    new_state_dict[name] = v

# Overwrite the file with the clean version
torch.save(new_state_dict, "./checkpoints/best_model.pt")
print("Model fixed! You can now run predict.py.")