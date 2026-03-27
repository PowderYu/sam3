import torch
from collections import OrderedDict

checkpoint = "/nas/yu/code/sam3/run/checkpoints/checkpoint.pt"

wrapped_model = torch.load(checkpoint, map_location="cpu")
model = wrapped_model["model"]
new_state_dict = OrderedDict(("detector." + k, v) for k, v in model.items())
torch.save(new_state_dict, checkpoint.replace(".pt", "_converted.pt"))