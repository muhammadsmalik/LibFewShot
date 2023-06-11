import torch_xla.core.xla_model as xm

print("Before initializing device")
device = xm.xla_device()
print("After initializing device", device)
