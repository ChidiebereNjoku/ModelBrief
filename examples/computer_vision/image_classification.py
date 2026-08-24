# Requires modelbrief[pytorch]
import torch
from modelbrief import ModelBrief
images=torch.randn(12,3,16,16); labels=torch.arange(12)%2; model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(3*16*16,2)); ModelBrief(model,X_test=images,y_test=labels.numpy(),task="computer_vision").show()
