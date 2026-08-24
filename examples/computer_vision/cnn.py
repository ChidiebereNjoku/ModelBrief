import torch
from modelbrief import ModelBrief
X=torch.randn(10,1,16,16); y=torch.arange(10)%2; model=torch.nn.Sequential(torch.nn.Conv2d(1,4,3),torch.nn.ReLU(),torch.nn.Flatten(),torch.nn.Linear(4*14*14,2)); ModelBrief(model,X_test=X,y_test=y.numpy(),task="computer_vision").html("reports/cnn.html")
