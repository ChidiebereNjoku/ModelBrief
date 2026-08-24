import torch
from modelbrief import ModelBrief
X=torch.randn(40,4); y=(X[:,0]>0).long(); model=torch.nn.Sequential(torch.nn.Linear(4,2)); opt=torch.optim.Adam(model.parameters());
for _ in range(20): opt.zero_grad(); loss=torch.nn.functional.cross_entropy(model(X),y); loss.backward(); opt.step()
ModelBrief(model,X_test=X,y_test=y.numpy(),task="classification").show()
