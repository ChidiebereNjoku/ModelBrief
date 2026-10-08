def plot_residuals(y,pred):
 import matplotlib.pyplot as plt
 r=[a-b for a,b in zip(y,pred)]; fig,ax=plt.subplots(); ax.scatter(pred,r,alpha=.6); ax.axhline(0,color="black"); ax.set(title="Residuals",xlabel="Predicted",ylabel="Residual"); fig.tight_layout(); return fig
