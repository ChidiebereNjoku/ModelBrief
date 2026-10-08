def plot_distribution(values,title="Distribution"):
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(); ax.hist(values,bins=30); ax.set_title(title); fig.tight_layout(); return fig
