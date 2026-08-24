def plot_segmentation(image,mask):
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(); ax.imshow(image); ax.imshow(mask,alpha=.4,cmap="viridis"); ax.axis("off"); return fig
