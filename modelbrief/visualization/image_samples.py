def plot_image_samples(images,labels=None,n=9):
 import matplotlib.pyplot as plt
 n=min(n,len(images)); fig,axes=plt.subplots(1,n,squeeze=False,figsize=(3*n,3))
 for i in range(n): axes[0,i].imshow(images[i],cmap="gray"); axes[0,i].axis("off"); axes[0,i].set_title(str(labels[i]) if labels is not None else f"Sample {i}")
 fig.tight_layout(); return fig
