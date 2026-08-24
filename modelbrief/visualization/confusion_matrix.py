def plot_confusion_matrix(matrix,labels=None):
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(); im=ax.imshow(matrix,cmap="Blues"); fig.colorbar(im,ax=ax); ax.set(title="Confusion matrix",xlabel="Predicted",ylabel="Actual")
 if labels is not None: ax.set_xticks(range(len(labels)),labels); ax.set_yticks(range(len(labels)),labels)
 for i,row in enumerate(matrix):
  for j,v in enumerate(row): ax.text(j,i,str(v),ha="center",va="center")
 fig.tight_layout(); return fig
