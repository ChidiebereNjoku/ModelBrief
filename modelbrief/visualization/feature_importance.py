def plot_feature_importance(items,top_n=20):
 import matplotlib.pyplot as plt
 items=list(items)[:top_n][::-1]; fig,ax=plt.subplots(); ax.barh([x[0] for x in items],[x[1] for x in items]); ax.set_title("Feature importance"); fig.tight_layout(); return fig
