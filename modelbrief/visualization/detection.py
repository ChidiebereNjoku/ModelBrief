def draw_boxes(image,boxes):
 import matplotlib.pyplot as plt
 from matplotlib.patches import Rectangle
 fig,ax=plt.subplots(); ax.imshow(image)
 for b in boxes: ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,color="red"))
 return fig
