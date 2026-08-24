import numpy as np
from modelbrief.analysis.image_segmentation import analyse
y=np.array([[1,1,0],[0,0,0]]); print(analyse(y,y.copy()))
