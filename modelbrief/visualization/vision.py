def normalize_image(image):
 import numpy as np
 a=np.asarray(image); lo,hi=float(a.min()),float(a.max()); return (a-lo)/(hi-lo) if hi>lo else a
