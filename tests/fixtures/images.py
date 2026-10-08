import numpy as np
def image_data(): return np.random.default_rng(1).normal(size=(12,8,8,1)),np.arange(12)%2
