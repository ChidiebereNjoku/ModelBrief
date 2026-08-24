import numpy as np
import tensorflow as tf
from modelbrief import ModelBrief
X=np.random.randn(40,4).astype("float32"); y=(X[:,0]>0).astype(int); model=tf.keras.Sequential([tf.keras.layers.Input((4,)),tf.keras.layers.Dense(2,activation="softmax")]); model.compile("adam","sparse_categorical_crossentropy"); model.fit(X,y,epochs=2,verbose=0); ModelBrief(model,X_test=X,y_test=y,task="classification").show()
