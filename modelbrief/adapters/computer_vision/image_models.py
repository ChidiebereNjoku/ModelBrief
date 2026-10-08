from __future__ import annotations

def is_image_batch(X):
    shape=getattr(X,"shape",None)
    return bool(shape is not None and len(shape)==4 and (shape[1] in (1,3,4) or shape[-1] in (1,3,4)))
