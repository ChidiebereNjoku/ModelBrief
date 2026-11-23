# Computer vision

ModelBrief supports image-classification workflows with model-ready image tensors or arrays. PyTorch and TensorFlow/Keras adapters can be used when the supplied model accepts the image shape and dtype passed to `ModelBrief`. Set `task="computer_vision"` for image-classification analysis.

```python
import torch
from modelbrief import ModelBrief

images = torch.randn(12, 3, 16, 16)
labels = torch.arange(12) % 2

model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(3 * 16 * 16, 2),
)

report = ModelBrief(
    model,
    X_test=images,
    y_test=labels.numpy(),
    task="computer_vision",
)
report.show()
```

Additional repository examples cover:

- [`examples/computer_vision/cnn.py`](../examples/computer_vision/cnn.py) — a PyTorch CNN image-classification workflow.
- [`examples/computer_vision/image_classification.py`](../examples/computer_vision/image_classification.py) — model-ready image tensors and labels.
- [`examples/computer_vision/object_detection.py`](../examples/computer_vision/object_detection.py) — object-detection boxes and intersection-over-union analysis.
- [`examples/computer_vision/image_segmentation.py`](../examples/computer_vision/image_segmentation.py) — image-segmentation ground truth and predictions.

Object-detection and segmentation helpers analyse already-produced boxes or masks; specialised models may need an application-specific adapter or output conversion.
