from ultralytics import YOLO
import os

# Load the model
model = YOLO("yolo11n.pt")

# Display model information
model.info()

# Export to ONNX with 640x640 size
model.export(format='onnx',
            simplify=True,
            opset=11,
            imgsz=(640, 640),
           )

"""
# Create a new model instance for 320x320
model = YOLO("yolo11n.pt")
model.export(format='onnx',
            simplify=True,
            opset=11,
            imgsz=(320, 320),
           )
"""

