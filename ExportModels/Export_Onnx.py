from ultralytics import YOLO

model = YOLO('path/to/model.pt')

model.export(format="onnx", opset=12, dynamic=False, imgsz=640)

# simplify=True
