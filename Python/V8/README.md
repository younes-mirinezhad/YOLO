# YOLOv8
Export models in python  
#
### install ultralytics
pip install ultralytics  
#
### Export normal onnx file
1: modelPath is the pt file path  
2: Call this function in main.py : exportNormalOnnx(modelPath)  
#
### Export tensorRT engine file
1: import Builder from v1_EngineBuilder file  
2: Set pytorch model path ---> modelPath_PT="path/to/model.pt"  
3: Set onnx end2end model path ---> odelPath_ONNX="path/to/model_e2e.onnx"  
4: Set tensorRT engine model path ---> modelPath_Engine="path/to/model_e2e.engine."  
5: set modelPath: ---> setModelPath(modelPath_PT, odelPath_ONNX, modelPath_Engine)  
6: Set configs ---> setConfigs(input_shape=[1, 3, 640, 640], topk=100, conf_thres=0.25, iou_thres=0.65)  
6: Call build function  
