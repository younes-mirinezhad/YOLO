from ultralytics import YOLO
from v1_EngineBuilder import Builder
import cv2
from v1_TRT_Detector import TRT_Detector
from yolo_PT_Detector import YOLO_PT_Detector
# class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
#                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
#                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
#                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
#                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
#                'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
#                'hair drier', 'toothbrush' ]
# modelPath_PT = "Models/Detection/yolov8n.pt"
# odelPath_ONNX = "Models/Detection/yolov8n_end2end.onnx"
# modelPath_Engine = "Models/Detection/yolov8n_end2end.engine"
# filePath = "/home/chiko/Downloads/Compressed/cars.mp4"

modelPath_PT1 = "/media/chiko/HDD_1/Work/Watchtower/02_Flare/Development/Models/Detection/V4/Normal/best.pt"
modelPath_ONNX1 = "/media/chiko/HDD_1/Work/Watchtower/02_Flare/Development/Models/Detection/V4/Normal/best_end2end.onnx"
modelPath_Engine1 = "/media/chiko/HDD_1/Work/Watchtower/02_Flare/Development/Models/Detection/V4/Normal/best_end2end.engine"

modelPath_PT2 = "/media/chiko/HDD_1/Work/Watchtower/02_Flare/Development/Models/Detection/V4/InsideAugmentation/best.pt"
modelPath_ONNX2 = "/media/chiko/HDD_1/Work/Watchtower/02_Flare/Development/Models/Detection/V4/InsideAugmentation/best_end2end.onnx"
modelPath_Engine2 = "/media/chiko/HDD_1/Work/Watchtower/02_Flare/Development/Models/Detection/V4/InsideAugmentation/best_end2end.engine"

modelPath_PT3 = "/media/chiko/HDD_1/Work/Watchtower/02_Flare/Development/Models/Detection/V4/OutsideAugmentation/best.pt"
modelPath_ONNX3 = "/media/chiko/HDD_1/Work/Watchtower/02_Flare/Development/Models/Detection/V4/OutsideAugmentation/best_end2end.onnx"
modelPath_Engine3 = "/media/chiko/HDD_1/Work/Watchtower/02_Flare/Development/Models/Detection/V4/OutsideAugmentation/best_end2end.engine"

class_names = [ 'Flame', 'Smoke']
filePath = "/media/chiko/HDD_1/Work/Watchtower/02_Flare/Dataset/13.mp4"

isVideo = True

def exportNormalOnnx(modelPath):
    model = YOLO(modelPath)
    model.export(format="onnx", opset=12, dynamic=False, imgsz=640) #simplify=True

def v1_buildEngine(modelPath_PT, modelPath_ONNX, modelPath_Engine):
    builder = Builder()
    builder.setModelPath(modelPath_PT, modelPath_ONNX, modelPath_Engine)
    builder.setConfigs(input_shape=[1, 3, 640, 640], topk=100, conf_thres=0.2, iou_thres=0.5)
    builder.build()

if __name__ == '__main__':
    import torch
    assert torch.cuda.is_available(), "CUDA is not available! Check your setup."

    # exportNormalOnnx(modelPath_PT1) # Export onnx
    v1_buildEngine(modelPath_PT1, modelPath_ONNX1, modelPath_Engine1) # Build engine # Export onnx_e2e and Engine

    # exportNormalOnnx(modelPath_PT2) # Export onnx
    v1_buildEngine(modelPath_PT2, modelPath_ONNX2, modelPath_Engine2) # Build engine # Export onnx_e2e and Engine

    # exportNormalOnnx(modelPath_PT3) # Export onnx
    v1_buildEngine(modelPath_PT3, modelPath_ONNX3, modelPath_Engine3) # Build engine # Export onnx_e2e and Engine

    trt_detector1 = TRT_Detector()
    trt_detector1.setClassName(class_names)
    trt_detector1.loadModel(modelPath_Engine1)

    trt_detector2 = TRT_Detector()
    trt_detector2.setClassName(class_names)
    trt_detector2.loadModel(modelPath_Engine2)

    trt_detector3 = TRT_Detector()
    trt_detector3.setClassName(class_names)
    trt_detector3.loadModel(modelPath_Engine3)

    # yolo_pt_detector = YOLO_PT_Detector()
    # yolo_pt_detector.loadModel(modelPath_PT)

    if isVideo:
        cap = cv2.VideoCapture(filePath)
        fnum = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            trt_annotated1 = trt_detector1.inference(frame)
            cv2.imshow("TRT Inference 1", trt_annotated1)

            trt_annotated2 = trt_detector2.inference(frame)
            cv2.imshow("TRT Inference 2", trt_annotated2)

            trt_annotated3 = trt_detector3.inference(frame)
            cv2.imshow("TRT Inference 3", trt_annotated3)

            # yolo_pt_annotated = yolo_pt_detector.inference(frame)
            # cv2.imshow("YOLO PT Inference", yolo_pt_annotated)
            
            fnum += 1
            print("-----> Frame: ", fnum)
            key = cv2.waitKey()
            if key == ord('q'):
                break
            else:
                continue
            
        cap.release()
        cv2.destroyAllWindows()
    else:
        frame = cv2.imread(filePath)

        trt_annotated = trt_detector1.inference(frame)
        cv2.imshow("TRT Inference 1", trt_annotated)

        yolo_pt_annotated = trt_detector2.inference(frame)
        cv2.imshow("TRT Inference 2", yolo_pt_annotated)

        yolo_pt_annotated = trt_detector3.inference(frame)
        cv2.imshow("TRT Inference 3", yolo_pt_annotated)

        key = cv2.waitKey()

        cv2.destroyAllWindows()
