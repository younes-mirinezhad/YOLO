from ultralytics import YOLO
from v1_EngineBuilder import Builder
import cv2
from v1_TRT_Detector import TRT_Detector
from yolo_PT_Detector import YOLO_PT_Detector
from onnx_Segmentor import ONNX_Segmentor

class_names = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
               'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
               'hair drier', 'toothbrush' ]


def exportNormalOnnx(modelPath):
    model = YOLO(modelPath)
    model.export(format="onnx", batch=1, opset=12, imgsz=640) #simplify=True, dynamic=False

def v1_buildEngine(modelPath_PT, modelPath_ONNX, modelPath_Engine):
    builder = Builder()
    builder.setModelPath(modelPath_PT, modelPath_ONNX, modelPath_Engine)
    builder.setConfigs(input_shape=[1, 3, 640, 640], topk=100, conf_thres=0.2, iou_thres=0.5)
    builder.build()

def detect_TRT(modelPath_Engine, isVideo, filePath):
    trt_detector = TRT_Detector()
    trt_detector.setClassName(class_names)
    trt_detector.loadModel(modelPath_Engine)

    if isVideo:
        cap = cv2.VideoCapture(filePath)
        fnum = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            trt_annotated = trt_detector.inference(frame)
            cv2.imshow("TRT Inference", trt_annotated)
            
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

        trt_annotated = trt_detector.inference(frame)
        cv2.imshow("TRT Inference", trt_annotated)

        key = cv2.waitKey()

        cv2.destroyAllWindows()

def detect_PT_YOLO(modelPath_PT, isVideo, filePath):
    yolo_pt_detector = YOLO_PT_Detector()
    yolo_pt_detector.loadModel(modelPath_PT)

    if isVideo:
        cap = cv2.VideoCapture(filePath)
        fnum = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            yolo_pt_annotated = yolo_pt_detector.inference(frame)
            cv2.imshow("YOLO PT Inference", yolo_pt_annotated)
            
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

        yolo_pt_annotated = yolo_pt_detector.inference(frame)
        cv2.imshow("YOLO PT Inference", yolo_pt_annotated)

        key = cv2.waitKey()

        cv2.destroyAllWindows()

def segment_OXXN(modelPath_ONNX, isVideo, filePath):
    onnx_Segmentor = ONNX_Segmentor()
    onnx_Segmentor.setClassName(class_names)
    onnx_Segmentor.loadModel(modelPath_ONNX)

    if isVideo:
        cap = cv2.VideoCapture(filePath)
        fnum = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            start_time = cv2.getTickCount()
            segmentedImg = onnx_Segmentor.inference(frame)
            end_time = cv2.getTickCount()
            inference_time = (end_time - start_time) / cv2.getTickFrequency() * 1000

            cv2.imshow("ONNX Inference", segmentedImg)
            
            fnum += 1
            print(f"-----> Frame: {fnum} , Inference time: {inference_time:.2f} ms")
            key = cv2.waitKey()
            if key == ord('q'):
                break
            else:
                continue
            
        cap.release()
        cv2.destroyAllWindows()
    else:
        frame = cv2.imread(filePath)

        segmentedImg = onnx_Segmentor.inference(frame)
        cv2.imshow("ONNX Inference", segmentedImg)

        key = cv2.waitKey()

        cv2.destroyAllWindows()

if __name__ == '__main__':
    # exportNormalOnnx("model.pt") # Export onnx

    # v1_buildEngine(modelPath_PT, modelPath_ONNX, modelPath_Engine) # Export onnx_e2e and Build TRT_Engine

    isVideo = False
    filePath = "../Files/Images/000000000025.jpg"

    # modelPath_PT = "../Files/Models/Detection/yolov8n.pt"
    # detect_PT_YOLO(modelPath_PT, isVideo, filePath)

    # modelPath_Engine = "../Files/Models/Detection/yolov8n_b5_640_end2end.engine"
    # detect_TRT(modelPath_Engine, isVideo, filePath)

    # modelPath_ONNX = "../Files/Models/Segmentation/yolov8n-seg_b1_640.onnx"
    # segment_OXXN(modelPath_ONNX, isVideo, filePath)

