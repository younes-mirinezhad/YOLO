from ultralytics import YOLO

class YOLO_PT_Detector():
    def __init__(self):
        self.model = None

    def loadModel(self, modelPath):
        self.model = YOLO(modelPath, task='detect')

    def inference(self, image):
        res = self.model(image)
        annotated_frame = res[0].plot()
        return annotated_frame
