import os

from imageai.Detection import ObjectDetection


class execution_path():
    execution_path = os.getcwd()


def training_model():
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(execution_path.execution_path, "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
