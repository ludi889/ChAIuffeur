import os

from imageai.Detection import ObjectDetection


def setup_detector():
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(os.path.join(os.getcwd(), "resnet50_coco_best_v2.1.0.h5"))
    detector.loadModel()
    return detector
