import os

import numpy as np
from imageai.Detection import ObjectDetection

import retinanet_configuration


def evaluate_model():
    paused = False
    # model = retinanet_configuration.training_model()
    # setting the model in rush - it's predicting action basing on given screenshot
    file_name = 'training_data.npy'
    training_data = list(np.load(file_name, allow_pickle=True))
    while True:
        if not paused:
            # 1366 x 768 resolution
            # screen = get_screen()
            detector = ObjectDetection()
            detector.setModelTypeAsRetinaNet()
            detector.setModelPath(
                os.path.join(retinanet_configuration.execution_path.execution_path, "resnet50_coco_best_v2.0.1.h5"))
            detector.loadModel()
            for index, i in enumerate(training_data):
                screen = i[0]
                filename = "imagenew" + str(index) + ".jpg"
                predictions = detector.detectObjectsFromImage(input_image=screen, input_type="array",
                                                              output_image_path=os.path.join(
                                                                  retinanet_configuration.execution_path.execution_path,
                                                                  filename), minimum_percentage_probability=25)
