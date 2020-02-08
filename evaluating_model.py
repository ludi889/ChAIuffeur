import os

import cv2
import numpy as np
import pytesseract
from imageai.Detection import ObjectDetection

import retinanet_configuration
import reward


def sortSecond(val):
    return val[1]


def evaluate_model():
    paused = False
    # model = retinanet_configuration.training_model()
    # setting the model in rush - it's predicting action basing on given screenshot
    file_name = 'training_data.npy'
    training_data = list(np.load(file_name, allow_pickle=True))
    car_possible_objects = ['car', 'truck']
    # setting pytesseract
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    # setting detector
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(
        os.path.join(retinanet_configuration.execution_path.execution_path, "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    # define the dictionary of digit segments so we can identify
    # each digit on the thermostat
    DIGITS_LOOKUP = {
        (1, 1, 1, 0, 1, 1, 1): 0,
        (0, 0, 1, 0, 0, 1, 0): 1,
        (1, 0, 1, 1, 1, 0, 1): 2,
        (1, 0, 1, 1, 0, 1, 1): 3,
        (0, 1, 1, 1, 0, 1, 0): 4,
        (1, 1, 0, 1, 0, 1, 1): 5,
        (1, 1, 0, 1, 1, 1, 1): 6,
        (1, 0, 1, 0, 0, 1, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 1, 0, 1, 1): 9
    }

    # while True:
    if not paused:
        # 1366 x 768 resolution
        # screen = get_screen()
        for index, i in enumerate(training_data):
            non_agent_objects_coordinates = []
            possible_agents = []
            screen = i[0]
            filename = "imagenew" + str(index) + ".jpg"
            predictions = detector.detectObjectsFromImage(input_image=screen, input_type="array",
                                                          output_image_path=os.path.join(
                                                              retinanet_configuration.execution_path.execution_path,
                                                              filename), minimum_percentage_probability=50)
            current_speed = []
            for ind in predictions:
                object_type = ind.get('name')
                box_points = ind.get('box_points')
                # if object is clock get speed from clock
                if object_type == 'clock':
                    image_copy = screen.copy()
                    clock_coordinates = box_points
                    # clock = image_copy[clock_coordinates[1] + 135:clock_coordinates[3] - 60,
                    #        clock_coordinates[0] + 70:clock_coordinates[2] - 70]
                    clock = image_copy[clock_coordinates[1] + 135:clock_coordinates[3] - 50,
                            clock_coordinates[0] + 65:clock_coordinates[2] - 65]
                    kernel = np.ones((3, 3), np.uint8)
                    clock = cv2.cvtColor(clock, cv2.COLOR_BGR2GRAY)
                    ret, clock = cv2.threshold(clock, 85, 255, cv2.THRESH_BINARY)
                    clock = cv2.erode(clock, kernel, iterations=1)
                    # find the contours in the mask
                    cnts, hierarchy = cv2.findContours(clock.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    digits = []
                    for c in cnts:
                        if 200 < cv2.contourArea(c) < 1500:
                            # extract the digit ROI

                            (x, y, w, h) = cv2.boundingRect(c)
                            roi = clock[y:y + h, x:x + w]
                            print(cv2.contourArea(c))
                            # compute the width and height of each of the 7 segments
                            # we are going to examine
                            # (roiH, roiW) = roi.shape
                            # (dW, dH) = (int(roiW * 0.15), int(roiH * 0.05))
                            # dHC = int(roiH * 0.05)
                            # define the set of 7 segments
                            segments = [
                                ((1, 2), (20, 3)),  # top
                                ((3, 2), (4, 17)),  # top-left
                                ((17, 2), (18, 17)),  # top-right
                                ((0, 15,), (22, 16)),  # center
                                ((3, 17), (4, 34)),  # bottom-left
                                ((17, 17), (18, 34)),  # bottom-right
                                ((2, 30), (22, 31))  # bottom
                            ]
                            on = [0] * len(segments)
                            # loop over the segments
                            for (indexnext, ((xA, yA), (xB, yB))) in enumerate(segments):
                                area = (xB - xA) * (yB - yA)
                                # cv2.line(roi, (xA, yA), (xB, yB), (255, 255, 255), 1)
                                # extract the segment ROI, count the total number of
                                # thresholded pixels in the segment, and then compute
                                # the area of the segment
                                segROI = roi[yA:yB, xA:xB]
                                areah, areaw = segROI.shape
                                total = (areah * areaw) - cv2.countNonZero(segROI)
                                # if the total number of non-zero pixels is greater than
                                # 50% of the area, mark the segment as "on"
                                if total / float(area) > 0.65:
                                    on[indexnext] = 1
                                # lookup the digit and draw it on the image
                            try:
                                # between 220 and 230 is the area of digit 1
                                if 220 < cv2.contourArea(c) < 230:
                                    on = [0, 0, 1, 0, 0, 1, 0]
                                digit = DIGITS_LOOKUP[tuple(on)]
                                current_speed.append(digit)
                                cv2.rectangle(clock, (x, y), (x + w, y + h), (0, 0, 0), 1)
                                cv2.putText(clock, str(digit), (x + 7, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (255, 255, 255), 2)
                            except KeyError:
                                pass
                        else:
                            pass
                    cv2.imwrite("cnt" + str(index) + ".jpg", clock)
                center_x = box_points[2] - box_points[0]
                center_y = box_points[3] - box_points[1]
                if object_type not in car_possible_objects:
                    non_agent_objects_coordinates.append((center_x, center_y))
                else:
                    probability = ind.get('percentage_probability')
                    possible_agents.append([(center_x, center_x), probability])
            if not possible_agents:
                pass
            else:
                possible_agents.sort(reverse=True, key=sortSecond)
                agent_coordinates = possible_agents.pop(0)[0]
                for a in possible_agents:
                    non_agent_objects_coordinates.append(a[0])
                for object_coordinates in non_agent_objects_coordinates:
                    xa, ya = agent_coordinates
                    xo, yo = object_coordinates
                    dist = int(np.math.sqrt((xo - xa) ** 2 + (yo - ya) ** 2))
                    current_speed = ''.join(str(digit) for digit in current_speed)
                    reward.reward_function(dist, current_speed)
