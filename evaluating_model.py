import os
import time

import cv2.cv2 as cv2
import numpy as np
import pytesseract
from pytesseract import Output

import reward
from detector_setup import setup_detector

non_agent_objects_coordinates = []
possible_agents = []

PYTESSERACT_PATH = r'C:/Program Files/Tesseract-OCR/tesseract.exe'
DATA_PATH = f'{os.getcwd()}\\data\\'

# Threshold is a threshold of  of difference between columns/rows to stop clearing
TRESHOLD = 35


def sortSecond(val):
    return val[1]


def look_for_tresh_column(erode):
    to_delete = 0
    for column in erode:
        to_delete += 1
        diff = abs((erode[0].mean() - column.mean()))
        if diff > TRESHOLD or np.all(column == 0):
            print(f'Diff to break is {diff}')
            break
    return to_delete


def delete_black_borders(erode):
    TRESH_TO_CLEAR = 0.2
    # From top side
    print(f'Top side fraction is {erode[0].tolist().count(0) / len(erode[0])}')
    if (erode[0].tolist().count(0) / len(erode[0])) > TRESH_TO_CLEAR:
        print('Clearing')
        to_delete = look_for_tresh_column(erode)
        erode_no_top = erode[to_delete:]
    else:
        erode_no_top = erode
    '''
    
    '''
    # From bottom side
    erode_no_top = np.flipud(erode_no_top)
    print(f'Bottom side fraction is {erode_no_top[0].tolist().count(0) / len(erode_no_top[0])}')
    if (erode_no_top[0].tolist().count(0) / len(erode_no_top[0])) > TRESH_TO_CLEAR:
        print('Clearing')
        erode_flip = np.flipud(erode)
        to_delete = look_for_tresh_column(erode_flip)
        erode_no_top_no_bottom = erode_no_top[:-to_delete]
    else:
        erode_no_top_no_bottom = erode_no_top

    # From left side
    erode_no_top_no_bottom = erode_no_top_no_bottom.T
    print(f'Left side fraction is {erode_no_top_no_bottom[0].tolist().count(0) / len(erode_no_top_no_bottom[0])}')
    if (erode_no_top_no_bottom[0].tolist().count(0) / len(erode_no_top_no_bottom[0])) > TRESH_TO_CLEAR:
        print('Clearing')
        erode = erode.T
        to_delete = look_for_tresh_column(erode)
        erode_no_top_no_bottom_no_left = erode_no_top_no_bottom[:-to_delete]
    else:
        erode_no_top_no_bottom_no_left = erode_no_top_no_bottom

    # From right side
    erode_no_top_no_bottom_no_left = np.flipud(erode_no_top_no_bottom_no_left)
    print(
        f'Right side fraction is {erode_no_top_no_bottom_no_left[0].tolist().count(0) / len(erode_no_top_no_bottom_no_left[0])}')
    if (erode_no_top_no_bottom_no_left[0].tolist().count(0) / len(erode_no_top_no_bottom_no_left[0])) > TRESH_TO_CLEAR:
        print('Clearing')
        erode = erode.T
        erode_flip = np.flipud(erode)
        to_delete = look_for_tresh_column(erode_flip)
        erode_no_borders = erode_no_top_no_bottom_no_left[to_delete:]
    else:
        erode_no_borders = erode_no_top_no_bottom_no_left
    erode_no_borders = np.rot90(erode_no_borders)
    erode_no_borders = np.fliplr(erode_no_borders)
    return erode_no_borders


def get_current_speed(screen, box_points, index):
    image_copy = screen.copy()
    clock_coordinates = box_points
    clock = image_copy[clock_coordinates[1] + 145:clock_coordinates[3] - 30,
            clock_coordinates[0] + 85:clock_coordinates[2] - 70]
    ## Working on image to get best results - heuristic like work'
    # TODO get money for google vision
    resized = cv2.resize(clock, (0, 0), fx=16, fy=16)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    ret, tresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
    erode = cv2.erode(tresh, (13, 13), iterations=3)
    #erode = delete_black_borders(erode)
    # cv2.imshow('noborders', noborders)
    # cv2.waitKey()
    speed = pytesseract.image_to_string(erode, lang='letsgodigital',
                                        config='--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789').strip()
    noborders = cv2.putText(erode, speed, (00, 185), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 5, cv2.LINE_AA)
    cv2.imwrite(os.path.join(DATA_PATH + "cnt" + str(index) + ".jpg"), noborders)
    if speed:
        print(f'Speed is {speed} for {"cnt" + str(index) + ".jpg"}')
        return speed
    else:
        print(f'Speed couldn\'t be read from image')


def evaluate_model():
    paused = False
    file_name = 'training_data.npy'
    training_data = list(np.load(file_name, allow_pickle=True))
    car_possible_objects = ['car', 'truck']
    # setting pytesseract
    pytesseract.pytesseract.tesseract_cmd = PYTESSERACT_PATH
    # setting detector
    detector = setup_detector()

    # while True:
    if not paused:
        # 1366 x 768 resolution
        # screen = get_screen()
        for index, i in enumerate(training_data):
            speed = None
            screen = i[0]
            # print(type(screen))
            filename = "imagenew" + str(index) + ".jpg"
            predictions = detector.detectObjectsFromImage(input_image=screen, input_type="array",
                                                          output_image_path=os.path.join(DATA_PATH, filename),
                                                          minimum_percentage_probability=50)
            for prediction in predictions:
                object_type = prediction.get('name')
                box_points = prediction.get('box_points')
                # if object is clock get speed from clock
                if object_type == 'clock':
                    try:
                        speed = get_current_speed(screen, box_points, index)
                    except (IndexError, SystemError):
                        continue
                center_x = box_points[2] - box_points[0]
                center_y = box_points[3] - box_points[1]
                if object_type not in car_possible_objects:
                    non_agent_objects_coordinates.append((center_x, center_y))
                else:
                    probability = prediction.get('percentage_probability')
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
                    reward.reward_function(dist, speed)


if __name__ == '__main__':
    start = time.time()
    evaluate_model()
    end = time.time()
    print(f'Elapsed time is {end - start}')
