import numpy as np

import controls
from acquiring_data import get_screen
from tflearn_configuration import trainingmodel


def evaluate_model():
    # preparing model configuration
    # TODO model configuration can be changed to more accessible option
    WIDTH = 136
    HEIGHT = 76
    LR = 1e-4
    EPOCHS = 2
    MODEL_NAME = 'Model-learning_rate{}-epochs{}'.format(LR, EPOCHS)
    model = trainingmodel(WIDTH, HEIGHT, LR)
    # loading the model
    model.load(MODEL_NAME)
    paused = False
    # setting the model in rush - it's predicting action basing on given screenshot
    while True:
        if not paused:
            # 1366 x 768 resolution
            screen = get_screen()
            action = list(np.around(model.predict([screen.reshape(136, 76, 1)])[0]))
            x_position, y_position, action = int(action[0]), int(action[1]), int(action[2])
            controls.make_action(x_position, y_position, action)
