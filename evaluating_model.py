import numpy as np
import pyWinhook as pyHook
import pythoncom

import controls
from acquiring_data import get_screen
from tflearn_configuration import trainingmodel


def OnKeyboardEvent(event):
    if event == 'q':
        paused = True


def evaluate_model():
    # preparing model configuration
    # TODO model configuration can be changed to more accessiable option
    WIDTH = 136
    HEIGHT = 76
    LR = 1e-4
    EPOCHS = 3
    MODEL_NAME = 'WarcraftIII-learning_rate{}-epochs{}'.format(LR, EPOCHS)
    model = trainingmodel(WIDTH, HEIGHT, LR)
    model.load(MODEL_NAME)
    # create a hook manager
    hm = pyHook.HookManager()
    # hm.MouseMove = OnMouseEvent
    hm.KeyUp = OnKeyboardEvent
    # set the hook
    hm.HookKeyboard()
    # wait forever
    try:
        pythoncom.PumpMessages()
    except KeyboardInterrupt:
        pass
    paused = False
    while (True):
        if not paused:
            # 1366 x 768 resolution
            screen = get_screen()
            action = list(np.around(model.predict([screen.reshape(136, 76, 1)])[0]))
            x_position, y_position, action = action[0], action[1], action[2]
            controls.make_action(x_position, y_position, action)
        else:
            choice = input('Waiting for action. Input start to continue')
            if choice == 'start':
                paused = False
