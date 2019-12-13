import time

import pyautogui
import tensorflow_datasets as tfds

# this module is prepared for fitted model to make predicted actions
# TODO this probably can be reworked to dialog choice using TKinter, but I didn't find such need in my case
# TODO this has also to be checked in practice in terms of making actual actions

# loading encoder
encoder = tfds.features.text.TokenTextEncoder.load_from_file('encoder')


# action is taken based on x-y position of mouse and action predicted by model
def make_action(position_x, position_y, action):
    # making array out of action - encoder require it for proper decoding
    action_array = [action]
    print(action_array)
    # decoding action coded in int by encoder to string of corresponding action
    action = encoder.decode(action_array)
    action = action.lower()
    if position_x or position_y == 0:
        position_x, position_y = 1, 1
    # moving mouse to position of action
    pyautogui.moveTo(position_x, position_y)
    # regarding action which has to be made
    if action == 'move':
        pass
    elif action == 'left':
        pyautogui.leftClick()
    elif action == 'right':
        pyautogui.rightClick()
    elif action == 'middle':
        pyautogui.middleClick()
    elif 'bind' in action:
        number = action[-1]
        pyautogui.hotkey('shift', number)
    elif 'add' in action:
        number = action[-1]
        pyautogui.hotkey('ctrl', number)
    else:
        pyautogui.keyDown(action)
        time.sleep(2)
        pyautogui.keyUp(action)
    print('Action {} will be made on X-coordinate {} and Y-coordinate {}'.format(action, position_x, position_x))
