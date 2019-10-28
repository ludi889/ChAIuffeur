import pickle

import pyautogui

# this module is prepared for fitted model to make predicted actions
# acquring tokenizer which has to be the same, as the one used in tokenizing data in data processing
# TODO this probably can be reworked to dialog choice using TKinter, but I didn't find such need in my case
# TODO It also has to be checked in practice because unfortunately I didn't make it to proper model state

tokenizer = pickle.load('tokenizer.pickle', 'rb')


# action is taken based on x-y position of mouse and action predicted by model
def make_action(position_x, position_y, action):
    action = tokenizer.tokenize(action)
    pyautogui.moveRel(position_x, position_y)
    if action == 'mouse move':
        pass
    elif action == 'mouse left down':
        pyautogui.leftClick()
    elif action == 'mouse right down':
        pyautogui.rightClick()
    elif 'bind' in action:
        number = action[-1]
        pyautogui.hotkey('shift', number)
    elif 'add' in action:
        number = action[-1]
        pyautogui.hotkey('ctrl', number)
    else:
        pyautogui.press(action)
