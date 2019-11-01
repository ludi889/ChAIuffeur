import os

import cv2
import mss
import numpy as np
import pyWinhook as pyHook
import pythoncom
import wx

# getting information about screen display resolution
app = wx.App(False)
screen_width, screen_height = wx.GetDisplaySize()


# getting screenshot, changing it colours to Grayscale and resizing it for more convenient size for CNN
def get_screen():
    with mss.mss() as sct:
        screen = np.array(sct.grab((0, 0, screen_width, screen_height)))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screen = cv2.resize(screen, (136, 76))
    return screen


def get_data():
    # file names for training data arrays
    file_name = 'training_data.npy'
    copy_file_name = 'training_data_copy.npy'

    # deciding if previous file with data is saved. If yes, it is opened. If not it's created
    if os.path.isfile(file_name):
        print('File exists, loading previous data!')
        print(os.path.realpath(file_name))
        training_data = list(np.load(file_name, allow_pickle=True))
        np.save(copy_file_name, training_data)
    else:
        print('File does not exist, starting fresh!')
        training_data = []

    # saving data after acquiring 2500 sets of inputs and screenshots
    def save_data(screen, output):
        training_data.append([screen, output])
        if len(training_data) % 2500 == 0:
            print(len(training_data))
            np.save(file_name, training_data)
        print("Frames taken: " + str(len(training_data)))
        index = len(training_data) - 1
        print(training_data[index])

    # getting inputs and screen on mouse event
    def OnMouseEvent(event):
        action = event.MessageName
        screen = get_screen()
        output = [event.Position, 0]
        if action == 'mouse move':
            output[1] = 'move'
        elif action == 'mouse left down':
            output[1] = 'left'
        elif action == 'mouse right down':
            output[1] = 'right'
        elif action == 'mouse middle down':
            output[1] = 'middle'
        save_data(screen, output)
        return True

    # getting inputs and screen on keyboard event
    def OnKeyboardEvent(event):
        if event == 'Q':
            input('Pause. Press Enter to continue')
        screen = get_screen()
        # TODO a position of keyboard action could potentially be fixed if the specific position is needed
        output = [(1, 1), event.Key]
        # Getting special actions combinations with control and shift, for example binding units etc.
        ctrl_pressed = pyHook.GetKeyState(pyHook.HookConstants.VKeyToID('VK_CONTROL'))
        shift_pressed = pyHook.GetKeyState(pyHook.HookConstants.VKeyToID('VK_SHIFT'))
        try:
            if ctrl_pressed and int(pyHook.HookConstants.IDToName(event.KeyID)) in range(10):
                output[1] = 'bind' + event.Key
        except ValueError:
            pass
        try:
            if shift_pressed and int(pyHook.HookConstants.IDToName(event.KeyID)) in range(10):
                output[1] = 'add' + event.Key
        except ValueError:
            pass
        save_data(screen, output)
        return True

    # create a hook manager
    hm = pyHook.HookManager()
    # watch for all mouse events
    hm.MouseLeftDown = OnMouseEvent
    hm.MouseRightDown = OnMouseEvent
    hm.MouseMiddleDown = OnMouseEvent
    # MouseMove should be periodically disabled, because it's throttling other inputs
    hm.MouseMove = OnMouseEvent
    hm.KeyUp = OnKeyboardEvent
    # set the hook
    hm.HookMouse()
    hm.HookKeyboard()
    # wait forever
    try:
        pythoncom.PumpMessages()
    except KeyboardInterrupt:
        pass

    # looping getting data
    while True:
        pass
