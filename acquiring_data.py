import os
import pickle

import cv2
import mss
import numpy as np
import pyWinhook as pyHook
import pythoncom
import wx

# getting information about screen display resolution
from numpy.ma import indices

app = wx.App(False)
screen_width, screen_height = wx.GetDisplaySize()

training_data_arrays_cycles = 0
# file names for training data arrays
FILE_NAME = 'training_data.npy'
COPY_FILE_NAME = 'training_data_copy.npy'


# getting screenshot and changing it colours to Grayscale
def get_screen():
    with mss.mss() as sct:
        screen = np.array(sct.grab(sct.monitors[1]))
        # Drop alpha channel
        screen = screen[:, :, :-1]
        screen = np.array(screen)
        #screen = cv2.resize(screen, (640, 480))
        return screen


def process_data():
    start = True
    global training_data_arrays_cycles
    # to correctly process range of files as it is not inclusive we have to add one to cycles
    # Backup
    print('Processing Data')
    if os.path.isfile(FILE_NAME):
        training_data = np.load(FILE_NAME, allow_pickle=True)
        start = False
    for i in range(training_data_arrays_cycles):
        data_to_append = np.load(f'training_data{i}.npy', allow_pickle=True)
        if start:
            training_data = data_to_append
            start = False
        else:
            training_data = np.concatenate((data_to_append, training_data))
        print(f'{i} out of {training_data_arrays_cycles - 1} processed.')
        #os.remove(f'training_data{i}.npy')
    print('Saving data')
    np.save('training_data.npy', training_data)
    print('Data processed successfully')


def get_data():
    training_data = []

    # saving data after acquiring 500 sets of inputs and screenshots
    def save_data(screen, output):
        global training_data_arrays_cycles
        training_data.append([screen, output])
        if len(training_data) % 500 == 0:
            np.save(f'training_data{training_data_arrays_cycles}', training_data)
            training_data.clear()
            training_data_arrays_cycles += 1
            print(f'Data Checkpoint {training_data_arrays_cycles}')
        print("Frames taken: " + str(len(training_data)))

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
        if event.Key == 'Q':
            process_data()
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
    '''
    Don't need mouse events for now but they could be used in future
    # set the hook
    # hm.HookMouse()
    hm.MouseLeftDown = OnMouseEvent
    hm.MouseRightDown = OnMouseEvent
    hm.MouseMiddleDown = OnMouseEvent
    MouseMove should be periodically disabled, because it's throttling other inputs
    hm.MouseMove = OnMouseEvent
    '''

    hm.KeyDown = OnKeyboardEvent
    hm.HookKeyboard()
    # wait forever
    try:
        pythoncom.PumpMessages()
    except KeyboardInterrupt:
        pass

    # looping getting data
    while True:
        pass


if __name__ == '__main__':
    get_data()
    process_data()
