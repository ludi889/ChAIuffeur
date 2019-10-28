import numpy as np

import data_processing
from tflearn_configuration import trainingmodel


def fit_model():
    # preparing model configuration
    # TODO model configuration can be changed to more accessiable option
    WIDTH = 136
    HEIGHT = 76
    LR = 1e-4
    EPOCHS = 3
    MODEL_NAME = 'WarcraftIII-learning_rate{}-epochs{}'.format(LR, EPOCHS)
    model = trainingmodel(WIDTH, HEIGHT, LR)

    # loading, processing data and saving preprocessed data
    # TODO this probably can be reworked to dialog choice using TKinter, but I didn't find such need in my case
    copy_file_name = 'training_data_copy.npy'
    file_name = 'training_data.npy'
    processed_file_name = 'training_data_processed.npy'
    training_data = list(np.load(file_name, allow_pickle=True))
    training_data = data_processing.prepare_data(training_data)
    np.save(processed_file_name, training_data)

    # preparing data for model
    training_data = list(np.load(processed_file_name, allow_pickle=True))
    train_file_name = 'train.npy'
    test_file_name = 'test.npy'
    train_data_amount = int(0.9 * len(training_data)) - 1

    # splitting data for training set and test set
    train_data = training_data[:train_data_amount]
    test_data = training_data[train_data_amount:]

    # saving train and test data sets for backup
    # TODO this probably can be reworked to dialog choice using TKinter, but I didn't find such need in my case
    np.save(train_file_name, train_data)
    np.save(test_file_name, test_data)

    # loading train and test data sets
    # TODO this probably can be reworked to dialog choice using TKinter, but I didn't find such need in my case
    train_data = list(np.load(train_file_name, allow_pickle=True))
    test_data = list(np.load(test_file_name, allow_pickle=True))

    # setting train and test data for training model
    X = np.array([i[0] for i in train_data]).reshape(-1, WIDTH, HEIGHT, 1)
    Y = [i[1] for i in train_data]

    test_x = np.array([i[0] for i in test_data]).reshape(-1, WIDTH, HEIGHT, 1)
    test_y = [i[1] for i in test_data]

    # fitting model
    model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, validation_set=({'input': test_x}, {'targets': test_y}),
              batch_size=500, shuffle=True, snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    model.save(MODEL_NAME)
    print("Done")
