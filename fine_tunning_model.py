import numpy as np

import data_processing


# TODO get to know what kind of data shape should be passed to EfficientNet model to allow fine-tuning.
# TODO In the meantime this module is pretty dead
def fit_model():
    # Static parameters for data pipeline and fitting model
    BATCH_SIZE = 500
    # preparing model configuration
    # TODO model configuration can be changed to more accessiable option
    LR = 1e-5
    EPOCHS = 2
    MODEL_NAME = 'EfficientNet_model'

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

    # setting train and test data for training model
    X = np.array([i[0] for i in training_data]).reshape(-1, 128, 128, 3)
    Y = [i[1] for i in training_data]

    # fitting model
    model.fit(x=[X], y=[Y], epochs=EPOCHS, validation_split=0.1, verbose=1,
              batch_size=BATCH_SIZE, shuffle=True)
    # saving model
    model.save(MODEL_NAME)
    print("Done")
