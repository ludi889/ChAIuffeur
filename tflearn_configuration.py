import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization


def trainingmodel(width, height, lr):
    network = input_data(shape=[None, width, height, 1], name='input')
    network = conv_2d(network, 96, 11, strides=4, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 256, 5, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = conv_2d(network, 384, 3, activation='relu')
    network = conv_2d(network, 256, 3, activation='relu')
    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)
    network = fully_connected(network, 4096, activation='tanh')
    network = dropout(network, 0.25)
    network = fully_connected(network, 3, activation='softmax')
    sgd = tflearn.optimizers.SGD(learning_rate=0.01, lr_decay=0.96, decay_step=100)
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=lr, name='targets')
    model = tflearn.DNN(network, checkpoint_path='model_training_model',
                        max_checkpoints=1, tensorboard_verbose=2, tensorboard_dir='log')

    return model
