from keras.models import Sequential
from keras.layers import Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D



# dense layers of size N are replaced with conv layers of N filters. 
def get_conv(input_shape=(64,64,3), heatmapping=True):
    '''
    heatmapping: default True. When True, no Flatten layer is added, the output from 
    the model could be used to draw a heatmap. 

    '''

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,input_shape=input_shape, output_shape=input_shape))
    model.add(Convolution2D(10, 3, 3, activation='relu', name='conv1',input_shape=input_shape, 
                            border_mode="same"))
    model.add(Convolution2D(10, 3, 3, activation='relu', name='conv2',border_mode="same"))
    model.add(MaxPooling2D(pool_size=(8,8)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(128,8,8,activation="relu",name="dense1")) # This replaces Dense(128)
    model.add(Dropout(0.5))
    model.add(Convolution2D(1,1,1,name="dense2", activation="tanh")) # This replaces Dense(1)

    if not heatmapping:
        model.add(Flatten())

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    return model