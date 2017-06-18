
'''
TODO #1: Implement a function to window time series
'''

### TODO: fill out the function below that transforms the input series and window-size into a set of
### input/output pairs for use with our RNN model

def window_transform_series(series,window_size):
    # containers for input/output pairs
    X = []
    y = []

    start_index = 0
    stop_index = window_size
    for i in range(series.size - window_size):
        input_seq = series[start_index:stop_index]
        X.append(input_seq)
        y.append(series[stop_index])
        start_index += 1
        stop_index += 1

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y




'''
TODO #2: Create a simple RNN model using keras to perform regression
'''

### TODO: create required RNN model
# import keras network libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras

# given - fix random seed - so we can all reproduce the same results on our default time series
np.random.seed(0)

# TODO: build an RNN to perform regression on our time series input/output data
model = Sequential()
model.add(LSTM(5, input_shape=(window_size,1)))
model.add(Dense(1))


# build model using keras documentation recommended optimizer initialization
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# compile the model
model.compile(loss='mean_squared_error', optimizer=optimizer)




'''
TODO #3: Finish cleaning a large text corpus
'''



### TODO: list all unique characters in the text and remove any non-english ones

# find all unique characters in the text
chars = sorted(list(set(text)))

# remove as many non-english characters and character sequences as you can
punctuation = [' ', '!', ',', '.', ':', ';', '?']
text = ''.join(c for c in text if c.isalpha() or punctuation)

# shorten any extra dead space created above
text = text.replace('  ',' ')




'''
TODO #4: Implement a function to window a large text corpus
'''

### TODO: fill out the function below that transforms the input text and window-size into a set of
### input/output pairs for use with our RNN model

def window_transform_text(text,window_size,step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    start_index = 0
    stop_index = window_size
    while stop_index < len(text):
    # for i in range(int(round((len(text) - window_size)/step_size))):
        input_seq = text[start_index:stop_index]
        inputs.append(input_seq)
        outputs.append(text[stop_index])
        start_index += step_size
        stop_index += step_size

    return inputs,outputs




'''
TODO #5: Create a simple RNN model using keras to perform multiclass classification
'''

### necessary functions from the keras library
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import keras
import random

# TODO build the required RNN model: a single LSTM hidden layer with softmax activation, categorical_crossentropy loss
model = Sequential()
model.add(LSTM(200, input_shape=(window_size,len(chars))))
model.add(Dense(len(chars), activation='softmax'))

# initialize optimizer
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# compile model --> make sure initialized optimizer and callbacks - as defined above - are used
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
