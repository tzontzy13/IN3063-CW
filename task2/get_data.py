import keras

# Referece: Aayush Agrawal. 2020
# Building Neural Network from scratch | by Aayush Agrawal | Towards Data Science.
# [ONLINE] Available at: https://towardsdatascience.com/building-neural-network-from-scratch-9c88535bf8e9
# [Accessed 15 December 2020]

# get data as in the refference above, under the "Full network" section
def load_dataset():
    
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    # Normalizes the data from the [0-255] range to the decimal [0-1] range
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.

    # Reserves the last 10000 training examples as the validation data
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    X_train = X_train.reshape([X_train.shape[0], -1])
    X_val = X_val.reshape([X_val.shape[0], -1])
    X_test = X_test.reshape([X_test.shape[0], -1])
        
    return X_train, y_train, X_val, y_val, X_test, y_test