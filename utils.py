from cmath import log
import numpy as np
import pickle
import time

def sigmoid(Z: np.ndarray):
    return 1/(1 + np.exp(-Z))

def sigmoid_prime(Z: np.ndarray):
    # return sigmoid(Z) * (1 - sigmoid(Z))
    return Z

def relu(Z: np.ndarray):
    return Z * (Z > 0)

def relu_prime(Z: np.ndarray):
    return (Z>0).astype(int)

def leaky_relu(Z: np.ndarray, a: int =0.01):
    return np.max((Z, a*Z), axis=0)

def leaky_relu_prime(Z: np.ndarray, a: int =0.01):
    dZ = (Z>0).astype(int)
    return dZ + (Z<0).astype(int) * a

def tanh(Z: np.ndarray):
    return np.tanh(Z)

def tanh_prime(Z: np.ndarray):
    return 1 - np.tanh(Z)

# def softmax(Z: np.ndarray):
#     exp = np.exp(Z)
#     return exp / np.sum(exp, axis=0)

def softmax(Z: np.ndarray):
    #Z - np.max(Z)
    e_x = np.exp(Z)
    return e_x / e_x.sum(axis=0)

def softmax_prime(Z: np.ndarray) -> int:
    return Z

def cross_entropy(AL: np.ndarray, y: np.ndarray):
    AL = np.clip(AL, 1e-12, 1. - 1e-12)
    k, m = y.shape

    if k == 1:
        cost = -np.mean(
            y * np.log(AL) + (1-y) * np.log(1-AL)
        )
    else:
        cost = (-1/m) * np.sum(
            np.sum(y * np.log(AL), axis=0)
        )

    cost = np.squeeze(cost)
    return cost

def pickle_model(model, file_prefix: str = "pickled_obj", dir_path: str = "pickles"):
    unique_suffix = int(time.time()*10) - 10000000000
    pic_file = f"{file_prefix}-{unique_suffix}.sav"
    
    pickle.dump(model, open(f"{dir_path}/{pic_file}", 'wb'))

    return f"{dir_path}/{pic_file}"


def unpickle_model(file_name: str):
    
    model = pickle.load(open(file_name, 'rb'))

    return model

def one_hot(Y: np.ndarray) -> np.ndarray:
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def get_accuracy(Y_: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.sum(Y_ == Y) / Y.size