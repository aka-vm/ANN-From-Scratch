import numpy as np
from typing import Union

from utils import *

num = Union[int, float]

class Layer:

    def __init__(
        self, 
        input_size: int, 
        output_size: int,
        activation_fxn: str = "sigmoid"):

        # Declayering vars
        self.nodes = output_size
        self.input_size = input_size
        self.output_size = output_size
        self.W: np.ndarray
        self.b: np.ndarray
        self.Z: np.ndarray
        self.A: np.ndarray
        self.A_prev: np.ndarray
        self.dW: np.ndarray
        self.db: np.ndarray
        
        # initalizing params
        self.W = np.random.rand(
            output_size, input_size
            ) * np.sqrt(2/input_size)
        self.b = np.random.rand(
            output_size, 1
            ) * np.sqrt(2/input_size)

        # initailizing Activation and Activation Prime
        self.activation = eval(activation_fxn)
        self.activation_prime = eval(activation_fxn+"_prime")

    def linear_forward(
        self, 
        A_prev: np.ndarray, 
        training=True) -> np.array:

        Z = self.W.dot(A_prev) + self.b
        A = self.activation(Z)

        if training:
            self.Z = Z
            self.A_prev = A_prev
        return A 

    def linear_backwards(
        self, 
        dA: np.ndarray) -> np.array:

        m: int = dA.shape[1]
        
        if self.activation in [sigmoid, softmax]:
            dZ = dA
        else:
            dZ = dA * self.activation_prime(self.Z)

        dW = (1/m) * np.dot(dZ, self.A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.W.T, dZ)

        self.dW = dW
        self.db = db

        return dA_prev

    def update_params(
        self, 
        learning_rate: float = 0.01) -> np.array:
        
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * self.db

    def use_layer(
        self,
        A_prev: np.ndarray) -> np.array:
        
        return self.linear_forward(A_prev)
