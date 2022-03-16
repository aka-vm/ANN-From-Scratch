import numpy as np
from typing import List, Dict, Optional, Union

from time import time as teltm
from layer import Layer
from utils import *

hidden_layer_ = List[Dict[str, Union[str, int]]]
num = Union[int, float]



class Network:

    n_layers: int
    default_hidden_layers_info: hidden_layer_ = [
                {'nodes': 4 ,'activation': 'relu'},
                {'nodes': 5 ,'activation': 'relu'},]
    
    def loss(self, AL: np.ndarray, y: np.ndarray) -> np.ndarray: ...

    def loss_prime(self, AL: np.ndarray, y: np.ndarray) -> np.ndarray:
        return AL - y

    def check_data_for_errors(self, X: np.ndarray, y: np.ndarray):
        if X.shape[0] != self.n_x:
            raise ValueError(
                f"X is not compitable with this Network. \nX.shape[0] = {X.shape[0]}; Expected {self.n_x}")

        if y.shape[0] != self.n_y:
            raise ValueError(
                f"y is not compitable with this Network. \nY.shape[0] = {y.shape[0]}; Expected {self.n_y}")

        if X.shape[1] != y.shape[1]:
            raise ValueError(
                f"X and y don't have the same sample space.\nm_x = {X.shape[1]} ≠ m_y = {y.shape[1]}"
            )

    def __init__(
        self, 
        n_x: int, 
        n_y: int, 
        hidden_layers_info: Optional[hidden_layer_] = None,
        random_seed: Optional[num] = None,
        cost_fxn: str = "cross_entropy"):
        """
        n_x: int 
            number of cols in input data X.

        n_y: int 
            number of cols in Output data Y.

        layer_info: List[Dict{nodes, activation}] 
            Gives info about hidden layers of its Nodes and Activation Functions
            use blank_list `[]` for zero hidden layers
            Note: final layer will use softmax(n_y>1) or sigmoid(n_y=1).
            DEFAULT -> [
                {'nodes': 4 ,'activation': 'relu'},
                {'nodes': 5 ,'activation': 'relu'},
            ]"""
        np.random.seed(random_seed)
        self.n_x =  n_x
        self.n_y =  n_y
        self.layers: List[Layer] = []
        self.init_layers(n_x, n_y, hidden_layers_info)
        self.loss = eval(cost_fxn)

    def init_layers(
        self, 
        n_x: int, 
        n_y: int, 
        hidden_layers_info: Optional[hidden_layer_]):
        # hidden_layers_info = self.default_hidden_layers_info if hidden_layers_info==None else hidden_layers_info
        
        n_prev = n_x


        if hidden_layers_info == None:
            hidden_layers_info = self.default_hidden_layers_info

        if len(hidden_layers_info)>0:
            for hidden_layer in hidden_layers_info:
                n_cur = hidden_layer["nodes"]
                self.layers.append(
                    Layer(n_prev, 
                          n_cur, 
                          hidden_layer["activation"])
                )
                n_prev = n_cur

        self.layers.append(
                    Layer(
                        n_prev, n_y, 
                        "sigmoid" if n_y==1 else "softmax")
                )
        self.n_layers = len(self.layers)

    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        learning_rate: float = 0.05,
        epoch: int = 5000, 
        print_cost: bool = False):
        
        self.check_data_for_errors(X, y)

        layers = self.layers

        start_time = teltm()

        for i in range(epoch):
            AL = self.network_forward(X)
            self.network_backward(AL, y)
            self.update_params(learning_rate)


            if print_cost and ((i+1) % (epoch//20) == 0 or i == epoch-1):
                cost = self.loss(AL, y)
                total_time = teltm() - start_time
                total_time = round(total_time, 2)
                unit = 's'
                if total_time > 100:
                    total_time = int(total_time)
                    unit = "min"
                    total_time = f"{total_time//60}:{total_time%60}"

                print(f"Cost after iteration {i+1}: {np.squeeze(cost).round(6)}      took {total_time} {unit}")
                # print(
                #     f"""Accracy = {
                #         round(
                #             get_accuracy(
                #                 self.predict(X), y
                #                 ), 4)}
                #     """)

        self.remove_cache()

        return self

    def network_forward(
        self, 
        X: np.ndarray, 
        training: bool = True) -> np.ndarray:

        layers = self.layers
        A = X
        L = len(layers)

        for layer in layers:
            A_prev = A
            A = layer.linear_forward(A_prev, training=training)

        return A

    def network_backward(self, AL: np.ndarray, y: np.ndarray):
        layers = self.layers
        L = len(layers)
        m = AL.shape[1]
        y = y.reshape(AL.shape)
        dA_prev = self.loss_prime(AL, y)     # it dZ, i know

        for layer in layers[::-1]:
            dA_prev = layer.linear_backwards(dA_prev)

    def update_params(self, learning_rate: float):
        layers = self.layers

        for layer in layers:
            layer.update_params(learning_rate)

    def predict(self, X: np.ndarray) -> np.ndarray:
        AL = self.network_forward(X, training=False)
        n_y = AL.shape[0]

        if n_y > 1:
            return np.argmax(AL, 0)
        
        return AL>0.5

    def remove_cache(self):
        layers = self.layers
        for layer in layers:
            layer.A = None
            layer.A_prev = None
            layer.Z = None

