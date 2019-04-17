# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 08:44:03 2019

@author: Usuario
"""

import torch

class SLP(torch.nn.Module):
    """
    SLP (Single Layer Perceptron) para aproximación de funciones
    """
    def __init__(self, input_shape, out_shape, device = torch.device('cpu')):
        """
        :param input_shape: Tamaño o forma de los datos de entrada
        :param output_shape: Tamaño o forma de lso datos de salida
        :param device: El dispositivo ("Cpu" , "cda") que la slp utiliza para almacenar los inputs de iteración
        """
        super(SLP, self).__init__()
        self.device = device
        self.input_shape = input_shape[0]
        self.hidden_shape = 40
        self.linear1 = torch.nn.Linear(self.input_shape, self.hidden_shape) # Esto representa la función de activación (por default es RELU)
        self.out = torch.nn.Linear(self.hidden_shape, out_shape)
            
    
    def forward(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        x = torch.nn.functional.relu(self.linear1(x))
        x = self.out(x)
        
        return x
    
        