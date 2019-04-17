# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:28:24 2019

@author: Usuario
"""
import torch
class CNN(torch.nn.Module):
    """
    Una red neuronal convolucional que tomará decisiones segun 
    los pixeles de la imagen
    """
    def __init__(self, input_shape, output_shape, device = "cpu"):
        """
        :param input_shape  : Dimensión de la imagen, que supondremos viene reescalada
        a canales de 84 x 84 pixeles
        :param output_shape : Dimensión del espacio de acciones
        :param device       : Donde la red neuronal se debe almacenar en cada iteración 
        """
        # input_shape CX84X84
        super(cnn, self).__init__()
        self.device = device
        self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(input_shape[0], 64, kernel_size = 4, stride = 2, padding = 1),
                torch.nn.ReLU()
                )
        
        self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(64, 32, kernel_size = 4, stride = 2, padding = 0),
                torch.nn.ReLU()
                )
        
        self.layer3 = torch.nn.Sequential(
                torch.nn.Conv2d(32, 32, kernel_size = 3, stride = 1, padding = 0),
                torch.nn.ReLU()
                )
        
        self.out = torch.nn.Linear(18*18*32, output_shape)
        
    def forward(self, x):
        x = torch.from_numpy(x).float().to(self.device)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = x.view(x.shape[0], -1)
        x = self.out(x)
        
        return x
    
    