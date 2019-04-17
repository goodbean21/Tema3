# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 09:27:42 2019

@author: Usuario
"""

from collections import namedtuple
import random

Experience = namedtuple("Experience", ['obs', 'action', 'reward', 'next_obs', 'done'])
 
class ExperienceMemory(object):
    """
    Un buffer que simula la memoria del agente
    """
    
    def __init__(self, capacity = int(1e6)):
        """
        :param capacity: Capacidad total de la memoria cíclica (Número máximo de experiencias almacenables)
        :return:
        """
        self.capacity = capacity
        self.memory_idx = 0 # Identificador de experiencia actual
        self.memory = []
        
    def sample(self, batch_size):
        """
        :param batch_size: Tamaño de la memoria a recuperar
        :return: Una muestrea aleatoria del tamaño batch_size de experiencias de la memoria
        """
        assert batch_size <= self.get_size(), "Err: Batch is bigger than current memory size"
        return random.sample(self.memory, batch_size)
    
    def get_size(self):
        """
        :return: Número de experiencias almacenadas
        """
        return len(self.memory)
        
    def store(self, exp):
        """
        :param exp: Objeto experiencia a ser almacenado
        :return:
        """
        self.memory.insert(self.memory_idx % self.capacity, exp)
        self.memory_idx += 1
        
        