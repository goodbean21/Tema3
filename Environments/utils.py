# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 16:05:39 2019

@author: Usuario
"""
import gym, cv2
import numpy as np

class ResizeReshapeFrames(gym.ObservationWrapper):
    def __init__(self, env):
        super(ResizeReshapeFrames, self).__init__(env)
        
        if len(self.observation_space.shape) ==3:
            self.desired_width = 84
            self.desired_height = 84
            self.desired_channels = self.observation_space.shape[2]
            
            self.observation_space = gym.spaces.Box(0, 255, 
                                                    (self.desired_channels, 
                                                     self.desired_height, 
                                                     self.desired_width), 
                                                     dtype = np.uint8)
            
        gym.ObservationWrapper.__init__(self, env)
        
        
    def observation(self, obs):
        if len(obs.shape) == 3:
            obs = cv2.resize(obs, (self.desired_width, self.desired_height))
            
            if obs.shape[2] < obs.shape[0]:
                obs = np.reshape(obs, (obs.shape[2], obs.shape[1], obs.shape[0]))
        
        return obs