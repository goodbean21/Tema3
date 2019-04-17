# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 12:32:30 2019

@author: Usuario
"""

class LinearDecaySchedule(object):
    
    def __init__(self, initial_value, final_value, max_step):
        assert initial_value > final_value, "Err: initial valule needs to be larger than final value"
        
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_factor = (initial_value - final_value)/max_step
    
    def __call__(self, step_num):
        current_value = self.initial_value - step_num * self.decay_factor
        if (current_value < self.final_value):
            current_value = self.final_value
        
        return current_value
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
   
    MAX_NUM_EPISODES = 1000
    STEPS_PER_EP = 300
    
    epsilon_initial = 1.0
    epsilon_final = 0.005
   
    linear_schedule = LinearDecaySchedule(initial_value = epsilon_initial,
                                                 final_value = epsilon_final,
                                                 max_step = 0.5 * MAX_NUM_EPISODES * STEPS_PER_EP)
    epsilon= [linear_schedule(step) for step in range(MAX_NUM_EPISODES * STEPS_PER_EP)]
    
    plt.plot(epsilon)
    plt.show()