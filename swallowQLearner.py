# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 09:43:58 2019

@author: Usuario
"""
import torch, gym , random
import numpy as np
from Libs.perceptron import SLP
from utils.decay_schedule import LinearDecaySchedule
from utils.experience import ExperienceMemory, Experience

MAX_NUM_EP = 100000
MAX_STEP_PER_EP = 300

class SwallowQLearner(object): 
    def __init__(self, env, learning_rate = 0.005, gamma = 0.98):
        self.gamma = gamma
        self.learning_rate = learning_rate
        
        self.epsilon_max = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = LinearDecaySchedule(initial_value = self.epsilon_max,
                                                 final_value = self.epsilon_min,
                                                 max_step = 0.5 * MAX_NUM_EP * MAX_STEP_PER_EP)
        
        self.step_num = 0
        self.policy = self.epsilon_greedy_Q
        
        self.obs_shape = env.observation_space.shape
        self.action_shape = env.action_space.n
        
        self.Q = SLP(self.obs_shape, self.action_shape)
        self.Q_Optimizer = torch.optim.Adam(self.Q.parameters(), lr = learning_rate)
        
        self.memory = ExperienceMemory(capacity = int(1e5))
        self.device = torch.device("cuda" if torch.cuda.is_available () else "cpu")

    def get_Action(self, obs):
        return self.policy(obs)
    
    def epsilon_greedy_Q(self, obs):
        if random.random() < self.epsilon_decay(self.step_num):
            action = random.choice([a for a in range(self.action_shape)])
        else:
            action = np.argmax(self.Q(obs).data.to(torch.device('cpu')).numpy)
        
        return action
    def replay_experience(self, batch_size):
        """
        Vuelve a jugar usando experiencias aleatorias (Desde la memoria almacenada)
        :param batch_size: Tamaño de muestra a tomar de la memoria
        :return: 
        """
        experience_batch = self.memory.sample(batch_size)
        self.learn_from_batch_experience(experience_batch)        
    
    def learn_from_batch_experience(self, experiences):
        """
        Actualizar la NN con experiencias anteriores
        :param experiences: Es un arreglo de experiencias
        :return:
        """
        batch_xp = Experience(*zip(*experiences))
        
        obs_batch = np.array(batch_xp.obs)
        action_batch= np.array(batch_xp.action)
        reward_batch = torch.from_numpy(np.array(batch_xp.reward)).float()
        next_obs_batch = np.array(batch_xp.next_obs)
        done_batch = np.array(batch_xp.done)
        
        td_target = reward_batch + torch.from_numpy((~done_batch).astype(int)).float() * \
                    torch.from_numpy(np.tile(self.gamma, len(next_obs_batch))).float() * \
                    self.Q(next_obs_batch).detach().max(1)[0].data
        
        td_target = td_target.to(self.device)
        action_idx = torch.from_numpy(action_batch).to(self.device)
        
        td_error = torch.nn.functional.mse_loss(
                self.Q(obs_batch).gather(1, action_idx.view(-1, 1).long()),
                td_target.float().unsqueeze(1))
        
        self.Q_Optimizer.zero_grad()
        td_error.mean().backward()
        self.Q_Optimizer.step()
        
    def learn(self, obs, action, reward, next_obs):
        td_target = reward + self.gamma * torch.max(self.Q(next_obs))
        td_error = torch.nn.functional.mse_loss(self.Q(obs)[action], td_target)
        
        self.Q_Optimizer.zero_grad()
        td_error.backward()
        self.Q_Optimizer.step() 

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    agent = SwallowQLearner(env)
    
    first_episode = True
    episode_rewards = list()
    
    for episode in range(MAX_NUM_EP):
        obs = env.reset()
        total_reward = 0.0
        
        for step in range(MAX_STEP_PER_EP):
            #env.render()
            action = agent.get_Action(obs)
            next_obs, reward, done, info = env.step(action)
            
            agent.memory.store(Experience(obs, action, reward, next_obs, done))
            agent.learn(obs, action, reward, next_obs)
            
            obs = next_obs
            total_reward += reward
            
            if done:
                if first_episode:
                    max_reward = total_reward
                    first_episode = False
                
                episode_rewards.append(total_reward)
                
                if total_reward > max_reward:
                    max_reward = total_reward
                
                print("\n Episodio {} finalizado con {} iteraciones, Recompensa= {}, Recompenda media= {}, Mejor recompensa= {}".
                      format(episode, step+1, total_reward, np.mean(episode_rewards), max_reward))
                
                if agent.memory.get_size() > 100:
                    agent.replay_experience(32)
                
                break
    
    env.close()