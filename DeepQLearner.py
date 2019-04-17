# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:52:12 2019

@author: Usuario
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 09:43:58 2019

@author: Usuario
"""
import torch, gym , random
from datetime import datetime
import numpy as np

from argparse import ArgumentParser

from Libs.perceptron import SLP
from Libs.cnn import CNN

from utils.decay_schedule import LinearDecaySchedule
from utils.experience import ExperienceMemory, Experience
from utils.params_manager import ParamsManager

import Environments.atari as Atari
import Environments.utils as env_utils

from tensorboardX import SummaryWriter

# Parseador de argumentos
args = ArgumentParser("DeepQLearning")
args.add_argument("--params-file", help = "Path del fichero JSON de parámetros. El valor por defecto es parameters.json",
                  default="param.json", metavar = "PFILE")
args.add_argument("--env", help = "Entorno de ID de Atari disponible en OpenAI Gym. El valor por defecto será SeaquestNoFrameskip-v4",
                  default = "SeaquestNoFrameskip-v4", metavar="ENV")
args.add_argument("--gpu-id", help = "ID de la GPU a utilizar, por defecto 0", default = 0, type = int, metavar = "GPU_ID")
args.add_argument("--test", help = "Modo de testing para jugar sin aprender. Por defecto está desactivado", 
                  action = "store_true", default = False)
args.add_argument("--render", help = "Renderiza el entorno en pantalla. Desactivado por defecto", action="store_true", default=False)
args.add_argument("--record", help = "Almacena videos y estados de la performance del agente", action="store_true", default=False)
args.add_argument("--output-dir", help = "Directorio para almacenar los outputs. Por defecto = ./trained_models/results",
                  default = "./trained_models/results")
args = args.parse_args()

# Parametros globales
manager = ParamsManager(args.params_file)
agent_params = manager.get_agent_params()
global_step_num = 0

#Fichero del log
summary_filename_prefix = agent_params['summary_filename_prefix']
summary_filename = summary_filename_prefix + args.env + datetime.now().strftime("%y-%m-%d-%H-%M")

# Sumary writter de TBX
writer = SummaryWriter(summary_filename)

manager.export_agent_params(summary_filename +"/"+"agent_params.json")
manager.export_env_params(summary_filename +"/"+"env_params.json")

# Habilitar entranamiento por GPU
use_cuda = agent_params['use_cuda']
device = torch.device("cuda:" +str(args.gpu_id) if torch.cuda.is_available() and use_cuda else "cpu")

# Hbilitar la semilla aleotoria
seed = agent_params['seed']
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available() and use_cuda:
    torch.cuda.manual_seed_all(seed)

class DeepQLearner(object): 
    def __init__(self, obs_shape, action_shape, params):
        
        self.params = params
        self.gamma = self.params['gamma']
        self.learning_rate = self.params['learning_rate']
        self.best_reward_mean = -float("inf")
        self.best_mean = -float("inf")
        self.training_steps_completes = 0
        
        self.epsilon_max = self.params['epsilon_max']
        self.epsilon_min = self.params['epsilon_min']
        self.epsilon_decay = LinearDecaySchedule(initial_value = self.epsilon_max,
                                                 final_value = self.epsilon_min,
                                                 max_step = self.params['epsilon_decay_final_step'])
        
        self.step_num = 0
        self.policy = self.epsilon_greedy_Q
        
        if len(self.obs_shape) == 1: ## Solo se existe 1D en el espacio de observaciones
            self.DQN = SLP
            
        elif len(self.obs_shape) == 3: ## El estado de observaciones es una imagen o un objeto 3D
            self.DQN = CNN
            
        self.Q = self.DQN(obs_shape, action_shape, device).to(device)
        self.Q_Optimizer = torch.optim.Adam(self.Q.parameters(), lr = self.learning_rate)
        
        if self.params['use_target_network']:
            self.Q_target = self.DQN(obs_shape, action_shape, device).to(device)
        
        self.memory = ExperienceMemory(capacity = int(self.params['experience_memory_size']))

    def get_Action(self, obs):
        obs = np.array(obs)
        obs = obs/255.0;
        if len(obs.shape) == 3:
            if obs.shape[2] < obs.shape[0]:
                obs =  obs.reshape(obs.shape[2], obs.shape[1], obs.shape[0])
            
            obs = np.expand_dims(obs, 0)
        
        return self.policy(obs)
    
    def epsilon_greedy_Q(self, obs):
        writer.add_scalar("DQL/Epislon", self.epsilon_decay(self.step_num), self.step_num)
        
        self.step_num += 1
        
        if random.random() < self.epsilon_decay(self.step_num) and not self.params['test']:
            action = random.choice([a for a in range(self.action_shape)])
        
        else:
            action = np.argmax(self.Q(obs).data.to(torch.device('cpu')).numpy)
        
        return action

    def learn(self, obs, action, reward, next_obs, done):
        if done:
            td_target = reward + 0.0
            
        else:
            td_target = reward + self.gamma * torch.max(self.Q(next_obs))
            
        td_error = torch.nn.functional.mse_loss(self.Q(obs)[action], td_target)
        
        self.Q_Optimizer.zero_grad()
        td_error.backward()
        
        writer.add_scalar("DQL/td_error", td_error.mean(), self.step_num)
        
        self.Q_Optimizer.step() 

    
    def replay_experience(self, batch_size = None):
        """
        Vuelve a jugar usando experiencias aleatorias (Desde la memoria almacenada)
        :param batch_size: Tamaño de muestra a tomar de la memoria
        :return: 
        """
        batch_size = batch_size if batch_size is not None else self.params['replay_batch_size'] 
        
        experience_batch = self.memory.sample(batch_size)
        self.learn_from_batch_experience(experience_batch)        
        self.training_steps_completes += 1
    
    def learn_from_batch_experience(self, experiences):
        """
        Actualizar la NN con experiencias anteriores
        :param experiences: Es un arreglo de experiencias
        :return:
        """
        batch_xp = Experience(*zip(*experiences))
        
        obs_batch = np.array(batch_xp.obs)/255.0
        action_batch= np.array(batch_xp.action)
        reward_batch = torch.from_numpy(np.array(batch_xp.reward)).float()
        if self.params['clip_rewards']:
            reward_batch = np.sign(reward_batch)
            
        next_obs_batch = np.array(batch_xp.next_obs)/255.0
        done_batch = np.array(batch_xp.done)
        
        if self.params['use_target_network']:
            if self.step_num % self.params['target_network_freq'] == 0:
                self.Q_target.load_state_dict(self.Q.state_dict)
                td_target = reward_batch + torch.from_numpy((~done_batch).astype(int)).float() * \
                            torch.from_numpy(np.tile(self.gamma, len(next_obs_batch))).float() * \
                            self.Q_target(next_obs_batch).detach().max(1)[0].data
        
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
    
    def save(self, env_name):
        file_name = self.params['save_dir'] + "DQL_" +env_name+".ptn"
        agent_state = {
                    "Q" : self.Q.state_dict(),
                    "best_reward_mean" : self.best_reward_mean,
                    "best_reward" : self.best_reward
                    }
        torch.save(agent_state, file_name)
        print("Estado del agente esta guardado en: ", file_name)
        
    def load(self, env_name):
        file_name = self.params['load_dir'] + "DQL_" +env_name+".ptn"
        agent_state = torch.load(file_name, map_location = lambda storage, loc: storage)
        self.Q.load_state.dict(agent_state["Q"])
        self.Q.to(device)
        self.best_reward_mean = agent_state["best_reward_mean"]
        self.best_reward = agent_state["best_reward"]
        print("El modelo se ha cargado desde: ", file_name, ". Que tiene una mejor recompensa media: ",
              self.best_reward_mean, ". Recompensa máxima: ", self.best_reward)

if __name__ == "__main__":
    env_conf = manager.get_env_params()
    env_conf["env_game"] = args.env
    
    if args.test:
        env_conf["episodic_life"] = False
    
    reward_type = "LIFE" if env_conf["episodic_life"] else "GAME"
    
    custom_region_available = False
    for key, value in env_conf["useful_region"].items():
        if key in args.env:
            env_conf["useful_region"] = value
            custome_region_available = True
            break
    
    if custom_region_available is not True:
        env_conf["useful_region"] = env_conf["useful_region"]["Default"]
    
    print("Configuración a utilizar: ", env_conf)
    
    atari_env = False
    for game in Atari.get_games_list():
        if game.replace("_", "") in args.env.lower():
            atari_env = True
    
    if atari_env:
        env = Atari.make_env(args.env, env_conf)
    
    else:
        env = env_utils.ResizeReshapeFrames(gym.make(args.env))
    
    
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.n
    
    agent_params = manager.get_agent_params()
    agent_params["test"] = args.test
    
    agent = DeepQLearner(obs_shape, action_shape, agent_params)
    
    episode_rewards = list()
    previous_checkpoint_mean_ep_rew = agent.best_mean
    num_improved_episodes_before_checkpoint = 0
    if agent_params['load_trained_model']:
        try:
            agent.load(env_conf['env_name'])
            previous_checkpoint_mean_ep_rew = agent.best_mean
        
        except FileNotFoundError:
            print("ERROR: No existe ningun modelo entrenado para este entorno")
    
    episode = 0
    while global_step_num < agent_params['max_training_step']:
        obs = env.step()
        total_reward = 0.0
        done = False
        step = 0
        
        while not done:
            if env_conf['render'] or args.render:
                env.render()
                
            action = agent.get_Action(obs)
            next_obs, reward, done, info = env.step(action)
            
            agent.memory.store(Experience(obs, action, reward, next_obs, done))
            
            obs = next_obs
            total_reward += reward
            step += 1
            global_step_num += 1
            
            if done:
                episode += 1
                episode_rewards.append(total_reward)
                
                if total_reward > agent.best_reward:
                    agent.best_reward = total_reward
                    
                if np.mean(episode_rewards) > previous_checkpoint_mean_ep_rew:
                    num_improved_episodes_before_checkpoint += 1
                
                if num_improved_episodes_before_checkpoint >= agent_params['save_freq']:
                    previous_checkpoint_mean_ep_rew = np.mean(episode_rewards)
                    agent.best_mean = np.mean(episode_rewards)
                    agent.save(env_conf['env_name'])
                    num_improved_episodes_before_checkpoint = 0
                
                print("\n Episodio #{}, finalizado con {} iteraciones. Con {} estados: recompensa = {}, recompensa media = {:.2f}, mejor recompensa = {}".
                      format(episode, step + 1, reward_type, total_reward, np.mean(episode_rewards), agent.best_reward))
                
                writer.add_scalar("main/ep_reward", total_reward, global_step_num)
                writer.add_scalar("main/mean_ep_reward", np.mean(episode_rewards), global_step_num)
                writer.add_scalar("main/max_ep_reward", agent.best_reward, global_step_num)
                
                if agent.memory.get_size() >= 2 * agent_params['replay_start_size'] and not args.test:
                    agent.replay_experience()
                
                break
    
    env.close()
    writer.close()