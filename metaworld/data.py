import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import metaworld.envs.mujoco.env_dict as _env_dict
from arguments import get_args
from metaworld.policies import *


''' 测试环境编号为[2,3,14,16,22]
'''

def get_all_env_names():
    envs = []
    for env_name in _env_dict.MT50_V2:
        envs.append(env_name)

    return envs

def get_training_and_testing_env_number():
    training_number = [i for i in range(50)]
    testing_number = [2,3,14,16,22]
    for i in testing_number:
        training_number.remove(i)

    return training_number, testing_number

class DMILDatasetForOneTask(Dataset):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.data = {}

        max_length = 0
        # reasonable_trails = [i for i in range(2000)]
        # max_length = 490
        reasonable_trails = []
        for trails in range(2000):
            dones = np.load(("./demos/"+env+'/'+str(trails)+"/dones.npy"))
            if len(np.where(dones==True)[0]) > 0:   # exclude failure trails
                trail_length = np.where(dones==True)[0][0]

                standard_length = 100   # only get trails that is shorter than 100 steps
                if env == 'peg-unplug-side-v2':
                    standard_length = 140
                if env == 'bin-picking-v2':
                    standard_length = 160

                if trail_length <= standard_length:
                    reasonable_trails.append(trails)
                    # extend some demo to tell the robot how to do when complete the task 
                    trail_length+= 10
                
                    if trail_length >= max_length:
                        max_length = trail_length
        
        for trails in reasonable_trails:
            # we load one more step of obs than action
            self.data[str(trails)] = {}
            self.data[str(trails)]['obs'] = np.load("./demos/"+env+'/'+str(trails)+"/obs.npy")[:max_length-1,:]
            self.data[str(trails)]['act'] = np.load("./demos/"+env+'/'+str(trails)+"/acts.npy")[:max_length-1,:]

        self.trail_length = max_length
        self.length = len(reasonable_trails)
        self.reasonable_trails = reasonable_trails

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # obs in shape [length, state_shape]
        # act in shape [length, action_shape]
        
        return self.data[str(self.reasonable_trails[index])]['obs'], self.data[str(self.reasonable_trails[index])]['act']
