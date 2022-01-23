import torch
from data import DMILDatasetForOneTask, get_all_env_names
from arguments import get_args
from learner import Learner
import numpy as np
import random
import copy
import os
import torch.optim as optim
from utils import *
from torch.utils.tensorboard import SummaryWriter



if __name__ == '__main__':
    args = get_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.makedirs('./few_shots', exist_ok=True)

    suite_names = ['ML10_train', 'ML10_test', 'ML45_train', 'ML45_test']

    for suite_name in suite_names:
        print('getting few_shots for suite: ', suite_name)
        if suite_name == 'ML10_train':
            all_env_names = [get_all_env_names()[i] for i in [1, 6, 11, 18, 28, 31, 33, 34, 46, 48]]
        elif suite_name == 'ML10_test':
            all_env_names = [get_all_env_names()[i] for i in [13, 19, 27, 45, 47]]
        elif suite_name == 'ML45_train':
            env_list = [i for i in range(50)]
            for testing_env in [2,3,14,16,17]:
                env_list.remove(testing_env)
            all_env_names = [get_all_env_names()[i] for i in env_list]
        elif suite_name == 'ML45_test':
            all_env_names = [get_all_env_names()[i] for i in [2,3,14,16,17]]

        print('getting training data...')
        dmil_datas = {}
        for task_name in all_env_names:
            print('getting ', task_name, '...')
            dmil_datas[task_name] = DMILDatasetForOneTask(task_name)

        name = './few_shots/' + suite_name + '_fewshots.npy'
        get_few_shots(all_env_names, dmil_datas, name,  multi_shots=10)
        
        del dmil_datas
