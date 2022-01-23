from numpy.lib.financial import ipmt
import torch
import gym, d4rl
from torch.utils.data import Dataset, DataLoader
import random
from arguments import get_args
import numpy as np
from data import KitchenDataset
import time
from learner import Learner
import torch.optim as optim
from utils import *


def test_kitchen(env_name, data_file, high_level_net, sub_skills, finetune_step):
    print('test ', env_name, ' begin...')
    traindata = KitchenDataset(data_file)
    trainloader = DataLoader(traindata, batch_size=20000, shuffle=True)

    # finetune
    high_level_meta_opt = optim.Adam(high_level_net.parameters(), lr=1e-4)
    sub_skill_meta_opts = [optim.Adam(sub_skills[i].parameters(), lr=1e-4) for i in range(len(sub_skills))]

    high_level_net.train()
    for i in range(len(sub_skills)):
        sub_skills[i].train()

    for step in range(finetune_step):
        spt_state, spt_action = next(iter(trainloader))
        spt_state, spt_action = spt_state.cuda().float().reshape(-1, 30), spt_action.cuda().float().reshape(-1, 9)

        high_fast_weights = high_level_net.parameters()
        sub_skill_fast_weights = []
        for i in range(len(sub_skills)):
            sub_skill_fast_weights.append(sub_skills[i].parameters())

        # fine-tune high-level net
        with torch.no_grad():
            cat_from_sub = get_sub_skill_cat_from_subskills(sub_skills, sub_skill_fast_weights, spt_state, spt_action)
        high_level_net.train()
        cat_from_high = high_level_net(spt_state, vars=high_fast_weights, bn_training=True)
        loss_finetune_high = torch.nn.CrossEntropyLoss()(cat_from_high, cat_from_sub)
        if loss_finetune_high.item() > 0:
            high_level_meta_opt.zero_grad()
            loss_finetune_high.backward()
            torch.nn.utils.clip_grad_norm_(high_level_net.parameters(), max_norm=10, norm_type=2)
            high_level_meta_opt.step()

        # fine-tune sub-skills
        high_fast_weights = high_level_net.parameters()
        with torch.no_grad():
            high_level_net.eval()
            cat_from_high = high_level_net(spt_state, vars=high_fast_weights, bn_training=True)
            cat_from_high = torch.argmax(cat_from_high, dim=1).reshape(spt_action.shape[0], 1)
        for i in range(len(sub_skills)):
            sub_skills[i].train()
            pred = sub_skills[i](spt_state, vars=sub_skill_fast_weights[i], bn_training=True)
            # get mask of sub_skill_i
            mask = cat_from_high.eq(i)
            mask = mask.repeat(1, pred.shape[-1])
            masked_output = pred.mul(mask)
            masked_action = spt_action.mul(mask)
            loss = torch.nn.MSELoss()(masked_output, masked_action)
            if loss.item() == loss.item(): 
                sub_skill_meta_opts[i].zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sub_skills[i].parameters(), max_norm=10, norm_type=2)
                sub_skill_meta_opts[i].step()

    # test
    high_level_net.eval()
    for i in range(len(sub_skills)):
        sub_skills[i].eval()

    env = gym.make(env_name)
    env.reset()

    # skills = []
    # hard_skills = []

    all_rews = []
    max_rew = 0
    print('testing...')
    for tries in range(5):
        total_rew = 0
        obs = env.reset()
        for i in range(300):
            env.render()
            obs = obs[:30]
            cat_from_high = high_level_net(torch.from_numpy(obs).cuda().float().unsqueeze(0), vars=high_fast_weights, bn_training=False)
            # skill_probs = torch.nn.Softmax()(cat_from_high).detach().cpu().numpy()[0]
            cat_from_high = torch.argmax(cat_from_high, dim=1)[0]
            act = sub_skills[cat_from_high](torch.from_numpy(obs).unsqueeze(0).cuda().float(), vars=sub_skill_fast_weights[cat_from_high], bn_training=False).detach().cpu().numpy()[0]
            obs, rew, done, info = env.step(act)
            total_rew += rew
            print(i,cat_from_high,rew)

            # skills.append(skill_probs)
            # hard_skills.append(cat_from_high.detach().cpu().numpy())
        print(env_name, ' finetune_step=', finetune_step, ' rew: ', total_rew)
        all_rews.append(total_rew)
        if max_rew < total_rew:
            max_rew = total_rew

    # np.save('skills.npy', np.array(skills))
    # np.save('kitchen_hard_skills.npy', np.array(hard_skills))


    env.close()

    return np.array(all_rews).sort()[-5:]


if __name__ == '__main__':
    args = get_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False  # for maml
    np.random.seed(args.seed)
    random.seed(args.seed)
    high_config = [
        ('linear', [512, 30]),
        ('bn', [512]),
        ('relu', [True]),
        ('linear', [512, 512]),
        ('bn', [512]),
        ('relu', [True]),
        ('linear', [512, 512]),
        ('bn', [512]),
        ('relu', [True]),
        ('linear', [4, 512]),
    ]

    sub_skill_config = [
        ('linear', [512, 30]),
        ('bn', [512]),
        ('relu', [True]),
        ('linear', [512, 512]),
        ('bn', [512]),
        ('relu', [True]),
        ('linear', [512, 512]),
        ('bn', [512]),
        ('relu', [True]),
        ('linear', [9, 512]),
    ]

    high_level_net = Learner(high_config).cuda()
    sub_skills = [Learner(sub_skill_config).cuda() for _ in range(args.test_skill_num)]

    demo_file = './kitchen_demos/'+args.test_suite+'_test.npy'
    env_name = 'kitchen-'+args.test_suite+'-v0'

    
    model_name = args.test_suite + '_' + str(args.seed) + '_' + str(args.test_skill_num)
    if args.continuous:
        model_name += '_continuous'
    if args.soft:
        model_name += '_soft'

    high_model_name = args.output_dir+'/high/'+model_name
    sub_skill_name = args.output_dir+'/sub_skill/'+model_name
    # load_model
    high_level_net = torch.load(high_model_name+'.pkl', map_location='cuda:0')
    for i in range(args.test_skill_num):
        sub_skills[i] = torch.load(sub_skill_name+str(i)+'.pkl', map_location='cuda:0')


    # test
    rew = test_kitchen(env_name, demo_file, high_level_net, sub_skills, args.finetune_step)
    print('test_skill_num=', args.test_skill_num, ' suite=', args.test_suite, ' finetune_step=', args.fine_tune_step, ' rew=', rew)
