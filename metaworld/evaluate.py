import torch
import random
from arguments import get_args
import numpy as np
import metaworld.envs.mujoco.env_dict as _env_dict
from learner import Learner
from utils import *

def get_all_envs():
    envs = {}
    env_names = []
    for env_name in _env_dict.MT50_V2:
        # print(env_name)
        env_names.append(env_name)
        envs[env_name] = _env_dict.MT50_V2[env_name]
    return env_names, envs

def test_env_simple(env_name, tau0, env_index, high_level_net, sub_skills, k_shot, finetune_step):
    print('test ', env_name, ' begin...')

    for i in range(env_index*10, env_index*10 + k_shot):
        if i == env_index*10:
            spt_state = torch.from_numpy(tau0[i][:, 0:39]).cuda().float()
            spt_action = torch.from_numpy(tau0[i][:, 39:]).cuda().float()
        else:
            spt_state = torch.cat((spt_state, torch.from_numpy(tau0[i][:, 0:39]).cuda().float()))
            spt_action = torch.cat((spt_action, torch.from_numpy(tau0[i][:, 39:]).cuda().float()))

    spt_state = spt_state.reshape(-1, 39)
    spt_action = spt_action.reshape(-1, 4)

    # fine_tune
    high_level_net.train()
    for i in range(len(sub_skills)):
        sub_skills[i].train()
    high_fast_weights = high_level_net.parameters()
    sub_skill_fast_weights = []
    for i in range(len(sub_skills)):
        sub_skill_fast_weights.append(sub_skills[i].parameters())

    success_rate = 0
    for _ in range(finetune_step):
        # fine-tune high-level net
        with torch.no_grad():
            cat_from_sub = get_sub_skill_cat_from_subskills(sub_skills, sub_skill_fast_weights, spt_state, spt_action)
        high_level_net.train()
        cat_from_high = high_level_net(spt_state, vars=high_fast_weights, bn_training=True)
        loss_finetune_high = torch.nn.CrossEntropyLoss()(cat_from_high, cat_from_sub)
        high_grad = torch.autograd.grad(loss_finetune_high, high_fast_weights)
        high_fast_weights = list(map(lambda p: p[1] - args.test_update_lr * p[0], zip(high_grad, high_fast_weights)))
        # fine-tune sub-skills
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
            if loss.item() == loss.item():  # remove nan, since some sub skill may not be trained in this epoch
                sub_skill_grad = torch.autograd.grad(loss, sub_skill_fast_weights[i])
                sub_skill_fast_weights[i] = list(map(lambda p: p[1] - args.test_update_lr * p[0], zip(sub_skill_grad, sub_skill_fast_weights[i])))

    # test
    high_level_net.eval()
    for i in range(len(sub_skills)):
        sub_skills[i].eval()

    env = envs[env_name]()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True

    # test for 20 times
    for tries in range(20): 
        obs = env.reset()

        rews = []
        for i in range(300):
            cat_from_high = high_level_net(torch.from_numpy(obs).cuda().float().unsqueeze(0), vars=high_fast_weights, bn_training=False)
            cat_from_high = torch.argmax(cat_from_high, dim=1)[0]
            act = sub_skills[cat_from_high](torch.from_numpy(obs).unsqueeze(0).cuda().float(), vars=sub_skill_fast_weights[cat_from_high], bn_training=False).detach().cpu().numpy()[0]
            obs, rew, done, info = env.step(act)
            rews.append(rew)

            if info['success'] == True:
                success_rate += 0.05
                print(env_name, 'k_shots=', k_shot, ' finetune_step=', finetune_step, ' tries:', tries, True)
                break
        if info['success'] == False:
            print(env_name, 'k_shots=', k_shot, ' finetune_step=', finetune_step, ' tries:', tries, False)

    env.close()

    return success_rate

if __name__ == '__main__':
    args = get_args()
    env_names, envs = get_all_envs()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False  # for maml
    np.random.seed(args.seed)
    random.seed(args.seed)

    k_shots = [1,3]
    fine_tune_steps = [10, 30, 50, 100, 300, 500]

    if args.finetune_step > 0:
        fine_tune_steps = [args.finetune_step]
    
    high_config = [
        ('linear', [512, 39]),
        ('bn', [512]),
        ('relu', [True]),
        ('linear', [512, 512]),
        ('bn', [512]),
        ('relu', [True]),
        ('linear', [512, 512]),
        ('bn', [512]),
        ('relu', [True]),
        ('linear', [args.test_skill_num, 512]),
    ]

    sub_skill_config = [
        ('linear', [512, 39]),
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

    high_level_net = Learner(high_config).cuda()
    sub_skills = [Learner(sub_skill_config).cuda() for _ in range(args.test_skill_num)]

    model_name = args.test_suite + '_' + str(args.seed) + '_' + str(args.test_skill_num)
    if args.continuous:
        model_name += '_continuous'
    if args.soft:
        model_name += '_soft'
    if args.dmil_high:
        model_name += '_dmil_high'
    if args.dmil_low:
        model_name += '_dmil_low'

    high_model_name = args.output_dir+'/high/'+model_name
    sub_skill_name = args.output_dir+'/sub_skill/'+model_name
    # load_model
    high_level_net = torch.load(high_model_name+'.pkl', map_location='cuda:0')
    for i in range(args.test_skill_num):
        sub_skills[i] = torch.load(sub_skill_name+str(i)+'.pkl', map_location='cuda:0')

    tau0 = np.load('./few_shots/'+args.test_suite+'_fewshots.npy')

    # get name
    if args.test_suite == 'ML10_train':
        env_list = [1,6,11,18,28,31,33,34,46,48]
    elif args.test_suite == 'ML10_test':
        env_list = [13, 19, 27, 45, 47]
    elif args.test_suite == 'ML45_train':
        env_list = [i for i in range(50)]
        for testing_env in [2, 3, 14, 16, 17]:
            env_list.remove(testing_env)
    elif args.test_suite == 'ML45_test':
        env_list = [2, 3, 14, 16, 17]

    # test
    all_success_rates = []
    for k_shot in k_shots:
        for env_index in range(len(env_list)):
            for fine_tune_step in fine_tune_steps:
                sr = test_env_simple(env_names[env_list[env_index]], tau0, env_index, high_level_net, sub_skills, k_shot, fine_tune_step)
                all_success_rates.append(sr)
                print('test_skill_num=', args.test_skill_num, ' suite=', args.test_suite, ' k_shot=', k_shot, ' env=', env_names[env_list[env_index]], ' finetune_step=', fine_tune_step, ' sr=', sr)

    all_success_rates = np.array(all_success_rates).reshape((len(k_shots),len(env_list), len(fine_tune_steps))) 
    np.save('skill_'+str(args.test_skill_num)+'_suite_'+args.test_suite+'_all_success_rates.npy', all_success_rates)
    print('----------------------------------------')
