import torch
import torch.nn.functional as F
import numpy as np
from arguments import get_args

args = get_args()

def get_sub_skill_cat_from_subskills(fast_sub_skills, sub_skill_fast_weights, states, actions, soft=False):
    ''' states: shape[batch_size*length, 39]
        actions: shape[batch_size*length, 4]

        forward each state-action pair to each sub-skill and then
        compute the minimum loss among them to get the sub skill category

        return: sub_skill_cat in shape[batch_size, length]
    '''
    num, _ = states.shape
    max_cat = len(fast_sub_skills)

    loss = torch.ones((max_cat, num)).cuda()   # shape: [K, batch_size]
    loss_func = torch.nn.MSELoss(reduce=False)
    for i in range(max_cat):
        fast_sub_skills[i].eval()
        pred = fast_sub_skills[i](states, vars=sub_skill_fast_weights[i], bn_training=True)
        loss[i] = torch.sum(loss_func(pred, actions), dim=1)    # shape:[batch_size*length,]
    
    if soft:
        # soft:
        weight = torch.sum(loss, dim=0).reshape(1, -1)
        weight = 1. / weight
        weight = weight.repeat(max_cat, 1)
        soft_cat_from_sub = loss.mul(weight).reshape(num, max_cat)
        soft_cat_from_sub = 1 - soft_cat_from_sub

        return soft_cat_from_sub

    else:
        cat = torch.argmin(loss, dim=0) # shape:[batch_size*length,]

        return cat

def SoftCrossEntropy(input, target,):
    log_input = -F.log_softmax(input, dim=1)
    loss = torch.sum(torch.mul(log_input, target)) / input.shape[0]
    return loss


def continuous_nomarlization(cat_from_high, test=False):
    # punish surplus switching points along one trajectory
    cat_from_high = torch.argmax(cat_from_high, dim=1)
    if test == False:
        cat_from_high = cat_from_high.reshape(args.batch_size, -1)
    else:
        cat_from_high = cat_from_high.reshape(1, -1)
    minus = cat_from_high[:, 1:] - cat_from_high[:, 0:-1]
    return args.continuous_coefficient * torch.sum(torch.ne(minus, 0)) / (cat_from_high.shape[0]*cat_from_high.shape[1])

def get_few_shots(env_names, dmil_data, file_name, multi_shots=1):
    tau0 = []
    for env_name in env_names:
        length = dmil_data[env_name].length
        print('getting ', env_name, '...', ' total length is: ', length)
        state, action = dmil_data[env_name].__getitem__(np.random.randint(length))
        tau0.append(np.concatenate((state, action), axis=1))
        if multi_shots > 1:
            for _ in range(multi_shots-1):
                state, action = dmil_data[env_name].__getitem__(np.random.randint(length))
                tau0.append(np.concatenate((state, action), axis=1))

    np.save(file_name, np.array(tau0))

