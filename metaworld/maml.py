import numpy as np
import random
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data import DMILDatasetForOneTask, get_all_env_names
from arguments import get_args
from learner import Learner
from utils import *
from torch.utils.tensorboard import SummaryWriter

def maml_train(args, high_level_net, sub_skills, high_model_name, sub_skill_name):
    for epoch in range(args.epoch):
        print('epoch: ', epoch)
        high_losses_q = 0  
        sub_skill_losses_q = [0 for _ in range(len(sub_skills))] 

        for task_name in all_env_names:
            spt_state, spt_action = next(iter(dmil_loaders[task_name]))  # shape: [batch_size, length, size]
            spt_state, spt_action = spt_state.cuda().float().reshape(-1, 39), spt_action.cuda().float().reshape(-1, 4)
            qr_state, qr_action = next(iter(dmil_loaders[task_name]))  # shape: [batch_size, length, size]
            qr_state, qr_action = qr_state.cuda().float().reshape(-1, 39), qr_action.cuda().float().reshape(-1, 4)

            high_fast_weights = high_level_net.parameters()
            sub_skill_fast_weights = []
            for i in range(len(sub_skills)):
                sub_skill_fast_weights.append(sub_skills[i].parameters())

            # finetune K steps
            if epoch >= args.pretrain_epoch:
                for _ in range(args.fast_interations):
                    if not args.dmil_high:
                        # fine-tune high-level net
                        with torch.no_grad():
                            cat_from_sub = get_sub_skill_cat_from_subskills(sub_skills, sub_skill_fast_weights, spt_state, spt_action, soft=args.soft)
                        high_level_net.train()
                        cat_from_high = high_level_net(spt_state, vars=high_fast_weights, bn_training=True)
                        if args.soft:
                            loss_finetune_high = SoftCrossEntropy(cat_from_high, cat_from_sub)
                        else:
                            loss_finetune_high = torch.nn.CrossEntropyLoss()(cat_from_high, cat_from_sub)

                        # during finetune we don't need to use continuous variant. We only need it during outer loops
                        high_grad = torch.autograd.grad(loss_finetune_high, high_fast_weights, create_graph=True)
                        high_fast_weights = list(map(lambda p: p[1] - args.update_lr * p[0], zip(high_grad, high_fast_weights)))

                    if not args.dmil_low:
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
                                sub_skill_grad = torch.autograd.grad(loss, sub_skill_fast_weights[i], create_graph=True)
                                sub_skill_fast_weights[i] = list(map(lambda p: p[1] - args.update_lr * p[0], zip(sub_skill_grad, sub_skill_fast_weights[i])))

            # compute and add meta loss on query set
            with torch.no_grad():
                cat_from_sub = get_sub_skill_cat_from_subskills(sub_skills, sub_skill_fast_weights, qr_state, qr_action)
            high_level_net.train()
            cat_from_high = high_level_net(qr_state, vars=high_fast_weights, bn_training=True)
            loss_finetune_high = torch.nn.CrossEntropyLoss()(cat_from_high, cat_from_sub)
            if args.continuous:
                loss_finetune_high += continuous_nomarlization(cat_from_high)
            high_losses_q += loss_finetune_high

            # fine-tune sub-skills
            with torch.no_grad():
                high_level_net.eval()
                cat_from_high = high_level_net(qr_state, vars=high_fast_weights, bn_training=True)
                cat_from_high = torch.argmax(cat_from_high, dim=1).reshape(qr_action.shape[0], 1)

            for i in range(len(sub_skills)):
                sub_skills[i].train()
                pred = sub_skills[i](qr_state, vars=sub_skill_fast_weights[i], bn_training=True)
                # get mask of sub_skill_i
                mask = cat_from_high.eq(i)
                mask = mask.repeat(1, pred.shape[-1])
                masked_output = pred.mul(mask)
                masked_action = qr_action.mul(mask)
                loss = torch.nn.MSELoss()(masked_output, masked_action)
                if loss.item() == loss.item():  # remove nan, since some sub skill may not be trained in this epoch
                    sub_skill_losses_q[i] += loss


        # end of all tasks, meta updating
        # in this part we do not use continuous or soft variant, since this part means the few-shot phase
        # continuous and soft variants should work only during meta-training
        print('meta updatig...')
        if high_losses_q > 0:
            print('high loss:', high_losses_q.item())
            tb_writer.add_scalar('high_loss', high_losses_q.item(), global_step=epoch)
            high_level_meta_opt.zero_grad()
            high_losses_q.backward()
            torch.nn.utils.clip_grad_norm_(high_level_net.parameters(), max_norm=10, norm_type=2)
            high_level_meta_opt.step()
        for i in range(args.sub_skill_cat):
            if sub_skill_losses_q[i] > 0:
                print('subskill'+str(i)+' loss:', sub_skill_losses_q[i].item())
                tb_writer.add_scalar('sub_skill'+str(i)+'_loss', sub_skill_losses_q[i].item(), global_step=epoch)
                sub_skill_meta_opts[i].zero_grad()
                sub_skill_losses_q[i].backward()
                torch.nn.utils.clip_grad_norm_(sub_skills[i].parameters(), max_norm=10, norm_type=2)
                sub_skill_meta_opts[i].step()

        torch.save(high_level_net, high_model_name + '.pkl')
        for i in range(len(sub_skills)):
            torch.save(sub_skills[i], sub_skill_name+str(i)+'.pkl')


if __name__ == "__main__":
    args = get_args()

    # setting parameters
    torch.cuda.set_device(args.gpu_index)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False  # for maml
    np.random.seed(args.seed)
    random.seed(args.seed)

    model_name = args.suite + '_' + str(args.seed) + '_' + str(args.sub_skill_cat)
    if args.continuous:
        model_name += '_continuous'
    if args.soft:
        model_name += '_soft'
    if args.dmil_high:
        model_name += '_dmil_high'
    if args.dmil_low:
        model_name += '_dmil_low'

    tb_folder_name = os.path.join(args.tb_dir, model_name)
    high_model_name = args.output_dir+'/high/'+model_name
    sub_skill_name = args.output_dir+'/sub_skill/'+model_name

    tb_writer = SummaryWriter(log_dir=tb_folder_name)

    if not args.output_dir:
        print('No experiment result will be saved.')
        raise

    os.makedirs(os.path.join(args.output_dir, 'high'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'sub_skill'), exist_ok=True)
    
    # get dataset
    if args.suite == 'ML10':
        all_env_names = [get_all_env_names()[i] for i in [1, 6, 11, 18, 28, 31, 33, 34, 46, 48]]
    elif args.suite == 'ML45':
        env_list = [i for i in range(50)]
        for testing_env in [2,3,14,16,17]:
            env_list.remove(testing_env)
        all_env_names = [get_all_env_names()[i] for i in env_list]

    print('getting training data...')
    dmil_datas = {}
    dmil_loaders = {}
    for task_name in all_env_names:
        print('getting ', task_name, '...')
        dmil_datas[task_name] = DMILDatasetForOneTask(task_name)
        dmil_loaders[task_name] = DataLoader(dmil_datas[task_name], batch_size=args.batch_size, shuffle=True)

    # model
    high_config = [
        ('linear', [512, 39]),
        ('relu', [True]),
        ('bn', [512]),
        ('linear', [512, 512]),
        ('relu', [True]),
        ('bn', [512]),
        ('linear', [512, 512]),
        ('relu', [True]),
        ('bn', [512]),
        ('linear', [args.sub_skill_cat, 512]),
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
    sub_skills = [Learner(sub_skill_config).cuda() for _ in range(args.sub_skill_cat)]

    if args.load_model:
        high_level_net = torch.load(high_model_name+'.pkl', map_location='cuda:'+str(args.gpu_index))
        for i in range(len(sub_skills)):
            sub_skills[i] = torch.load(sub_skill_name+str(i)+'.pkl', map_location='cuda:'+str(args.gpu_index))

    # opts
    high_level_meta_opt = optim.Adam(high_level_net.parameters(), lr=1e-4)
    sub_skill_meta_opts = [optim.Adam(sub_skills[i].parameters(), lr=1e-4) for i in range(len(sub_skills))]

    # train
    maml_train(args, high_level_net, sub_skills, high_model_name, sub_skill_name)

