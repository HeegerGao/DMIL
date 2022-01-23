import os
import shutil
import numpy as np
from metaworld.policies import *
import metaworld.envs.mujoco.env_dict as _env_dict

def get_policy_names(env_names):
    policy_names = []
    for env_name in env_names:
        base = "Sawyer"
        res = env_name.split("-")
        for substr in res:
            base += substr.capitalize()
        policy_name = base + "Policy"
        if policy_name == "SawyerPegInsertSideV2Policy":
            policy_name = "SawyerPegInsertionSideV2Policy"
        policy_names.append(policy_name)
    
    return policy_names

def get_all_envs():
    envs = []
    env_names = []
    for env_name in _env_dict.MT50_V2:
        # print(env_name)
        env_names.append(env_name)
        envs.append(_env_dict.MT50_V2[env_name])
    return env_names, envs

env_names, envs = get_all_envs()
policy_names = get_policy_names(env_names)

def collect_demos():
    if not os.path.exists("./demos"):
        os.makedirs("./demos")

    for i in range(50):
        env_name = env_names[i]
        env = envs[i]()
        env._partially_observable = False
        env._freeze_rand_vec = False
        env._set_task_called = True
        
        policy_name = policy_names[i]
        policy = globals()[policy_name]()

        path = "./demos/"+str(env_name)
        if os.path.exists(path):
            shutil.rmtree(path)
            os.mkdir(path)
        else:
            os.mkdir(path)

        print('env '+env_name+' collecting start!')
        
        for trail in range(2000):
            print('collecting '+str(trail)+' th trail...')
            trail_path = path+'/'+str(trail)
            os.mkdir(trail_path)

            obss = []
            acts = []
            rews = []
            dones = []

            obs = env.reset()
            for i in range(501):
                obss.append(obs)
                
                act = policy.get_action(obs)
                obs, rew, done, info = env.step(act+0.1*np.random.randn(4,))
                
                acts.append(act)
                rews.append(rew)

                # print(info)
                if info['success'] == True:
                    dones.append(True)
                else:
                    dones.append(done)

                if done:
                    obss.append(obs)
                    break
                # env.render()

            obss = np.array(obss)        
            acts = np.array(acts)        
            rews = np.array(rews)        
            dones = np.array(dones)        

            np.save(trail_path+'/obs.npy', obss)
            np.save(trail_path+'/acts.npy', acts)
            np.save(trail_path+'/rews.npy', rews)
            np.save(trail_path+'/dones.npy', dones)

        print('env '+env_name+' collecting done!')

if __name__ == '__main__':
    collect_demos()