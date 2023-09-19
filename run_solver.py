""" Calls sampling-based solver that uses agents' value functions to find the optimal contract.

The low-level implementation of the solver can be found inside environments/two_stage_train.py

"""



import copy
from ray.rllib.agents import ppo
from PIL import Image
from utils.ray_config_utils import get_config_and_env,get_neg_config,get_solver_config
from environments.env_utils import make_video_from_rgb_imgs
import numpy as np  
import time 
import torch 

def run_solver(params_dict,checkpoint_paths,logger):
    num_samples = params_dict.get('solver_samples',10)
    trainer_config, _ = get_config_and_env(params_dict)  # that way, don't need to pickle / store
    env_copy = get_solver_config(params_dict,trainer_config,checkpoint_paths)
    logger.set_stage(2)  
    for _, path in enumerate(checkpoint_paths):
        train_config = copy.deepcopy(trainer_config) 
        train_config['num_workers'] = 1
        train_config['evaluation_num_workers'] = 0
        train_config['num_gpus'] = 0

        if train_config.get('stop_cond') :
            del train_config['stop_cond']

        frozen_trainer = ppo.PPOTrainer(config=train_config, env=train_config['env'])
        frozen_trainer.load_checkpoint(path)
        all_rewards,all_contracts = [] , [] 
        for j in range(num_samples):
            env_obs = env_copy.reset()
            contract_param = env_copy.contract_param
            env_dones = {'__all__': False}
            if params_dict.get('joint'):
                active_agents = ['a0']
            else:
                active_agents = ['a' + str(i) for i in range(train_config['env_config']['num_agents'])]
            domain_steps = 0

            ep_rewards = 0 
            while (not env_dones['__all__']) and domain_steps < train_config['horizon']:
                act_dict = {}
                for key in active_agents:
                    if params_dict['shared_policy']:
                        act_dict[key] = frozen_trainer.compute_single_action(
                        env_obs[key], policy_id='policy')
                    else:
                        act_dict[key] = frozen_trainer.compute_single_action(
                            env_obs[key], policy_id=key)
                
                env_obs, r, env_dones, i = env_copy.step(act_dict)

                # clear inactive agents from rewards
                for key in env_dones:
                    if env_dones[key] and key in active_agents:
                        active_agents.remove(key)
                domain_steps += 1

                for key in env_obs:
                    ep_rewards += r[key]
            log_dict = {'ep_rewards':ep_rewards, 'contract_param':float(contract_param[0])} 
            logger.simple_log(log_dict)
            all_rewards.append(ep_rewards)
            all_contracts.append(contract_param) 
        logger.simple_log({'mean reward':np.mean(all_rewards),'mean contract':np.mean(all_contracts),
                            'std reward':np.std(all_rewards),'std contract':np.std(all_contracts)})
    time.sleep(300) # wait for logger to finish writing 
    print('Finished Stage 2')