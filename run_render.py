""" Contains rendering function for Cleanup and Harvest environments.

The rendering function takes as input the parameters of the config file, the stored path to load agents from and the number of renders required. 

"""
import copy
from ray.rllib.agents import ppo
from PIL import Image
from utils.ray_config_utils import get_config_and_env
from environments.env_utils import make_video_from_rgb_imgs
import torch 

def run_rendering(params_dict, store_path,num_renders=10):
    trainer_config, env_copy = get_config_and_env(params_dict)  # that way, don't need to pickle / store
    checkpoint_paths = params_dict['second_stage_only'] 
    for _, path in enumerate(checkpoint_paths):
        config_render = copy.deepcopy(trainer_config) 
        config_render['num_workers'] = 1
        config_render['evaluation_num_workers'] = 0
        config_render['num_gpus'] = 0

        if config_render.get('stop_cond') :
            del config_render['stop_cond']

        frozen_trainer = ppo.PPOTrainer(config=config_render, env=config_render['env'])
        frozen_trainer.load_checkpoint(path)
        for j in range(num_renders):
            env_obs = env_copy.reset()
            env_dones = {'__all__': False}
            if params_dict.get('joint'):
                active_agents = ['a0']
            else:
                active_agents = ['a' + str(i) for i in range(config_render['env_config']['num_agents'])]
            domain_steps = 0

            imgs = []  # array for GIF build up through the render
            ep_rewards = 0 
            while (not env_dones['__all__']) and domain_steps < config_render['horizon']:
                # add render logic
                try:
                    img = env_copy.full_map_to_colors() 
                except :
                    img = env_copy.base_env.full_map_to_colors()
                imgs.append(img)
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

            # save render GIF todo
            height, width, _ = imgs[0].shape

            # Upscale to be more legible
            width *= 20
            height *= 20
            imgs = [image.astype('uint8') for image in imgs]
            if 'contract_param' in i['a0'] :
                cp = i['a0']['contract_param']
                make_video_from_rgb_imgs(imgs,store_path,video_name='trajectory_{}_r{}_c{}'.format(j,ep_rewards,cp) ,resize=(width,height))
            else :
                make_video_from_rgb_imgs(imgs,store_path,video_name='trajectory_{}_r{}'.format(j,ep_rewards) ,resize=(width,height))
    print('Done rendering')