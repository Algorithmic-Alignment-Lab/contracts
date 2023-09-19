"""
This Python file defines functions for running an experiment using the Proximal Policy Optimization (PPO) algorithm.

Functions:
- `run_experiment(params_dict)`: Runs the experiment with the provided parameters.
- `ppo_learning(params_dict, config, wb_logger)`: Executes PPO training and returns checkpoint paths.

Parameters:
- `params_dict`: A dictionary containing experiment parameters.
- `config`: Configuration settings for the PPO algorithm.
- `wb_logger`: Custom logger for experiment logging.

"""

import json
import random 
import tensorflow as tf 

import ray
from utils.logger_utils import CustomLoggerCallback 
import numpy as np
from utils.env_creator_functions import *
from utils.ray_config_utils import get_config_and_env, get_neg_config

def run_experiment(params_dict) : 
    # Setting random number generator 
    seed = params_dict['seed'] 
    np.random.seed(seed) 
    random.seed(seed) 
    tf.random.set_seed(seed) 

    # Get config for subgame learning
    config, _ = get_config_and_env(params_dict) 
    # Setup logger for stage 1 
    wb_logger = CustomLoggerCallback(params_dict)
    
    if params_dict['second_stage_only']:
        checkpoint_paths = params_dict['second_stage_only']
    else :
        checkpoint_paths = ppo_learning(params_dict,config,wb_logger)
    
    # Store these paths somewhere, useful for loading somewhere else 
    with open(params_dict['store_path'], 'w') as fp:
        json.dump({"first_stage_paths": checkpoint_paths}, fp)

    neg_config = None 
    stage_2_weights = None
    if not params_dict['first_stage_only'] and params_dict['negotiate'] : 
        # Update the logger for the new stage
        wb_logger.set_stage(2)  
        # Get the config for training the negotiation game 
        neg_config = get_neg_config(params_dict,config,checkpoint_paths)
        # Train with the new config 
        stage_2_weights = ppo_learning(params_dict,neg_config,wb_logger)  

    return {'config_subgame': config,
            'config_negotiation':neg_config, 
            'exp_name': params_dict['exp_name'],
            'weight_directories': checkpoint_paths,
            'negotiation_weights': stage_2_weights,
            'logger': wb_logger}


def ppo_learning(params_dict,config,wb_logger) : 
    stop_condition = config['stop_cond'] 
    del config['stop_cond']
    
    analysis = ray.tune.tune.run('PPO',name=params_dict['exp_name'],stop=stop_condition,
                 config=config,callbacks=[wb_logger],local_dir=params_dict['results_dir'],
                 num_samples=1, verbose=0 ,checkpoint_freq=10, checkpoint_at_end=True)
    
    checkpoint_paths = []
    for trial in analysis.trials:
        checkpoint_paths.append(analysis.get_last_checkpoint(trial=trial)._local_path)
    return checkpoint_paths 
