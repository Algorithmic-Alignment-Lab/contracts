"""
This Python file contains functions to process parameters to generate configurations for running experiments in multi-agent environments. It leverages the Ray RLlib library for reinforcement learning experiments.

Functions:
- `parse_arguments_dict(experiment_name, arg_dict)`: Parses experiment arguments and returns a dictionary of experiment parameters.
- `get_config_and_env(params_dict)`: Retrieves the experiment configuration and environment settings based on the specified parameters.
- `get_neg_config(params_dict, config, checkpoint_paths)`: Generates configuration settings for the negotiation stage of the experiment.
- `get_solver_config(params_dict, config, checkpoint_paths)`: Generates configuration settings for the negotiation solver.

"""
import ray
from ray.rllib.evaluation.metrics import collect_metrics
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

import json
import copy

import contract.contract_list
from environments.two_stage_train import SeparateContractNegotiateStage, SeparateContractSubgameStage, SeparateContractCombinedStage, JointEnv, NegotiationSolver
import numpy as np
from utils.env_creator_functions import env_creator, get_base_env_tag
from utils.logger_utils import MetricsCallback 
from ray.rllib.algorithms.callbacks import MultiCallbacks
from ray.rllib.models import ModelCatalog 
from environments.Networks.vision_net import VisionNetwork

def parse_arguments_dict(experiment_name, arg_dict):
    # construct the input tuple to run_experiment
    contract_params = arg_dict.get("contract_params",{})
    contract_params['num_agents'] = arg_dict.get('num_agents') 
    contract_selection = getattr(contract.contract_list, arg_dict.get('contract'))(**contract_params)

    exp_name = experiment_name

    results_dir = arg_dict.get('results_dir','ray_results/')  
    wandb = arg_dict.get('wandb',False)
 
    model = {"fcnet_hiddens": [64, 64]}
    model_p =  arg_dict.get('model_params',{}) 
    model.update(model_p)
    
    base_env_tag = get_base_env_tag(arg_dict) 
    if model.get('custom_model') :
        model = arg_dict.get('model_params')
        custom_model = model['custom_model']
        if custom_model =='v1'  :
            ModelCatalog.register_custom_model("v1", VisionNetwork)
            print('Registering vision network') 
        else :
            raise NotImplementedError

    seed = arg_dict.get('seed',0) 

    evaluate= True 
    arg_dict['evaluation_interval'] = 1 
    if not arg_dict.get('evaluation') :
        arg_dict['eval_num_workers'] = 0 
        arg_dict['evaluation_interval'] = None 
        evaluate = False 
    
    #env_params if a dictionary needs to be passed in (this is mainly for MAPF environment)
    env_params = arg_dict.get('env_params',{})
    # env_args is a dictionary whose arguments are passed to the env (for eg, {image_obs=True})
    env_args = arg_dict.get('env_args',{})
    env_config =  {
                "num_agents": arg_dict.get("num_agents"),
                "env_params": env_params,
            } 
    env_config.update(env_args)

    base_env = env_creator(base_env_tag,env_config)

    convolutional = False
    if arg_dict.get('env_args'):
        if arg_dict['env_args'].get('image_obs'):
            convolutional = True

    gpus = arg_dict.get("num_gpus") 
    if gpus is None :
        gpus = 0 

    params_dict = {
        'contract': contract_selection,   # Which contract fn to use
        'base_env': base_env,             # Base environment (should be created above)
        'base_env_tag': base_env_tag,     # Base environment tag (should be created above)
        'exp_name': exp_name,             # Experiment name
        'num_agents': arg_dict.get("num_agents"), # Number of agents 
        'horizon': arg_dict.get("horizon"),       # Horizon (Episode length)
        'batch_size': arg_dict.get("batch_size"), # Batch size (Number of samples to collect between training iterations)
        'num_workers': arg_dict.get("num_workers"), # Number of workers (Number of parallel environments) 
        'eval_num_workers': arg_dict.get("eval_num_workers"), # Number of evaluation workers (Should only be used if there is an evaluation function to use)
        'num_gpus': gpus,       # Number of GPUs to use
        'joint': arg_dict.get("joint",False),                             # Whether to run the joint baseline   
        'separate': arg_dict.get("separate"),       # Whether to run the vanilla independent baseline with no contracting
        'combined': arg_dict.get("combined"),       # Whether to implement both contract negotiation and subgame stage together (very unstable in practice)
        'first_stage_only': arg_dict.get("first_stage_only"),   # Whether to only learn the subgame stage
        'second_stage_only': arg_dict.get("second_stage_only"), # Whether to only learn the contract negotiation stage 
        'store_renders': arg_dict.get("store_renders"),         # Whether to store renders of the environment
        'convolutional': convolutional,                         # Whether to use a convolutional network
        'shared_policy': arg_dict.get("shared_policy"),         # Whether to use a shared policy for the agents 
        'wandb' : wandb,                                        # Whether to use wandb
        'results_dir' : results_dir,                            # Directory to store results
        'seed' : seed ,                                         # Seed for the experiment        
        'parent_tag': arg_dict['parent_name'],                  # Parent tag for wandb    
        'evaluation_interval': arg_dict['evaluation_interval'], # Evaluation interval 
        'evaluation' : evaluate,                                # Whether to evaluate or not (should be used above)
        'negotiate' : True,                                     # Whether second stage will be implemented (modified again below)
        'num_timesteps': arg_dict.get("num_timesteps"),         # Number of timesteps to run the experiment for
        'negotiation':arg_dict.get("negotiation"),              # Negotiation arguments should be provided in this dictionary
        'store_path':arg_dict.get("store_path"),                # Path to store the model (experiment checkpoints)
        'env_params':env_params,                                # Any custom env params 
        'model_params':model,                                   # Model parameters (should be a dictionary, mainly for custom models)
        'env_args':env_args,                                    # Any custom env params 
        'solver':arg_dict.get("solver",False),                  # Whether to use a solver for second stage of contracting
        'minibatch_size':arg_dict.get("minibatch_size",4096),   # Minibatch size for the contract negotiation stage
        'solver_samples': arg_dict.get('solver_samples',10)     # Number of contracts sampled by solver per episode                                                                                                                             # Arguments to be passed to the environment (these arguments are passed                                                                 #    directly to the environment (not in a dictionary) - for general use)         
    }

    if params_dict['joint'] or params_dict['separate']  or params_dict['combined'] or params_dict['solver']:
        params_dict['negotiate'] = False 

    return params_dict


def get_config_and_env(params_dict):
    env_copy = None
    common_config = {
            "framework": "torch",
            "simple_optimizer": True,
            "train_batch_size": params_dict['batch_size'],
            "seed": params_dict['seed'],
            "evaluation_interval": params_dict['evaluation_interval'],
            "evaluation_num_workers": params_dict['eval_num_workers'],
            'create_env_on_driver': True,
            "evaluation_config": {
                'render_env': params_dict['store_renders'],
            },
            "model":params_dict['model_params'],
            "num_workers": params_dict['num_workers'],
            'sgd_minibatch_size': params_dict['minibatch_size'],
            "horizon": params_dict['horizon'],
            "gamma": 0.99,
            "multiagent": {
                "policies_to_train": ['policy'] if params_dict['shared_policy'] else ['a' + str(i) for i in range(params_dict['num_agents'])],
                "policies": {'policy': PolicySpec()} if params_dict['shared_policy'] else {'a'+str(i): PolicySpec() for i in range(params_dict['num_agents'])},
                "policy_mapping_fn": (lambda agent_id, episode, worker, **kwargs: 'policy')
                    if params_dict['shared_policy'] else (lambda agent_id, episode, worker, **kwargs: agent_id)
            },

            'callbacks': MultiCallbacks([lambda : MetricsCallback()]),
            'num_gpus': params_dict['num_gpus'],
            'stop_cond': {'timesteps_total': params_dict.get("num_timesteps")}
        }
    env_config =  {
                "num_agents": params_dict['num_agents'],
                "env_params":params_dict['env_params']
            } 
    env_config.update(params_dict['env_args']) 
    if params_dict['joint']:
        env_config.update({
                "base_env": params_dict['base_env']
            })
        config_subgame = {
            "env": "JointEnv",
            "env_config": env_config ,
            "multiagent": {
                "policies_to_train": ['a0'],
                "policies": {'a0': PolicySpec()},
                "policy_mapping_fn": lambda agent_id, episode, worker, **kwargs: agent_id
            }
        }
        config_subgame.update(common_config)
        env_copy = JointEnv(**(config_subgame['env_config']))

    elif params_dict['separate']:
        config_subgame = {
            "env": params_dict['base_env_tag'],
            "env_config": env_config
        }
        config_subgame.update(common_config)
        env_copy = params_dict['base_env']

    elif params_dict['combined']:
        env_config.update({
                "base_env": params_dict['base_env'],
                'contract': params_dict['contract'],
                'convolutional': params_dict['convolutional']
            })
        config_subgame = {
            "env": "ContractWrapperCombined",
            "env_config": env_config
        }
        config_subgame.update(common_config)
        env_copy = SeparateContractCombinedStage(**(config_subgame['env_config']))

    else:
        env_config.update({
                "base_env": params_dict['base_env'],
                'contract': params_dict['contract'],
                'convolutional': params_dict['convolutional']
            })
        config_subgame = {
            "env": "ContractWrapperSubgame",
            "env_config": env_config
        }

        config_subgame.update(common_config)
        env_copy = SeparateContractSubgameStage(**(config_subgame['env_config']))

    if params_dict['store_renders']:
        config_subgame['evaluation_config'] = {'render_env': True}

    return config_subgame, env_copy


def get_neg_config(params_dict,config,checkpoint_paths) : 
    config_negotiate = copy.deepcopy(config)
    config_frozen = copy.deepcopy(config)

    config_frozen['num_workers'] = 0
    config_frozen['evaluation_num_workers'] = 0
    config_frozen['num_gpus'] = 0

    if params_dict.get("negotiation") : 
        neg_args = params_dict['negotiation'] 
    else :
        neg_args = {} 

    if neg_args.get('sgd_minibatch_size') :
        config_negotiate['sgd_minibatch_size'] =  neg_args['sgd_minibatch_size']
    else :   
        config_negotiate['sgd_minibatch_size'] = 128 

    if neg_args.get('batch_size') :
        config_negotiate['train_batch_size'] =  neg_args['batch_size']
    else :   
        config_negotiate['train_batch_size'] = 256 

    if neg_args.get('num_timesteps') :  
        config_negotiate['stop_cond'] = {'timesteps_total': neg_args['num_timesteps'] }
    else :
        config_negotiate['stop_cond'] = {'timesteps_total': 1000}

    config_negotiate['evaluation_interval'] = 1 

    if not neg_args.get('evaluation') :
        config_negotiate['evaluation_num_workers'] = 0 
        config_negotiate['evaluation_interval'] = None 
    
    config_negotiate['horizon'] = 2
    config_negotiate['env_config']['horizon'] = params_dict['horizon']
    config_negotiate['env_config']['trainer_config'] = config_frozen
    config_negotiate['env_config']['trainer_env'] = 'ContractWrapperSubgame'
    config_negotiate['env_config']['trainer_path'] = checkpoint_paths[0] 
    
    config_negotiate['env_config']['shared'] = params_dict['shared_policy']
    config_negotiate['env'] = 'ContractWrapperNegotiate'
        
    return config_negotiate

def get_solver_config(params_dict,config,checkpoint_paths) : 
    config_negotiate = copy.deepcopy(config)
    config_frozen = copy.deepcopy(config)

    config_frozen['num_workers'] = 0
    config_frozen['evaluation_num_workers'] = 0
    config_frozen['num_gpus'] = 0
    config_negotiate['env_config']['horizon'] = params_dict['horizon']
    config_negotiate['env_config']['trainer_config'] = config_frozen
    config_negotiate['env_config']['trainer_env'] = 'ContractWrapperSubgame'
    config_negotiate['env_config']['trainer_path'] = checkpoint_paths[0] 
    
    config_negotiate['env_config']['shared'] = params_dict['shared_policy']
    env = NegotiationSolver(**config_negotiate['env_config'])
    return env 