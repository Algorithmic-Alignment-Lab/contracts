import numpy as np
from contract.contract import Contract
import gym
import copy 

# charge for so many cleaned squares
class CleanupContract(Contract):
    """
    A contract space for a multi-agent Cleanup environment.

    Contracts are parameterized by theta in [0, 0.2], which correspond to a payment per waste cell cleaned, paid for evenly by the other agents. 

    Parameters:
        num_agents (int): The number of agents in the environment.
        low_val (float, optional): The lower bound for the transfer values. Default is 0.
        high_val (float, optional): The upper bound for the transfer values. Default is 0.2.

    """ 
    def __init__(self, num_agents,low_val=0,high_val=0.2):
        super().__init__(gym.spaces.Box(shape=(1,), low=low_val, high=high_val), np.array([0.0]), num_agents)

    def compute_transfer(self, obs, acts,rews, params, infos=None):
        keys = list(acts.keys())
        transfers = {}
        for i in range(len(keys)):
            transfers[keys[i]] = - params[keys[i]][0] * infos[keys[i]]['cleaned_squares']
        return transfers
    
class HarvestFeaturemodLocalContract(Contract):
    """
    A contract space for a multi-agent Harvest environment.

    Contracts are parameterized by theta in [0, 10]. When an agent eats an apple in a low-density region, defined as an apple having less than 4 neighboring apples within a radius of 5, they transfer theta to the other agents, which is equally distributed to the other agents.


    Parameters:
        num_agents (int): The number of agents in the environment.
        low_val (float, optional): The lower bound for the transfer values. Default is 0.
        high_val (float, optional): The upper bound for the transfer values. Default is 10.

    """
    def __init__(self, num_agents,low_val=0,high_val=10.0):
        super().__init__(gym.spaces.Box(shape=(1,), low=low_val, high=high_val), np.array([0.0]), num_agents)

    def compute_transfer(self, obs, acts,rews, params, infos=None):
        keys = list(acts.keys())
        transfers = {}
        for i in range(len(keys)):
            # if low apples locally and ate an apple
            if infos[keys[i]]['feature_obs'][8] < 4 and infos[keys[i]]['eaten_close_apples'] > 0:
                transfers[keys[i]] = params[keys[i]][0]  # strong negative consumption penalty
            else:
                transfers[keys[i]] = 0
        return transfers 
        
class SelfdriveContractDistprop(Contract):
    """
    A contract space for a multi-agent self-driving car environment with distance-based reward transfers.

    The ambulance can propose a per-unit subsidy of theta in [0, 100] to the cars at the time of ambulance crossing. Each car is transferred theta times its distance behind the ambulance at time of merge by the ambulance. If a car is ahead of the ambulance at time of reward, it pays the ambulance theta times its distance ahead of the ambulance.

    Parameters:
        num_agents (int): The number of agents in the environment.

    """
    def __init__(self, num_agents):
        super().__init__(gym.spaces.Box(shape=(1,), low=0, high=100.0), np.array([0.0]), num_agents)

    def compute_transfer(self, obs, acts,rews, params, infos=None):
        transfers = {}
        transfers['a0'] = 0
        if 'a0' in acts.keys():
            if infos['a0']['just_passed']:
                id_behind = []
                for i in range(1, len(list(obs.values())[0]) // 2):
                    # if behind the ambulance, add name to list
                    if obs['a0'][2 + i] < 0:
                        id_behind.append('a' + str(i))
                # if ANY cars are behind
                if id_behind:
                    # compute the distance
                    dists = {}
                    sum_dists = 0
                    for id in id_behind:
                        id_dist = -obs['a0'][2 + int(id[-1])]
                        dists[id] = id_dist
                        sum_dists += id_dist

                    transfers['a0'] = (params['a0'][0] * sum_dists, {key: dists[key] / sum_dists for key in id_behind})
                else:
                    transfers['a0'] = 0

                # transfer back to the ambulance past
                for i in range(1, len(list(obs.values())[0]) // 2):
                    # if behind the ambulance, add name to list
                    if 'a' + str(i) in acts.keys() and 'a' + str(i) not in id_behind:
                        transfers['a' + str(i)] = (params['a0'][0] * (obs['a0'][2 + i]), {'a0': 1})

        for i in range(1, len(list(obs.values())[0]) // 2):
            if 'a' + str(i) not in transfers.keys():
                transfers['a' + str(i)] = 0
        return transfers

