import copy

from ray.rllib.env import MultiAgentEnv
from ray.rllib.agents import ppo
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.env.apis.task_settable_env import TaskSettableEnv
import gym
import random
import scipy 
import numpy as np
from copy import deepcopy 
# from utils.contracts import *

tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()

class SeparateContractEnv(MultiAgentEnv):
    """
    A wrapper on a multi-agent environment which modifies rewards based on contracting/

    Parameters:
        - base_env (gym.Env): The base environment on which contracts are applied.
        - contract: The contract object specifying the terms of the contract (contract logic found in contract/contract_list.py).
        - num_agents (int): The number of agents interacting in the environment.
        - convolutional (bool): Whether to use convolutional observations or not.
        - env_params (dict, optional): Additional environment parameters.
        - null_prob (float, optional): The probability of a null contract (no contract) during interactions.
        - **kwargs: Additional keyword arguments.
    """
    metadata = {
        "render.modes": ["rgb_array"],
    }

    def __init__(self, base_env, contract, num_agents, convolutional,env_params=None,
                 null_prob=0.0,**kwargs):
        self.num_agents = num_agents
        self.base_env = base_env
        self.contract = contract
        self.contract_low = self.contract.contract_space.low
        self.contract_high = self.contract.contract_space.high
        self.contract_state = {'a'+str(i): 0 for i in range(self.num_agents)}
        self.convolutional = convolutional
        self.null_prob         = null_prob

        # set common observation space
        if self.convolutional:
            # make appended observations additional channels, as opposed to appended features
            contract_space =  gym.spaces.Box(low=np.concatenate((self.contract_low, np.array([0.0]))),
                                                high=np.concatenate((self.contract_high, np.array([3.0]))))
            obs_space = self.base_env.observation_space
            space_dict={'contract': contract_space}
            if 'features' in obs_space.keys():
                space_dict['features'] = obs_space['features']
            if 'image' in obs_space.keys():
                space_dict['image'] = obs_space['image']
            self.observation_space = gym.spaces.Dict(space_dict)

        else:
            self.observation_space = gym.spaces.Box(low=np.concatenate((self.base_env.observation_space.low, self.contract_low, np.array([0.0]))),
                                                high=np.concatenate((self.base_env.observation_space.high, self.contract_high, np.array([3.0]))))

    def step(self, acts):
        # always act in the last section of the contract
        raw_obs, base_rew, dones, infos = self.base_env.step(acts)
        self.obs = {key: raw_obs[key] for key in acts.keys()}

        transfers = self.contract.compute_transfer(self.obs, acts,base_rew, self.params, infos)
    
        rews = {key: base_rew[key] for key in acts.keys()}

        total_transfers = 0 
        for i in range(self.num_agents):
            if 'a' + str(i) in acts.keys():
                # check proportions, default is even-division with integer transfer
                if type(transfers['a' + str(i)]) is tuple:
                    # of format (transfer_val, {id: proportion})
                    rews['a' + str(i)] -= transfers['a' + str(i)][0]
                    total_transfers += transfers['a' + str(i)][0] 
                    for j in range(self.num_agents):
                        if 'a' + str(j) in transfers['a' + str(i)][1].keys() and 'a'+str(j) in rews.keys():
                            # pay the recipient agent the proportionate amount of the tranfer, weighted by the proportion dict
                            rews['a' + str(j)] += transfers['a' + str(i)][0] * transfers['a' + str(i)][1]['a' + str(j)]
                else:
                    # assumed to be constant, be it integer, double, et
                    rews['a' + str(i)] -= transfers['a' + str(i)]
                    total_transfers += transfers['a' + str(i)] 
                    for j in range(self.num_agents):
                        if i != j and 'a' + str(j) in acts.keys():
                            # even distribution by default
                            rews['a' + str(j)] += transfers['a' + str(i)] / (len(acts.keys()) - 1)

        self.base_env.metrics['transfers'] += total_transfers 

        for k,v in rews.items():
            self.transferred_reward_dict[k].append(v) 

        if hasattr(self.base_env, 'compute_sustainability') and dones['__all__']:
            self.base_env.metrics['transfer_sustainability'] = self.base_env.compute_sustainability(deepcopy(self.transferred_reward_dict))
            self.base_env.metrics['transfer_equality'] = self.base_env.compute_equality(deepcopy(self.transferred_reward_dict))

        for k in acts.keys():
            infos[k]['contract_param']= self.params[k] 
            
        if self.convolutional:
            obs = {} 
            for i in range(self.num_agents):
                key = 'a'+str(i) 
                obs[key] = self.obs[key] 
                contract_obs =  np.concatenate((self.params[key], np.array([0])))
                obs[key].update({'contract':contract_obs})
            return (obs,rews,dones,infos)   
        else:
            return (
                {
                    key: np.concatenate((self.obs[key], self.params[key], np.array([0])))
                    for key in acts.keys()
                },
                rews,
                dones,
                infos
            )

    def render(self, mode='rgb'):
        return self.base_env.render()

    def reset(self):
        raise NotImplementedError
    
class SeparateContractSubgameStage(SeparateContractEnv):
    """
    The subgame stage of contracting.

    This class inherits from SeparateContractEnv and represents the subgame stage of contracting. 
    The reset functions samples a new random contract from the contract space.
    Other methods are implemented in the parent class. 

    Parameters:
        - base_env (gym.Env): The base environment on which contracts are applied.
        - contract: The contract object specifying the terms of the contract (contract logic found in contract/contract_list.py).
        - num_agents (int): The number of agents interacting in the environment.
        - convolutional (bool): Whether to use convolutional observations or not.
        - env_params (dict, optional): Additional environment parameters.
        - null_prob (float, optional): The probability of a null contract (no contract) during interactions.
        - **kwargs: Additional keyword arguments.

    Methods:
        - reset(): Reset the subgame stage.

    Note:
        This class extends SeparateContractEnv.
    """
    def __init__(self, base_env, contract,  num_agents, convolutional,env_params=None,
                 null_prob=0.0,**kwargs):
        super().__init__(base_env, contract, num_agents, convolutional,env_params,null_prob)

        # set action space
        self.action_space = self.base_env.action_space

    def reset(self):
        base_obs = self.base_env.reset()
        self.obs = copy.deepcopy(base_obs)
        # consistent update across all 
        if np.random.rand()>self.null_prob:
            rand_val = np.random.uniform(low=self.contract_low, high=self.contract_high)
        else :
            rand_val = self.contract_low
        self.contract_state = {'a'+str(i): 0 for i in range(self.num_agents)}
        self.params = {key: rand_val for key in ['a'+str(i) for i in range(self.num_agents)]}
        
        self.transferred_reward_dict = {} 
        for key in ['a'+str(i) for i in range(self.num_agents)]:
            self.transferred_reward_dict[key] = []

        if self.convolutional:
            obs = {} 
            for i in range(self.num_agents):
                key = 'a'+str(i) 
                obs[key] = self.obs[key] 
                contract_obs =  np.concatenate((self.params[key], np.array([0])))
                obs[key].update({'contract':contract_obs})
            return obs  

        else:
            return {
                'a'+str(i): np.concatenate((self.obs['a'+str(i)], self.params['a'+str(i)], np.array([0])))
                for i in range(self.num_agents)
            }


class SeparateContractNegotiateStage(SeparateContractEnv):
    """
    The negotiation stage class for negotiating contracts (stage 2).

    This class is the negotiation stage where agents negotiate and agree on contracts.
    It allows agents to propose and agree or reject contract terms.
    Agent's frozen policies from the subgame stage are used here.

    Parameters:
        - base_env (gym.Env): The base environment on which contracts are applied.
        - contract: The contract object specifying the terms of the contract.
        - num_agents (int): The number of agents interacting in the subgame stage.
        - horizon (int): Episode length.
        - trainer_config: Configuration for the trainer used during negotiation.
        - trainer_env: The environment used by the trainer during negotiation.
        - trainer_path: The path to the pre-trained trainer model. This is used to load the frozen agent policies
        from the subgame stage.
        - convolutional (bool): Whether to use convolutional observations or not.
        - shared (bool): Whether agents share the same policy during negotiation.
        - env_params (dict, optional): Additional environment parameters.
        - **kwargs: Additional keyword arguments.
    """
    def __init__(self, base_env, contract, num_agents, horizon, trainer_config, trainer_env, trainer_path, convolutional, shared,env_params=None, **kwargs):
        super().__init__(base_env, contract, num_agents, convolutional)

        self.horizon = horizon

        self.convolutional = convolutional

        # upload frozen policy from last step
        self.frozen_trainer = ppo.PPOTrainer(config=trainer_config, env=trainer_env)
        self.frozen_trainer.load_checkpoint(trainer_path)
        self.shared = shared

        # set action space (only to contract proposal, and accept-reject for other agents)
        self.action_space = gym.spaces.Box(low=np.concatenate((self.contract_low, np.array([0.0]))),
                                                high=np.concatenate((self.contract_high, np.array([1.0]))))

        self.metrics = {'contract':-1,'accepted':0} 
        

    def reset(self):
        self.metrics = {'contract':-1,'accepted':0}  
        base_obs = self.base_env.reset()
        self.obs = copy.deepcopy(base_obs)
        self.last_seen_obs = copy.deepcopy(base_obs)
        self.params = None
        self.contract_state = {'a' + str(i): 2 for i in range(self.num_agents)}

        self.transferred_reward_dict = {} 
        for key in ['a'+str(i) for i in range(self.num_agents)]:
            self.transferred_reward_dict[key] = []

        if self.convolutional:
            obs = {} 
            for i in range(self.num_agents):
                key = 'a'+str(i) 
                obs[key] = self.obs[key] 
                contract_obs =  np.concatenate((np.zeros(self.contract_low.shape), np.array([self.contract_state[key]])))
                obs[key].update({'contract':contract_obs})
            return obs  
        else:
            return {
                key: np.concatenate((self.obs[key], np.zeros(self.contract_low.shape), np.array([self.contract_state[key]])))
                for key in self.contract_state.keys()
            }

    def step(self, acts):
        if self.contract_state['a0'] == 2:
            self.metrics['contract'] = acts['a0'][:-1] 
            # proposal stage, proposed is first agent's middle dimensions
            self.params = {key: acts['a0'][:-1] for key in acts.keys()}
            self.contract_state = {'a'+str(i): 3 for i in range(self.num_agents)}
            rews = {'a'+str(i): 0.0 for i in range(self.num_agents)}
            dones = {'__all__': False}
            infos = {'a'+str(i): {} for i in range(self.num_agents)}
        elif self.contract_state['a0'] == 3:
            # agreement stage
            prod_prob = 1
            # randomly choose two agents to agree to proposed contract
            if self.num_agents > 3:
                chosen_agents = random.sample(range(1, self.num_agents), 2)
            else:
                chosen_agents = range(1, self.num_agents)
            for term in [acts['a'+str(i)][-1] for i in chosen_agents]:
                prod_prob *= term
            r = random.random()
            decision = 1 if r < prod_prob else 0  # decides the acceptance probability for both
            self.metrics['accepted'] = decision 
            for i in range(self.num_agents):
                self.contract_state['a'+str(i)] = 0
                self.params['a'+str(i)] = self.params['a'+str(i)] if decision == 1 else np.zeros(shape=self.contract_low.shape)
            rews = {'a'+str(i): 0.0 for i in range(self.num_agents)}
            dones = {'__all__': True}  # final timestep in negotiate phase
            env_dones = {'__all__': False}
            infos = {'a'+str(i): {} for i in range(self.num_agents)}
            active_agents = ['a'+str(i) for i in range(self.num_agents)]

            # iterate the subgame agents policy to game-termination, after accept / reject decided
            domain_steps = 0
            while (not env_dones['__all__']) and domain_steps < self.horizon:
                # compute actions from frozen model
                act_dict = {}
                for key in active_agents:
                    if self.shared:
                        if self.convolutional:
                            obs = self.obs[key] 
                            contract_obs =  np.concatenate((self.params[key], np.array([0])))
                            obs.update({'contract':contract_obs})
                            act_dict[key] = self.frozen_trainer.compute_single_action(
                            obs, policy_id='policy')
                        else:
                            act_dict[key] = self.frozen_trainer.compute_single_action(
                                np.concatenate((self.obs[key], self.params[key], np.array([0]))), policy_id='policy')
                    else:
                        if self.convolutional:
                            obs = self.obs[key] 
                            contract_obs =  np.concatenate((self.params[key], np.array([0])))
                            obs.update({'contract':contract_obs})
                            act_dict[key] = self.frozen_trainer.compute_single_action(
                            obs, policy_id=key)
                        else:
                            act_dict[key] = self.frozen_trainer.compute_single_action(
                            np.concatenate((self.obs[key], self.params[key], np.array([0]))), policy_id=key)

                # unsqueeze the actions, to get sensible values
                # for key in act_dict.keys():
                #     act_dict[key] = space_utils.unsquash_action(act_dict[key], self.base_env.action_space)

                _, env_rews, env_dones, infos = super().step(act_dict)

                self.last_seen_obs = {'a'+str(i): self.obs['a'+str(i)] if 'a'+str(i) in self.obs.keys() else self.last_seen_obs['a'+str(i)]
                                      for i in range(self.num_agents)}

                domain_steps += 1

                # increment rewards for current timestep
                for key in active_agents:
                    rews[key] += env_rews[key]

                # clear inactive agents from rewards
                for key in env_dones:
                    if env_dones[key] and key in active_agents:
                        active_agents.remove(key)

        if self.convolutional:
            obs = {} 
            for i in range(self.num_agents):
                key = 'a'+str(i) 
                obs[key] = self.last_seen_obs[key] 
                contract_obs =  np.concatenate((self.params[key],  np.array([self.contract_state['a'+str(i)]])))
                obs[key].update({'contract':contract_obs})
            return (
                obs,
                rews,
                dones,
                infos
            )

        else:
            return (
                {
                    'a'+str(i): np.concatenate((self.last_seen_obs['a'+str(i)], self.params['a'+str(i)], np.array([self.contract_state['a'+str(i)]])))
                    for i in range(self.num_agents)
                },
                rews,
                dones,
                infos
            )

class SeparateContractCombinedStage(SeparateContractEnv):
    """
    Combined contracting where both learning in the base env and contracting happens simultaneously. 
    The final action space of an agent is obtained by concatenating the action spaces from the base env and contracting. 

    Parameters:
        - base_env (gym.Env): The base environment on which contracts are applied.
        - contract: The contract object specifying the terms of the contract (contract logic can be found in contract/contract_lists.py).
        - num_agents (int): The number of agents interacting in the subgame stage.
        - convolutional (bool): Whether to use convolutional observations or not.
        - **kwargs: Additional keyword arguments. 

    """
    def __init__(self, base_env, contract, num_agents, convolutional,**kwargs):
        super().__init__(base_env, contract, num_agents, convolutional)

        if hasattr(self.base_env, 'continuous_action_space'): 
            self.action_space = gym.spaces.Box(low=np.concatenate((self.base_env.continuous_action_space.low, self.contract_low, np.array([0.0]))),
                                                high=np.concatenate((self.base_env.continuous_action_space.high, self.contract_high, np.array([1.0]))))
            self.continuous_action_space = True 
        else:
        # set action space (only to contract proposal, and accept-reject for other agents)
            self.action_space = gym.spaces.Box(low=np.concatenate((self.base_env.action_space.low, self.contract_low, np.array([0.0]))),
                                                high=np.concatenate((self.base_env.action_space.high, self.contract_high, np.array([1.0]))))

            self.continuous_action_space = False 

    def reset(self):
        base_obs = self.base_env.reset()
        self.obs = copy.deepcopy(base_obs)
        self.last_seen_obs = copy.deepcopy(base_obs)
        self.params = None
        self.transferred_reward_dict = {} 
        for key in ['a'+str(i) for i in range(self.num_agents)]:
            self.transferred_reward_dict[key] = []
        self.contract_state = {'a' + str(i): 2 for i in range(self.num_agents)}

        if self.convolutional:
            obs = {} 
            for i in range(self.num_agents):
                key = 'a'+str(i) 
                obs[key] = self.obs[key] 
                contract_obs =  np.concatenate((np.zeros(self.contract_low.shape), np.array([self.contract_state[key]])))
                obs[key].update({'contract':contract_obs})
            return obs 
        else:
            return {
                key: np.concatenate((self.obs[key], np.zeros(self.contract_low.shape), np.array([self.contract_state[key]])))
                for key in self.contract_state.keys()
            }

    def step(self, acts):
        if self.contract_state['a0'] == 2:
            # proposal stage, proposed is first agent's middle dimensions
            if self.continuous_action_space:
                self.params = {key: acts['a0'][self.base_env.continuous_action_space.shape[0]:-1] for key in acts.keys()}
            else:
                self.params = {key: acts['a0'][self.base_env.action_space.shape[0]:-1] for key in acts.keys()}
            self.contract_state = {'a'+str(i): 3 for i in range(self.num_agents)}
            rews = {'a'+str(i): 0.0 for i in range(self.num_agents)}
            dones = {'__all__': False}
            infos = {'a'+str(i): {} for i in range(self.num_agents)}
            
        elif self.contract_state['a0'] == 3:
            # agreement stage
            prod_prob = 1
            # randomly choose two agents to agree to proposed contract
            if self.num_agents > 3:
                chosen_agents = random.sample(range(1, self.num_agents), 2)
            else:
                chosen_agents = range(1, self.num_agents)
            for term in [acts['a'+str(i)][-1] for i in chosen_agents]:
                prod_prob *= term
            r = random.random()
            decision = 1 if r < prod_prob else 0  # decides the acceptance probability for both
            for i in range(self.num_agents):
                self.contract_state['a'+str(i)] = 0
                self.params['a'+str(i)] = self.params['a'+str(i)] if decision == 1 else np.zeros(shape=self.contract_low.shape)
            rews = {'a'+str(i): 0.0 for i in range(self.num_agents)}
            dones = {'__all__': False}  # final timestep in negotiate phase
            infos = {'a'+str(i): {} for i in range(self.num_agents)}
        else:
            if self.continuous_action_space:
                acts_soft = {key: acts[key][:self.base_env.continuous_action_space.shape[0]] for key in acts.keys()} 
                base_acts = {key: int(np.argmax(np.random.multinomial(1, scipy.special.softmax(acts_soft[key].astype(np.float64)))))
                for key in acts_soft.keys()} 
            else:
                base_acts = {key: acts[key][:self.base_env.action_space.shape[0]] for key in acts.keys()}
            return super().step(base_acts) 

        if self.convolutional:
            obs = {} 
            for i in range(self.num_agents):
                key = 'a'+str(i) 
                obs[key] = self.last_seen_obs[key] 
                contract_obs =   np.concatenate((self.params[key],  np.array([self.contract_state['a'+str(i)]])))
                obs[key].update({'contract':contract_obs}) 
            return (
                obs,
                rews,
                dones,
                infos
            )

        else:
            return (
                {
                    'a'+str(i): np.concatenate((self.last_seen_obs['a'+str(i)], self.params['a'+str(i)], np.array([self.contract_state['a'+str(i)]])))
                    for i in range(self.num_agents)
                },
                rews,
                dones,
                infos
            )


class JointEnv(MultiAgentEnv):
    """
    A single centralized agent controls all agents in the environments. 
    Agents' observations can be combined in different ways: duplicate/concatenate (concatenate individual observations), 
    or use global observations (map of the world).

    Parameters:
        - base_env (gym.Env): The base environment on which agents interact.
        - num_agents (int): The number of agents in the base environment (default is 2).
        - duplicate_obs (bool): Whether to duplicate agent observations (default is False).
        - concatenated_obs (bool): Whether to concatenate agent observations (default is False).
        - global_obs (bool): Whether to use global observations (default is False).
        - **kwargs: Additional keyword arguments.

    One of global_obs,duplicate_obs or concatenated_obs must be set to true in the config file. 
    If using pixel versions of Cleanup/Harvest, use either global obs or concatenated obs. 
    Otherwise, use duplicate_obs. 
    """
     
    def __init__(self, base_env, num_agents=2, duplicate_obs=False,concatenated_obs=False,global_obs=False,**kwargs):
        self.base_env = base_env
        self.num_agents = num_agents
        self.duplicate_obs = duplicate_obs
        self.global_obs = global_obs 
        self.concatenated_obs = concatenated_obs

        if global_obs:
            self.observation_space = self.base_env.global_observation_space
            self.action_space = self.base_env.global_action_space
        elif concatenated_obs:
            self.observation_space = self.base_env.concatenated_observation_space
            self.action_space = self.base_env.global_action_space 
        else:
            # only concatenate if duplicate_obs
            if self.duplicate_obs:
                self.observation_space = gym.spaces.Box(
                    low=np.concatenate([self.base_env.observation_space.low] * self.num_agents),
                    high=np.concatenate([self.base_env.observation_space.high] * self.num_agents)
                )
            else:
                self.observation_space = self.base_env.observation_space

            self.action_space = gym.spaces.Box(
                low=np.concatenate([self.base_env.action_space.low] * self.num_agents),
                high=np.concatenate([self.base_env.action_space.high] * self.num_agents)
            )
            self.curr_agent_lst = ['a'+str(i) for i in range(self.num_agents)]

    def reset(self):
        base_obs = self.base_env.reset()
         # assumes all agents in initial obs
        if self.global_obs :
            return {'a0': self.base_env.get_global_obs() } 
        elif self.concatenated_obs:
            # Get the values from the dictionary
            values = [] 
            for key in base_obs.keys():
                values.append(base_obs[key]['image'])
            # Concatenate the values along the last axis
            concatenated_array = np.concatenate(values, axis=-1)
            return {'a0': {'image':concatenated_array}} 
        else :
            self.agent_obs = base_obs.copy() 
            self.curr_agent_lst = ['a'+str(i) for i in range(self.num_agents)]
            return {'a0': np.concatenate([base_obs['a'+str(i)] for i in range(self.num_agents)])}

    def step(self, acts):
        if self.global_obs:
            return self.global_step(acts) 
        elif self.concatenated_obs:
            return self.concatenated_step(acts)
        else: 
            # account for agent
            action_dict = {}
            for i in range(self.num_agents):
                if 'a'+str(i) in self.curr_agent_lst:
                    # obtain action of appropriate length at appropriate index
                    action_dict['a'+str(i)] = np.array(acts['a0'][i*self.base_env.action_space.shape[0]:(i+1)*self.base_env.action_space.shape[0]])

            base_obs, env_rews, env_dones, env_infos = self.base_env.step(action_dict)

            # only overwrite observation when agent currently active
            for agent in self.curr_agent_lst:
                self.agent_obs[agent] = base_obs[agent]

            # aggregate into joint policy (no need to worry about dones stuff here)
            if self.duplicate_obs:
                obs = {'a0': np.concatenate([self.agent_obs['a'+str(i)] for i in range(self.num_agents)])}
            else:
                obs = {'a0': self.agent_obs[self.curr_agent_lst[0]]}  # take observation from arbitrary remaining agent
            rews = {'a0': sum([rew for rew in env_rews.values()])}  # report straightforward sum, not average
            dones = {'a0': env_dones['__all__'], '__all__': env_dones['__all__']}
            infos = {'a0': {key: sum([env_infos[agent][key] for agent in self.curr_agent_lst])
                            for key in env_infos[self.curr_agent_lst[0]].keys()}}

            for i in range(self.num_agents):
                if 'a'+str(i) in self.curr_agent_lst:
                    if 'a'+str(i) in env_dones.keys():
                        if env_dones['a'+str(i)]:
                            self.curr_agent_lst.remove('a'+str(i))

            return obs, rews, dones, infos

    def render(self, mode='rgb'):
        return self.base_env.render()
    
    def global_step(self,acts): 
        #retrieve agent actions form acts dict
        action_dict = {} 
        for i in range(self.num_agents):
            action_dict['a'+str(i)] = acts['a0'][i]  
        agent_list = ['a'+str(i) for i in range(self.num_agents)]
        # step in the env
        _, env_rews, env_dones, env_infos = self.base_env.step(action_dict) 
        obs = {'a0': self.base_env.get_global_obs() }
        rews = {'a0': sum([rew for rew in env_rews.values()])}  # report straightforward sum, not average 
        dones = {'a0': env_dones['__all__'], '__all__': env_dones['__all__']} 
        infos = {'a0': {key: sum([env_infos[agent][key] for agent in agent_list])
                            for key in env_infos[agent_list[0]].keys()}} 
        # return global obs, summed rewards, dodnes, infos 
        return obs, rews, dones, infos 
    
    def concatenated_step(self,acts): 
        #retrieve agent actions form acts dict
        action_dict = {} 
        for i in range(self.num_agents):
            action_dict['a'+str(i)] = acts['a0'][i]  
        agent_list = ['a'+str(i) for i in range(self.num_agents)]
        # step in the env
        base_obs, env_rews, env_dones, env_infos = self.base_env.step(action_dict) 
        values = [] 
        for key in base_obs.keys():
            values.append(base_obs[key]['image'])
        # Concatenate the values along the last axis
        concatenated_array = np.concatenate(values, axis=-1)
        obs = {'a0':{'image':concatenated_array}} 
        rews = {'a0': sum([rew for rew in env_rews.values()])}  # report straightforward sum, not average 
        dones = {'a0': env_dones['__all__'], '__all__': env_dones['__all__']} 
        infos = {'a0': {key: sum([env_infos[agent][key] for agent in agent_list])
                            for key in env_infos[agent_list[0]].keys()}} 
        # return global obs, summed rewards, dodnes, infos 
        return obs, rews, dones, infos

class NegotiationSolver(SeparateContractEnv):
    """
    An environment representing a negotiation solver for contract negotiation tasks.
    This solver uses agent's frozen value functions from the subgame stages to find an optimal contract.
    There are 2 possible decision rules - 
    1) max: find the contract that maximizes sum of agent values
    2) majority: find the contract that maximizes sum of agent values subject to majority of agents accepting the contract.
    Note - an agent accepts a contract if V(s,c) >= V(s,0) i.e., the agent gets a higher value with that contract compared to a null contract


    Parameters:
        - base_env (SeparateContractEnv): The base environment for contract negotiation.
        - contract (Contract): The contract definition used for negotiation.
        - num_agents (int): The number of agents participating in the negotiation.
        - horizon (int): Episode length
        - trainer_config (dict): Parameters from the subgame stage
        - trainer_env (str): The environment name used by the trainer.
        - trainer_path (str): The path to a pre-trained reinforcement learning model (from the subgame stage)
        - convolutional (bool): Whether the observation space is convolutional or not.
        - shared (bool): Whether agents share the same policy (default is False).
        - env_params (dict): Additional environment parameters (default is None).
        - contract_samples (int): The number of contract samples to sample and pick from during each episode (default is 50).
        - decision_rule (str): The negotiation decision rule ('majority' or 'max') (default is 'majority').
        - **kwargs: Additional keyword arguments.
    """ 
    def __init__(self, base_env, contract, num_agents, horizon, trainer_config, trainer_env, trainer_path, convolutional, shared,env_params=None,
                 contract_samples=50,decision_rule='majority',**kwargs):
        super().__init__(base_env, contract, num_agents, convolutional)

        self.horizon = horizon

        self.convolutional = convolutional
        # upload frozen policy from last step
        self.frozen_trainer = ppo.PPOTrainer(config=trainer_config, env=trainer_env)
        self.frozen_trainer.load_checkpoint(trainer_path)
        self.shared = shared
        self.contract_param_space = gym.spaces.Box(low=contract.contract_space.low, high=contract.contract_space.high)
        self.contract_low = contract.contract_space.low
        self.num_samples = contract_samples
        self.decision_rule = decision_rule
        self.config = trainer_config
        print('Decision Rule is {}'.format(self.decision_rule))
    def reset(self):
        self.metrics = {'contract':-1,'accepted':0}  
        base_obs = self.base_env.reset()
        self.obs = copy.deepcopy(base_obs)
        self.last_seen_obs = copy.deepcopy(base_obs)
        self.params = None
        self.contract_state = {'a' + str(i): 0 for i in range(self.num_agents)}

        self.transferred_reward_dict = {} 
        for key in ['a'+str(i) for i in range(self.num_agents)]:
            self.transferred_reward_dict[key] = []

        self.contract_param = self.negotiate()
        self.params = {key: self.contract_param for key in ['a'+str(i) for i in range(self.num_agents)]}
        
        if self.convolutional:
            obs = {} 
            for i in range(self.num_agents):
                key = 'a'+str(i) 
                obs[key] = self.obs[key] 
                contract_obs =  np.concatenate((self.params['a'+str(i)], np.array([0])))
                obs[key].update({'contract':contract_obs})
            return obs  
        else:
            return {
                key: np.concatenate((self.obs[key], self.params[key], np.array([0])))
                for key in self.contract_state.keys()
            }
        
    def compute_vals(self,obs):
        act_dict = {}
        vals = {} 
        for key in obs:
            if self.shared:
                act_dict[key] = self.frozen_trainer.compute_single_action(
                obs[key], policy_id='policy')
                model = self.frozen_trainer.get_policy('policy').model
            else:
                act_dict[key] = self.frozen_trainer.compute_single_action(
                obs[key], policy_id=key)
                model = self.frozen_trainer.get_policy(key).model
            vals[key] = model.value_function().item() 
        return vals 
    
    def negotiate(self):
        # Samples a set of random contracts and finds the best contract parameter (compares against an agent's values for a null contract)
        all_vals =[] 
        all_params = [] 
        # Null contract 
        default_param = self.contract_low 
        all_params.append(default_param)
        if self.convolutional:
            obs = {} 
            for i in range(self.num_agents):
                key = 'a'+str(i) 
                obs[key] = self.obs[key] 
                contract_obs =  np.concatenate((default_param, np.array([0])))
                obs[key].update({'contract':contract_obs})
        else:
            obs =  {
                key: np.concatenate((self.obs[key], default_param, np.array([0])))
                for key in self.contract_state.keys()
            }
        # Get the agent values from initial state, null contract
        default_vals = self.compute_vals(obs)
        all_vals.append(default_vals)
        for i in range(self.num_samples):
            # Sample a random contract
            sampled_param = self.contract_param_space.sample()
            all_params.append(sampled_param)
            if self.convolutional:
                obs = {} 
                for i in range(self.num_agents):
                    key = 'a'+str(i) 
                    obs[key] = self.obs[key] 
                    contract_obs =  np.concatenate((sampled_param, np.array([0])))
                    obs[key].update({'contract':contract_obs})
            else:
                obs =  {
                    key: np.concatenate((self.obs[key], sampled_param, np.array([0])))
                    for key in self.contract_state.keys()
                }
            # Get the agent values for that contract, initial state 
            vals = self.compute_vals(obs)
            all_vals.append(vals)
        return self.compute_best_param(all_vals,all_params) 

    def compute_best_param(self,all_vals,all_params,dec_rule=None):
        if dec_rule is None:
            dec_rule = self.decision_rule
        if dec_rule == 'max':
            # In this decision rule, just sample a contract that maximizes the predicted sum of agent values 
            welfares = [] 
            for k in all_vals:
                welfares.append(sum(k.values())) 
            best_idx = np.argmax(welfares)
            best_param = all_params[best_idx]
        elif dec_rule == 'majority': 
            # In this decision rule, sample a contract that maximizes the sum of agent values subject to majority of the agents accepting the contract
            # An agent accepts a contract c ig Vi(s,0) <= Vi(s,c) , i.e., an agent's value with that contract is greater than its value with the null contract
            default_vals = all_vals[0]
            accepted_vals, accepted_params = [default_vals], [all_params[0]] 
            for k1 in all_vals[1:] : 
                accepted, rejected = 0,0 
                for k in k1:
                    if k1[k] > default_vals[k]:
                        accepted+=1 
                    else:
                        rejected+=1
                if accepted >= rejected:
                    accepted_vals.append(k1)
                    accepted_params.append(all_params[all_vals.index(k1)])
            best_param = self.compute_best_param(accepted_vals,accepted_params,dec_rule='max') 
        else :
            pass 
        return best_param 
    
if __name__ == '__main__':
    pass 