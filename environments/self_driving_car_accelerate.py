import os.path

import gym
from ray.rllib.env import MultiAgentEnv
import numpy as np
import random
import torch
# from moviepy.editor import *

# threshold constants in environment (bounding per-timestep acceleration + velocity)
ACCEL_LOW_THRESH = -0.1
ACCEL_HIGH_THRESH = 0.1
VEL_LOW_THRESH = 0.0
VEL_HIGH_THRESH = 0.25  # normal car speed limit
VEL_HIGH_THRESH_AMBULANCE = 1.0  # ambulance speed limit

# names of agents passed must be a0, a1
class SelfAcceleratingCarEnv(MultiAgentEnv):
    def __init__(self, low_bound=-10.0, high_bound=10.0, start_vel=0.2, start_vel_ambulance=0.8, num_agents=2, collision_on=False, **kwargs):
        # todo: implement collision logic for multiple agents
        #super().__init__()
        self.num_agents = num_agents
        #self._agent_ids = ['a'+str(i) for i in range(self.num_agents)]
        self.low_bound = low_bound
        self.high_bound = high_bound
        self.start_vel = start_vel
        self.start_vel_ambulance = start_vel_ambulance
        self.collision_on = collision_on
        self.agent_positions = {'a'+str(i): self.low_bound for i in range(self.num_agents)}
        self.agent_vels = {'a'+str(i): start_vel for i in range(self.num_agents)}
        self.agent_dones = {"a"+str(i): False for i in range(num_agents)}
        self.agent_dones['__all__'] = False
        self.already_left = False
        self.crossed_agents = []
        self.dist_to_front = {'a'+str(i): -1 for i in range(self.num_agents)}
        self.current_episode_images = []
        self.metrics = {'transfers':0}

        # multiagent params (giving buffer in case of overrun)
        self.observation_space = gym.spaces.Box(low=self.low_bound - 20,
                                                high=self.high_bound + 20,
                                                shape=(2*(self.num_agents+1)+3,),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(low=ACCEL_LOW_THRESH,
                                           high=ACCEL_HIGH_THRESH,
                                           shape=(1,),
                                           dtype=np.float32)  # velocity between 0-1 absolute, vary all else in relation

    def reset(self):
        for key in ['a'+str(i) for i in range(self.num_agents)]:
            if key == 'a0':
                # always spawn ambulance in back half
                self.agent_positions[key] = random.random() * self.low_bound / 2 + self.low_bound / 2
                self.agent_vels[key] = self.start_vel_ambulance
            else:
                # start car between 3/16ths and 1/4, for initializations that scale well as the number of agents grows
                self.agent_positions[key] = random.random() * self.low_bound / 16 + self.low_bound * 3 / 16
                self.agent_vels[key] = self.start_vel
            # self.agent_positions[key] = random.choice([0.1*i for i in range(11)]) * self.low_bound
            self.agent_dones[key] = False
        self.agent_dones['__all__'] = False
        self.already_left = False
        self.crossed_agents = []  # in order of crossing too
        self.dist_to_front = {'a'+str(i): -1 for i in range(self.num_agents)}
        # if self.current_episode_images:
        #     clip = ImageSequenceClip(list(self.current_episode_images), fps=20)
        #     render_count = 1
        #     while os.path.exists('render_' + str(render_count) + '.gif'):
        #         render_count += 1
        #     clip.write_gif('render_' + str(render_count) + '.gif', fps=20)
        self.current_episode_images = []
        self.metrics = {'transfers':0}
        return {key: np.concatenate(([self.agent_positions[key], self.agent_vels[key]],
                                         [self.agent_positions['a'+str(i)] - self.agent_positions[key] for i in range(self.num_agents)] +
                        [self.agent_vels['a'+str(i)] for i in range(self.num_agents)],
                                         [1.0 if self.agent_positions['a0'] > 0 else 0.0],
                                         [1.0 if self.agent_positions[key] > 0 else 0.0],
                                        [0.0]))
                    for key in ['a'+str(i) for i in range(self.num_agents)]}

    def check_if_crashed(self, agent_new_pos):
        if self.crossed_agents:
            for i in range(len(self.crossed_agents)-1):
                updated_position_front = agent_new_pos[self.crossed_agents[i]] if self.crossed_agents[i] in agent_new_pos.keys() else self.agent_positions[self.crossed_agents[i]]
                updated_position_back = agent_new_pos[self.crossed_agents[i+1]] if self.crossed_agents[i+1] in agent_new_pos.keys() else self.agent_positions[self.crossed_agents[i+1]]

                # if you disobey the merge order, return crash occurred
                if updated_position_front < updated_position_back:
                    return True
        return False

    def make_new_pos_consistent(self, agent_new_pos):
        pre_merged = []

        if self.crossed_agents:
            for i in range(len(self.crossed_agents) - 1):
                updated_position_front = agent_new_pos[self.crossed_agents[i]] if self.crossed_agents[i] in agent_new_pos.keys() else self.agent_positions[self.crossed_agents[i]]
                updated_position_back = agent_new_pos[self.crossed_agents[i+1]] if self.crossed_agents[i+1] in agent_new_pos.keys() else self.agent_positions[self.crossed_agents[i+1]]

                # if you disobey the merge order, undo the current position change
                if updated_position_front < updated_position_back:
                    if self.crossed_agents[i] in agent_new_pos.keys() and self.crossed_agents[i+1] in agent_new_pos.keys():
                        agent_new_pos[self.crossed_agents[i+1]] = updated_position_front - 0.01
                        if agent_new_pos[self.crossed_agents[i+1]] < 0:
                            pre_merged.append(self.crossed_agents[i + 1])

        # update to reflect pre-merge
        self.crossed_agents = [a for a in self.crossed_agents if a not in pre_merged]

    def update_rel_rank(self, agent_new_pos):
        just_passed_agents = []

        for key in agent_new_pos.keys():
            if self.agent_positions[key] < 0.0 and agent_new_pos[key] > 0.0:
                just_passed_agents.append(key)

        rel_times = [(agent, -self.agent_positions[agent] / (agent_new_pos[agent] - self.agent_positions[agent])
            if agent_new_pos[agent] - self.agent_positions[agent] != 0.0 else 0.0)
                         for agent in just_passed_agents]

        # sort crossed agents by their relative arrival time
        sorted_rel_times = sorted([rel_times[i][0] for i in range(len(rel_times))], key=lambda k: k[1])

        for sort_agent in sorted_rel_times:
            self.crossed_agents.append(sort_agent)

    def update_infos(self, agent_new_pos, infos):
        for key in agent_new_pos.keys():
            # check if this particular agent just passed the barrier
            if self.agent_positions[key] < 0.0 and agent_new_pos[key] > 0.0:
                infos[key]['just_passed'] = True
                # get relative position after pass, which is determined at this moment
                dist_to_front = 0.0
                for i in range(self.num_agents):
                    # if distinct agent i ahead of current agent
                    if 'a' + str(i) != key:
                        if 'a' + str(i) not in agent_new_pos.keys() or agent_new_pos['a' + str(i)] > agent_new_pos[key]:
                            if 'a' + str(i) not in agent_new_pos.keys():
                                if self.high_bound - agent_new_pos[key] > dist_to_front:
                                    dist_to_front = self.high_bound + 1 - agent_new_pos[key]
                            else:
                                if agent_new_pos['a'+str(i)] - agent_new_pos[key] > dist_to_front:
                                    dist_to_front = agent_new_pos['a' + str(i)] - agent_new_pos['a' + str(i)]
                self.dist_to_front['a' + str(i)] = dist_to_front

                if key == 'a0':
                    # set ambulance stats
                    infos[list(agent_new_pos.keys())[0]]['ambulance_rank'] = self.crossed_agents.index('a0') + 1
                    infos[list(agent_new_pos.keys())[0]]['ambulance_dist_to_front'] = dist_to_front

    def step(self, acts):
        key_lst = list(acts.keys())

        if self.agent_dones['__all__']:
            return {key: np.concatenate(([self.agent_positions[key], self.agent_vels[key]],
                                         [self.agent_positions['a'+str(i)] - self.agent_positions[key] for i in range(self.num_agents)] +
                        [self.agent_vels['a'+str(i)] for i in range(self.num_agents)],
                                         [1.0 if self.agent_positions['a0'] > 0 else 0.0],
                                         [1.0 if self.agent_positions[key] > 0 else 0.0], 
                                         [1.0 if self.collision_check_all(self.agent_positions, acts.keys()) else 0.0]))
                    for key in acts.keys()}, \
                   {key: 0.0 for key in acts.keys()}, \
                   self.agent_dones, \
                   {key: {'just_passed': False,
                          'is_crashed': self.collision_check_all(self.agent_positions, acts.keys()) if key == key_lst[0] else 0,
                          'ambulance_rank': (self.crossed_agents.index('a0')+1 if 'a0' in self.crossed_agents else self.num_agents)
                       if key == key_lst[0] else 0.0} for key in key_lst}

        # randoms = {key: random.uniform(a=-0.01, b=0.01) for key in acts.keys()}  # random perturbation of velocity

        # allow ambulance to go faster
        agent_new_vel = {key: max([min([max([min([acts[key][0], ACCEL_HIGH_THRESH]), ACCEL_LOW_THRESH]) + self.agent_vels[key],
                                        VEL_HIGH_THRESH_AMBULANCE if key == 'a0' else VEL_HIGH_THRESH]), VEL_LOW_THRESH])  # threshold between 0.0 and 2.0
                         for key in acts.keys()}  # (manually added acceleration checks)

        self.agent_vels = {key: agent_new_vel[key] if key in acts.keys() else self.agent_vels[key]
                           for key in self.agent_vels.keys()}

        # modify position accounting for agent acceleration (move forward by accelerated velocity)
        agent_new_pos = {key: self.agent_vels[key] + self.agent_positions[key] for key in acts.keys()}

        # update infos
        infos = {key: {'just_passed': False,
                       'is_crashed': 0,
                       'ambulance_rank': (self.crossed_agents.index('a0')+1 if 'a0' in self.crossed_agents else self.num_agents)
                       if key == key_lst[0] else 0.0,
                       'ambulance_dist_to_front': (self.dist_to_front['a0'] if self.dist_to_front['a0'] > -1
                                                   else self.high_bound - self.low_bound)
                       if key == key_lst[0] else 0.0} for key in key_lst}

        self.update_rel_rank(agent_new_pos)
        self.update_infos(agent_new_pos, infos)

        # either detect collision and / or enforce compliance
        if self.collision_on:
            if self.check_if_crashed(agent_new_pos):
                # the agents have crashed as a result of prior action, finish the env immediately and penalize
                self.agent_dones['__all__'] = True

                for key in acts.keys():
                    self.agent_dones[key] = True

                infos[key_lst[0]]['ambulance_rank'] = self.num_agents  # ambulance rank is set to lowest value when crashed, will be self.num_agents
                infos[key_lst[0]]['is_crashed'] = 1
                return {key: np.concatenate(([self.agent_positions[key], self.agent_vels[key]],
                                             [self.agent_positions['a' + str(i)] - self.agent_positions[key] for i in
                                              range(self.num_agents)] +
                                             [self.agent_vels['a' + str(i)] for i in range(self.num_agents)],
                                             [1.0 if self.agent_positions['a0'] > 0 else 0.0],
                                             [1.0 if self.agent_positions[key] > 0 else 0.0],
                                             [1.0]))
                        for key in acts.keys()}, \
                       {key: -10000.0 for key in self.agent_positions.keys()}, \
                       self.agent_dones, \
                       infos
        else:
            self.make_new_pos_consistent(agent_new_pos)

        # update to new positions (relevant if not collided)
        self.agent_positions = {key: agent_new_pos[key] if key in acts.keys() else self.agent_positions[key]
                                for key in self.agent_positions.keys()}

        agent_rews = {key: -1.0 for key in acts.keys()}
        if 'a0' in agent_rews.keys():
            agent_rews['a0'] -= 99.0  # ambulance has higher penalty

        # update dones to determine if an agent is finished in the environment
        for i in range(self.num_agents):
            if self.agent_positions['a'+str(i)] > self.high_bound:
                self.agent_positions['a'+str(i)] = self.high_bound + 1  # set to sentinel value, to avoid overflow
                self.agent_dones['a'+str(i)] = True

        # set agent_dones to check if all are done
        all_is_done = True
        for key in acts.keys():
            if not self.agent_dones[key]:
                all_is_done = False
        self.agent_dones['__all__'] = all_is_done

        # give relative positions, not absolute positions, appended with signal for whether ambulance has passed
        return {key: np.concatenate(([self.agent_positions[key], self.agent_vels[key]],
                                     [self.agent_positions['a'+str(i)] - self.agent_positions[key] for i in range(self.num_agents)] +
                    [self.agent_vels['a'+str(i)] for i in range(self.num_agents)],
                                     [1.0 if self.agent_positions['a0'] > 0 else 0.0],
                                     [1.0 if self.agent_positions[key] > 0 else 0.0],
                                     [0.0]))
                for key in acts.keys()}, \
               agent_rews, \
               self.agent_dones, \
               infos

    def render(self, mode='rgb'):
        img = 255 * np.ones(shape=(round(self.high_bound - self.low_bound) * 20 + 20, self.num_agents * 50, 3))  # encodes white

        # draw in merge boundary (coloured red)
        img[200,:,1] = 0
        img[200,:,2] = 0

        # draw in cars
        for i in range(self.num_agents):
            agent_pixel_height = round((max([min([self.agent_positions['a'+str(i)], self.high_bound]), self.low_bound]) - self.low_bound) * 20)
            img[agent_pixel_height:agent_pixel_height+20,i*50+10:i*50+40,0] = 0
            img[agent_pixel_height:agent_pixel_height + 20, i * 50 + 10:i * 50 + 40, 1] = 0
            if i > 0: # if i==0, want ambulance to be blue
                img[agent_pixel_height:agent_pixel_height + 20, i * 50 + 10:i * 50 + 40, 2] = 0

        self.current_episode_images.append(img)
        return True
