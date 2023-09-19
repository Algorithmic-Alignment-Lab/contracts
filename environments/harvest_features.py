import sys

import gym
from ray.rllib.env import MultiAgentEnv
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy
from copy import deepcopy 

# the Harvest map
MAP = [
    "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",
    "@ P   P      A    P AAAAA    P  A P  @",
    "@  P     A P AA    P    AAA    A  A  @",
    "@     A AAA  AAA    A    A AA AAAA   @",
    "@ A  AAA A    A  A AAA  A  A   A A   @",
    "@AAA  A A    A  AAA A  AAA        A P@",
    "@ A A  AAA  AAA  A A    A AA   AA AA @",
    "@  A A  AAA    A A  AAA    AAA  A    @",
    "@   AAA  A      AAA  A    AAAA       @",
    "@ P  A       A  A AAA    A  A      P @",
    "@A  AAA  A  A  AAA A    AAAA     P   @",
    "@    A A   AAA  A A      A AA   A  P @",
    "@     AAA   A A  AAA      AA   AAA P @",
    "@ A    A     AAA  A  P          A    @",
    "@       P     A         P  P P     P @",
    "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@",
]

APPLE_RADIUS = 2
HARVEST_VIEW_SIZE = 7

MOVE_ACTIONS = [[0, -1], [0, 1], [-1, 0], [1, 0]]
FIRE_DIRECTIONS = [[-1, 0], [0, 1], [1, 0], [0, -1]]

SPAWN_PROB = [0, 0.005, 0.02, 0.05]

DEFAULT_COLOURS = {
    b" ": np.array([0, 0, 0], dtype=np.uint8),  # Black background
    b"0": np.array([0, 0, 0], dtype=np.uint8),  # Black background beyond map walls
    b"": np.array([180, 180, 180], dtype=np.uint8),  # Grey board walls
    b"@": np.array([180, 180, 180], dtype=np.uint8),  # Grey board walls
    b"A": np.array([0, 255, 0], dtype=np.uint8),  # Green apples
    b"F": np.array([255, 255, 0], dtype=np.uint8),  # Yellow firing beam
    b"P": np.array([159, 67, 255], dtype=np.uint8),  # Generic agent (any player)
    # Colours for agents. R value is a unique identifier
    b"1": np.array([0, 0, 255], dtype=np.uint8),  # Pure blue
    b"2": np.array([2, 81, 154], dtype=np.uint8),  # Sky blue
    b"3": np.array([204, 0, 204], dtype=np.uint8),  # Magenta
    b"4": np.array([216, 30, 54], dtype=np.uint8),  # Red
    b"5": np.array([254, 151, 0], dtype=np.uint8),  # Orange
    b"6": np.array([100, 255, 255], dtype=np.uint8),  # Cyan
    b"7": np.array([99, 99, 255], dtype=np.uint8),  # Lavender
    b"8": np.array([250, 204, 255], dtype=np.uint8),  # Pink
    b"9": np.array([238, 223, 16], dtype=np.uint8),  # Yellow
}

# multi-agent simplified cleanup domain
class HarvestFeatures(MultiAgentEnv):
    # use a0, a1 for agent names
    def __init__(self, num_agents=2, horizon=1000, image_obs=False,**kwargs):
        # map always reset as above
        self.map = MAP
        self.num_agents = num_agents
        self.horizon = horizon
        self.timesteps = 0
        self.world_map_color = np.full(
            (len(self.map) + HARVEST_VIEW_SIZE * 2, len(self.map[0]) + HARVEST_VIEW_SIZE * 2, 3),
            fill_value=0,
            dtype=np.uint8,
        )
        self.image_obs = image_obs

        # make a list of the potential apple and waste spawn points
        self.initialize_arrays()  # call before compute probs due to potential_waste_area
        self.initialize_players()

        if self.image_obs:
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(2 * HARVEST_VIEW_SIZE + 1, 2 * HARVEST_VIEW_SIZE + 1, 3),
                dtype=np.uint8,
            )
        else:
            self.observation_space = gym.spaces.Box(low=np.array([0.0]*(10 + 2 * self.num_agents)),
                                                    high=np.array([
                                                        len(self.map), len(self.map[0]), 4, len(self.map), len(self.map[0]), 4,
                                                        len(self.map), len(self.map[0]),
                                                        len(self.apple_points) + 1,
                                                        len(self.apple_points) + 1] + [1] * (2*self.num_agents)))
        self.action_space = gym.spaces.Discrete(7) # logits of the true action
        self.continuous_action_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(7,))

    def initialize_arrays(self):
        self.spawn_points = []
        self.wall_points = []
        self.apple_points = []

        for row in range(len(self.map)):
            for col in range(len(self.map[0])):
                if self.map[row][col] == "P":
                    self.spawn_points.append([row, col])
                    # self.single_update_map(row, col, b"P")
                elif self.map[row][col] == "@":
                    self.wall_points.append([row, col])
                    self.single_update_map(row, col, b"@")
                if self.map[row][col] == "A":
                    self.apple_points.append([row, col])
                    self.single_update_map(row, col, b"A")  # color in the

        self.current_apple_points = self.apple_points.copy()

    def initialize_players(self):
        # spawn agents
        random_idx = list(range(len(self.spawn_points)))
        random.shuffle(random_idx)  # permutes
        self.agent_pos = {'a'+str(i): self.spawn_points[random_idx[i]] for i in range(self.num_agents)}
        for agent_pos in self.agent_pos.values():
            self.single_update_map(agent_pos[0], agent_pos[1], b"P")
        self.agent_orientation = {'a'+str(i): np.random.randint(low=0, high=4) for i in range(self.num_agents)}

    def single_update_map(self, row, col, char):
        self.world_map_color[row + HARVEST_VIEW_SIZE, col + HARVEST_VIEW_SIZE] = DEFAULT_COLOURS[char]

    def count_apples_in_radius(self, radius, loc):
        # count number of CURRENT apples close
        num_apples = 0
        for j in range(-radius, radius + 1):
            for k in range(-radius, radius + 1):
                # if within radius and a current apple
                if j ** 2 + k ** 2 <= radius and [loc[0] + j,
                                                        loc[1] + k] in self.current_apple_points:
                    num_apples += 1

        return num_apples

    def spawn_apples(self):
        # spawn apples
        for apple_pos in self.apple_points:
            if apple_pos not in self.current_apple_points and apple_pos not in self.agent_pos.values():
                # count number of CURRENT apples close
                num_apples = self.count_apples_in_radius(APPLE_RADIUS, apple_pos)

                # spawn with given probability
                spawn_prob = SPAWN_PROB[min(num_apples, 3)]
                rand_num = random.random()
                if rand_num < spawn_prob:
                    self.current_apple_points.append(apple_pos)
                    self.single_update_map(apple_pos[0], apple_pos[1], b"A")

    def compute_closest_apples(self):
        closest_apples = {'a'+str(i): [0, 0] for i in range(self.num_agents)}  # sentinel value if there are no apples
        if self.current_apple_points:
            closest_apples = {}
            for key in self.agent_pos.keys():
                distances = np.sum(np.abs(np.array(self.current_apple_points) - np.array(self.agent_pos[key])), axis=1)
                min_dist_idx = np.argmin(distances)
                closest_apples[key] = self.current_apple_points[min_dist_idx]
        return closest_apples

    def compute_closest_pos(self):
        closest_pos = {'a'+str(i): [0, 0] for i in range(self.num_agents)}  # sentinel value if there is no waste
        for key in self.agent_pos.keys():
            distances = np.sum(np.abs(
                np.array([[np.inf, np.inf] if k == key else self.agent_pos[key] for k in self.agent_pos.keys()]) - np.array(self.agent_pos[key])
            ), axis=1)
            min_dist_idx = np.argmin(distances)
            closest_pos[key] = (self.agent_pos['a'+str(min_dist_idx)], self.agent_orientation['a'+str(min_dist_idx)])
        return closest_pos

    def step(self, acts):
        # execute moves
        move_squares = {}
        rewards = {key: 0.0 for key in acts.keys()}
        infos = {key: {'eaten_apples': 0, 'eaten_close_apples': 0} for key in acts.keys()}

        # do stay first, so that it has highest priority
        for key in acts.keys():
            if acts[key] > 3:
                move_squares[key] = self.agent_pos[key]

        # execute actual movements next, if requested
        for key in acts.keys():
            if acts[key] in range(4):
                tmp_move = self.agent_pos[key].copy()
                tmp_move[0] += MOVE_ACTIONS[acts[key]][0]
                tmp_move[1] += MOVE_ACTIONS[acts[key]][1]
                # if already moved there by other agent OR want to move into wall
                if tmp_move in move_squares.values() or tmp_move in self.wall_points:
                    move_squares[key] = self.agent_pos[key]  # no move, simply stay where one is
                else:
                    self.single_update_map(self.agent_pos[key][0], self.agent_pos[key][1], b"0")
                    self.single_update_map(tmp_move[0], tmp_move[1], b"P")
                    move_squares[key] = tmp_move

        # update movements to reflect this
        self.agent_pos = {key: move_squares[key] if key in move_squares.keys() else self.agent_pos[key] for key in self.agent_pos.keys()}

        # consume apples at all moved to squares
        for key in move_squares.keys():
            if move_squares[key] in self.current_apple_points:
                rewards[key] += 1
                infos[key]['eaten_apples'] += 1
                # self.single_update_map(move_squares[key][0], move_squares[key][1], b"0")
                if self.count_apples_in_radius(5, self.agent_pos[key]) < 4:
                    infos[key]['eaten_close_apples'] += 1
                    self.metrics['low_density_apples_eaten'] +=1
                self.current_apple_points.remove(move_squares[key]) 
                self.metrics['total_apples_eaten'] +=1 

        # execute rotations
        for key in acts.keys():
            if acts[key] == 5:
                self.agent_orientation[key] = (self.agent_orientation[key] + 1) % 4
            if acts[key] == 6:
                self.agent_orientation[key] = (self.agent_orientation[key] - 1) % 4

        # manage firing actions (iterate by agent index so we can reference other agents)
        for i in range(self.num_agents):
            if acts['a'+str(i)] in [7]:
                no_offset = np.array(self.agent_pos['a'+str(i)].copy())
                offset_1 = np.array(self.agent_pos['a'+str(i)].copy()) + np.array(FIRE_DIRECTIONS[(self.agent_orientation['a'+str(i)] + 1) % 4])
                offset_2 = np.array(self.agent_pos['a'+str(i)].copy()) - np.array(FIRE_DIRECTIONS[(self.agent_orientation['a'+str(i)] + 1) % 4])
                beam_width = [no_offset, offset_1, offset_2]
                for beam in beam_width:
                    for j in range(6):  # beam length is 5
                        current_beam = beam + j * np.array(FIRE_DIRECTIONS[self.agent_orientation['a'+str(i)]])
                        if np.ndarray.tolist(current_beam) in self.wall_points:
                            break  # stop looking at this particular beam if the wall has been hit
                        # elif acts['a'+str(i)] == 7:
                        #     # fire beam, check if another agent is in beam
                        #     for j in range(self.num_agents):
                        #         if i == j:
                        #             continue
                        #         if np.ndarray.tolist(current_beam) == self.agent_pos['a'+str(j)]:
                        #             rewards['a'+str(j)] -= 50.0  # costs 50 to be punished
            # if acts['a'+str(i)] == 7:
            #     rewards['a' + str(i)] -= 1.0  # costs 1 to punish opponent

        # spawn apples
        self.spawn_apples()

        # make observations, rewards, dones, infos to be passed
        compute_closest_apples = self.compute_closest_apples()
        compute_closest_pos = self.compute_closest_pos()  # necessary in multiagent case, get closest agent's position + orientation

        # get number of close apples
        total_close_apples = {key: self.count_apples_in_radius(5, self.agent_pos[key]) for key in self.agent_pos.keys()}
        feature_obs = {key: np.array([self.agent_pos[key][0], self.agent_pos[key][1],
                                           self.agent_orientation[key],
                                           compute_closest_pos[key][0][0], compute_closest_pos[key][0][1],
                                           compute_closest_pos[key][1],
                                           compute_closest_apples[key][0], compute_closest_apples[key][1],
                                           total_close_apples[key],
                                           len(self.current_apple_points)] + [0.0] * (2 * self.num_agents))
                            for key in ['a' + str(i) for i in range(self.num_agents)]}
        if self.image_obs:
            # convolutional features
            observations = {key: self.world_map_color[
                                 self.agent_pos[key][0]: self.agent_pos[key][0] + 2 * HARVEST_VIEW_SIZE + 1,
                                 self.agent_pos[key][1]: self.agent_pos[key][1] + 2 * HARVEST_VIEW_SIZE + 1,
                                 ]
                            for key in ['a' + str(i) for i in range(self.num_agents)]}

        else:
            # feature-engineered observations
            observations = feature_obs 

        for key in acts.keys():
            infos[key]['feature_obs'] = feature_obs[key]

        self.timesteps += 1
        dones = {'__all__': self.timesteps == self.horizon, 'a0': self.timesteps == self.horizon, 'a1': self.timesteps == self.horizon}  # rely on environment horizon: env never terminates
        
        raw_rewards = 0 
        for k,v in rewards.items() :
            raw_rewards +=v 
            self.total_reward_dict[k].append(v)
        self.metrics['raw_env_rewards'] += raw_rewards

        if dones['__all__']:
            self.metrics['equality'] = self.compute_equality(deepcopy(self.total_reward_dict))
            self.metrics['sustainability'] = self.compute_sustainability(deepcopy(self.total_reward_dict))
      
        return observations, rewards, dones, infos

    def reset(self):
        # reset player locations, orientations, and load everything again from map
        self.world_map_color = np.full(
            (len(self.map) + HARVEST_VIEW_SIZE * 2, len(self.map[0]) + HARVEST_VIEW_SIZE * 2, 3),
            fill_value=0,
            dtype=np.uint8,
        )

        self.initialize_arrays()
        self.initialize_players()

        # spawn apples
        self.spawn_apples()

        # compute closest apples and wastes
        compute_closest_apples = self.compute_closest_apples()
        compute_closest_pos = self.compute_closest_pos()  # necessary in multiagent case, get closest agent

        # get number of close apples
        total_close_apples = {key: self.count_apples_in_radius(5, self.agent_pos[key]) for key in self.agent_pos.keys()}

        if self.image_obs:
            # convolutional features
            observations = {key: self.world_map_color[
                             self.agent_pos[key][0]: self.agent_pos[key][0] + 2 * HARVEST_VIEW_SIZE + 1,
                             self.agent_pos[key][1]: self.agent_pos[key][1] + 2 * HARVEST_VIEW_SIZE + 1,
            ]
                for key in ['a' + str(i) for i in range(self.num_agents)]}

        else:
            # feature-engineered observations
            observations = {key: np.array([self.agent_pos[key][0], self.agent_pos[key][1],
                                           self.agent_orientation[key],
                                           compute_closest_pos[key][0][0], compute_closest_pos[key][0][1],
                                           compute_closest_pos[key][1],
                                           compute_closest_apples[key][0], compute_closest_apples[key][1],
                                           total_close_apples[key],
                                           len(self.current_apple_points)] + [0.0] * (2*self.num_agents))
                            for key in ['a'+str(i) for i in range(self.num_agents)]}

        self.timesteps = 0
        self.total_reward_dict = {} 
        for key in ['a'+str(i) for i in range(self.num_agents)]:
            self.total_reward_dict[key] = []
        # plt.imshow(self.observations['a0'])
        # plt.show()
        self.metrics = {'total_apples_eaten':0,'low_density_apples_eaten':0,'raw_env_rewards':0,'transfers':0}
        return observations

    def render(self):
        pass

    def compute_equality(self,reward_dict): 
        eq = 0 
        total_sum = 0
        reward_dict = {k:sum(v) for k,v in reward_dict.items()}
        n = len(reward_dict.keys())
        for i in reward_dict.keys(): 
            for j in reward_dict.keys(): 
                    eq += abs(reward_dict[i] - reward_dict[j])
            total_sum += reward_dict[i]
        if total_sum ==0 :
            total_sum = 0.001
        eq = 1 - eq/(2*n*total_sum)
        return eq

    def compute_sustainability(self,reward_dict): 
        avg_times =[] 
        for k in reward_dict.keys():
            t_sum = 0 
            for t,i in enumerate(reward_dict[k]):
                t_sum += t*i
            denom = sum(reward_dict[k]) 
            denom = max(denom,1)
            avg_times.append(t_sum/denom)
        return np.mean(avg_times)

