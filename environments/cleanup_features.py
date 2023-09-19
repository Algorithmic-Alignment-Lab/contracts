import gym
from ray.rllib.env import MultiAgentEnv
import numpy as np
import random
import scipy
from copy import deepcopy 
# the CleanUp map
MAP = [
    "@@@@@@@@@@@@@@@@@@",
    "@RRRRRR     BBBBB@",
    "@HHHHHH      BBBB@",
    "@RRRRRR     BBBBB@",
    "@RRRRR  P    BBBB@",
    "@RRRRR    P BBBBB@",
    "@HHHHH       BBBB@",
    "@RRRRR      BBBBB@",
    "@HHHHHHSSSSSSBBBB@",
    "@HHHHHHSSSSSSBBBB@",
    "@RRRRR   P P BBBB@",
    "@HHHHH   P  BBBBB@",
    "@RRRRRR    P BBBB@",
    "@HHHHHH P   BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHH    P  BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHHH  P P BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHH       BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHHH      BBBBB@",
    "@RRRRR       BBBB@",
    "@HHHH       BBBBB@",
    "@@@@@@@@@@@@@@@@@@",
]

CLEANUP_VIEW_SIZE = 7

MOVE_ACTIONS = [[0, -1], [0, 1], [-1, 0], [1, 0]]
FIRE_DIRECTIONS = [[-1, 0], [0, 1], [1, 0], [0, -1]]

thresholdDepletion = 0.4
thresholdRestoration = 0.0
wasteSpawnProbability = 0.5
appleRespawnProbability = 0.05


# multi-agent simplified cleanup domain
class CleanupFeatures(MultiAgentEnv):
    # use a0, a1 for agent names
    def __init__(self, num_agents=2,horizon=1000, image_obs=False,**kwargs):
        # map always reset as above
        self.map = MAP
        self.num_agents = num_agents
        self.horizon = horizon
        self.image_obs = image_obs
        self.timesteps = 0
        # make a list of the potential apple and waste spawn points
        self.initialize_arrays()  # call before compute probs due to potential_waste_area
        self.compute_probabilities()
        self.initialize_players()

        self.observation_space = gym.spaces.Box(low=np.array([0.0]*(12 + self.num_agents)),
                                                high=np.array([
                                                    len(self.map), len(self.map[0]), 4, len(self.map), len(self.map[0]), 4,
                                                    len(self.map), len(self.map[0]), len(self.map), len(self.map[0]),
                                                    len(self.apple_points) + 1, self.potential_waste_area + 1] + [np.inf] * self.num_agents))
        self.action_space = gym.spaces.Discrete(8)
        self.continuous_action_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(8,))

    def initialize_arrays(self):
        self.spawn_points = []
        self.wall_points = []
        self.apple_points = []
        self.waste_start_points = []
        self.waste_points = []
        self.river_points = []
        self.stream_points = []
        self.potential_waste_area = 0
        self.current_apple_spawn_prob = appleRespawnProbability
        self.current_waste_spawn_prob = wasteSpawnProbability

        for row in range(len(self.map)):
            for col in range(len(self.map[0])):
                if self.map[row][col] == "P":
                    self.spawn_points.append([row, col])
                elif self.map[row][col] == "@":
                    self.wall_points.append([row, col])
                if self.map[row][col] == "B":
                    self.apple_points.append([row, col])
                elif self.map[row][col] == "S":
                    self.stream_points.append([row, col])
                if self.map[row][col] == "H":
                    self.waste_start_points.append([row, col])
                    self.waste_points.append([row, col])
                    self.potential_waste_area += 1
                if self.map[row][col] == "R":
                    self.river_points.append([row, col])
                    self.waste_points.append([row, col])
                    self.potential_waste_area += 1

        self.current_waste_points = self.waste_start_points.copy()
        self.current_apple_points = []

    def initialize_players(self):
        # spawn agents
        random_idx = list(range(len(self.spawn_points)))
        random.shuffle(random_idx)  # permutes
        self.agent_pos = {'a'+str(i): self.spawn_points[random_idx[i]] for i in range(self.num_agents)}
        self.agent_orientation = {'a'+str(i): np.random.randint(low=0, high=4) for i in range(self.num_agents)}

    def spawn_apples_and_waste(self):
        # spawn apples
        for apple_point in self.apple_points:
            if apple_point not in self.current_apple_points and apple_point not in self.agent_pos.values():
                r = random.random()
                if r < self.current_apple_spawn_prob:
                    self.current_apple_points.append(apple_point.copy())

        # spawn at most one waste point
        for waste_point in self.waste_points:
            if waste_point not in self.current_waste_points:
                r = random.random()
                if r < self.current_waste_spawn_prob:
                    self.current_waste_points.append(waste_point.copy())
                    break  # added one, no need to continue

    def compute_closest_apples(self):
        closest_apples = {'a'+str(i): [0, 0] for i in range(self.num_agents)}  # sentinel value if there are no apples
        if self.current_apple_points:
            closest_apples = {}
            for key in self.agent_pos.keys():
                distances = np.sum(np.abs(np.array(self.current_apple_points) - np.array(self.agent_pos[key])), axis=1)
                min_dist_idx = np.argmin(distances)
                closest_apples[key] = self.current_apple_points[min_dist_idx]
        return closest_apples

    def compute_closest_wastes(self):
        closest_wastes = {'a'+str(i): [0, 0] for i in range(self.num_agents)}  # sentinel value if there is no waste
        if self.current_waste_points:
            for key in self.agent_pos.keys():
                distances = np.sum(np.abs(np.array(self.current_waste_points) - np.array(self.agent_pos[key])), axis=1)
                min_dist_idx = np.argmin(distances)
                closest_wastes[key] = self.current_waste_points[min_dist_idx]
        return closest_wastes

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
        infos = {key: {'cleaned_squares': 0} for key in acts.keys()}  # as info, pass the number of squares this agent cleaned

        # do stay first, so that it has highest priority
        for key in acts.keys():
            if acts[key] == 4:
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
                    move_squares[key] = tmp_move

        # update movements to reflect this
        self.agent_pos = {key: move_squares[key] if key in move_squares.keys() else self.agent_pos[key] for key in self.agent_pos.keys()}

        # consume apples at all moved to squares
        for key in move_squares.keys():
            if move_squares[key] in self.current_apple_points:
                rewards[key] += 1
                self.current_apple_points.remove(move_squares[key])

        # execute rotations
        for key in acts.keys():
            if acts[key] == 5:
                self.agent_orientation[key] = (self.agent_orientation[key] + 1) % 4
            if acts[key] == 6:
                self.agent_orientation[key] = (self.agent_orientation[key] - 1) % 4

        # manage firing actions (iterate by agent index so we can reference other agents)
        for i in range(self.num_agents):
            if acts['a'+str(i)] in [7, 8]:
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
                        elif acts['a'+str(i)] == 7:
                            # clean beam
                            if np.ndarray.tolist(current_beam) in self.current_waste_points:
                                self.current_waste_points.remove(np.ndarray.tolist(current_beam))  # clean the surface
                                infos['a'+str(i)]['cleaned_squares'] += 1  # increment cleaned_squares dict
                                self.metrics['dirt_cleaned'] +=1

            # if acts['a'+str(i)] == 7:
            #     rewards['a' + str(i)] -= 1.0  # costs 1 to punish opponent

        # spawn apples + waste
        self.compute_probabilities()
        self.spawn_apples_and_waste()

        # make observations, rewards, dones, infos to be passed
        compute_closest_apples = self.compute_closest_apples()
        compute_closest_wastes = self.compute_closest_wastes()
        compute_closest_pos = self.compute_closest_pos()  # necessary in multiagent case, get closest agent's position + orientation

        
        raw_rewards = 0 
        for k,v in rewards.items() :
            raw_rewards +=v 
            self.total_reward_dict[k].append(v)
        self.metrics['raw_env_rewards'] += raw_rewards
        
        observations = {key: np.array([self.agent_pos[key][0], self.agent_pos[key][1], self.agent_orientation[key],
                                        compute_closest_pos[key][0][0], compute_closest_pos[key][0][1], compute_closest_pos[key][1],
                                        compute_closest_apples[key][0], compute_closest_apples[key][1],
                                        compute_closest_wastes[key][0], compute_closest_wastes[key][1],
                                        len(self.current_apple_points), len(self.current_waste_points)]
                                      + [infos['a'+str(i)]['cleaned_squares'] if 'a'+str(i) in acts.keys() else 0 for i in range(self.num_agents)])
                        for key in acts.keys()}
        self.timesteps += 1
        dones = {'__all__': self.timesteps == self.horizon, 'a0': self.timesteps == self.horizon, 'a1': self.timesteps == self.horizon}  # rely on environment horizon: env never terminates

        if dones['__all__']:
            self.metrics['equality'] = self.compute_equality(deepcopy(self.total_reward_dict))
            self.metrics['sustainability'] = self.compute_sustainability(deepcopy(self.total_reward_dict))

        return observations, rewards, dones, infos

    def reset(self):
        # reset player locations, orientations, and load everything again from map
        self.initialize_arrays()
        self.initialize_players()

        # spawn apples and waste
        self.compute_probabilities()
        self.spawn_apples_and_waste()

        # compute closest apples and wastes
        compute_closest_apples = self.compute_closest_apples()
        compute_closest_wastes = self.compute_closest_wastes()
        compute_closest_pos = self.compute_closest_pos()  # necessary in multiagent case, get closest agent

        observations = {key: np.array([self.agent_pos[key][0], self.agent_pos[key][1],
                                       self.agent_orientation[key],
                                       compute_closest_pos[key][0][0], compute_closest_pos[key][0][1],
                                       compute_closest_pos[key][1],
                                       compute_closest_apples[key][0], compute_closest_apples[key][1],
                                       compute_closest_wastes[key][0], compute_closest_wastes[key][1],
                                       len(self.current_apple_points), len(self.current_waste_points)] + [0.0]*self.num_agents)
                        for key in ['a'+str(i) for i in range(self.num_agents)]}

        self.metrics = {'dirt_cleaned':0,'raw_env_rewards':0,'transfers':0}
        self.timesteps = 0
        self.total_reward_dict = {} 
        for key in ['a'+str(i) for i in range(self.num_agents)]:
            self.total_reward_dict[key] = []
        return observations

    def compute_probabilities(self):
        waste_density = 0
        if self.potential_waste_area > 0:
            waste_density = 1 - self.compute_permitted_area() / self.potential_waste_area
        if waste_density >= thresholdDepletion:
            self.current_apple_spawn_prob = 0
            self.current_waste_spawn_prob = 0
        else:
            self.current_waste_spawn_prob = wasteSpawnProbability
            if waste_density <= thresholdRestoration:
                self.current_apple_spawn_prob = appleRespawnProbability
            else:
                spawn_prob = (
                    1
                    - (waste_density - thresholdRestoration)
                    / (thresholdDepletion - thresholdRestoration)
                ) * appleRespawnProbability
                self.current_apple_spawn_prob = spawn_prob

    def compute_permitted_area(self):
        """How many cells can we spawn waste on?"""
        free_area = self.potential_waste_area - len(self.current_waste_points)
        return free_area

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

# if __name__ == '__main__':
#     env = CleanupFeatures()
#     env.reset()
#     print(env.current_waste_points)
#     env.spawn_apples_and_waste()
#     print(env.current_waste_points)
#     print(env.reset())
#     print(env.step({'a0': 2,  'a1': 4}))
#     print(env.current_waste_points)
#     print([2, 3] in env.current_waste_points)
#     print(env.step({'a0': 8, 'a1': 4}))
#     print(env.current_waste_points)
#     print([2, 3] in env.current_waste_points)
#     print(env.reset())
#     print([2, 3] in env.current_waste_points)
#     print(env.step({'a0': 5, 'a1': 4}))  # add rotation
#     print(env.step({'a0': 2,  'a1': 4}))
#     print(env.current_waste_points)
#     print([2, 3] in env.current_waste_points)
#     print(env.step({'a0': 8, 'a1': 4}))
#     print(env.current_waste_points)
#     print([2, 3] in env.current_waste_points)
