import numpy as np
from numpy.random import rand

from environments.Agent import CleanupAgent
from environments.map_env import MapEnv 
import gym 
import scipy 
from copy import deepcopy

CLEANUP_MAP = [
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

# Add custom actions to the agent
_CLEANUP_ACTIONS = {"FIRE": 5, "CLEAN": 5}  # length of firing beam, length of cleanup beam

# Custom colour dictionary
CLEANUP_COLORS = {
    b"C": np.array([100, 255, 255], dtype=np.uint8),  # Cyan cleaning beam
    b"S": np.array([113, 75, 24], dtype=np.uint8),  # Light grey-blue stream cell
    b"H": np.array([99, 156, 194], dtype=np.uint8),  # Brown waste cells
    b"R": np.array([113, 75, 24], dtype=np.uint8),  # Light grey-blue river cell
}

SPAWN_PROB = [0, 0.005, 0.02, 0.05]

CLEANUP_VIEW_SIZE = 7

thresholdDepletion = 0.4
thresholdRestoration = 0.0
wasteSpawnProbability = 0.5
appleRespawnProbability = 0.05


class CleanupEnv(MapEnv):
    def __init__(
        self,
        ascii_map=CLEANUP_MAP,
        num_agents=1,
        disable_firing=True,
        image_obs = True, 
        return_agent_actions=False,
        use_collective_reward=False,
        inequity_averse_reward=False,
        alpha=0.0,
        beta=0.0,
        horizon=1000,
        one_hot_id = False,
        **kwargs
    ):
        super().__init__(
            ascii_map,
            _CLEANUP_ACTIONS,
            CLEANUP_VIEW_SIZE,
            num_agents,
            return_agent_actions=return_agent_actions,
            use_collective_reward=use_collective_reward,
            inequity_averse_reward=inequity_averse_reward,
            alpha=alpha,
            beta=beta,
            image_obs=image_obs,
            horizon=horizon 
        )
        self.disable_firing = disable_firing

        if self.disable_firing: # Disable punishment beam 
            self.action_space =  gym.spaces.Discrete(8)
            self.continuous_action_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(8,))
        else:
            self.action_space =  gym.spaces.Discrete(9) 
            self.continuous_action_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(9,))

        # compute potential waste area
        unique, counts = np.unique(self.base_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        self.potential_waste_area = counts_dict.get(b"H", 0) + counts_dict.get(b"R", 0)
        self.current_apple_spawn_prob = appleRespawnProbability
        self.current_waste_spawn_prob = wasteSpawnProbability
        self.one_hot_id = one_hot_id
        self.compute_probabilities()

        # make a list of the potential apple and waste spawn points
        self.apple_points = []
        self.waste_start_points = []
        self.waste_points = []
        self.river_points = []
        self.stream_points = []
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"P":
                    self.spawn_points.append([row, col])
                elif self.base_map[row, col] == b"B":
                    self.apple_points.append([row, col])
                elif self.base_map[row, col] == b"S":
                    self.stream_points.append([row, col])
                if self.base_map[row, col] == b"H":
                    self.waste_start_points.append([row, col])
                if self.base_map[row, col] in [b"H", b"R"]:
                    self.waste_points.append([row, col])
                if self.base_map[row, col] == b"R":
                    self.river_points.append([row, col])
        
        self.current_waste_points = deepcopy(self.waste_start_points)
        self.current_apple_points = []
        self.color_map.update(CLEANUP_COLORS)

        global_img = gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(25, 18, 3),
                    dtype=np.uint8,
                )
        self.global_observation_space = gym.spaces.Dict({'image': global_img})  

        concatenated_img = gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(15, 15, 3*self.num_agents),
                    dtype=np.uint8,
                )
        self.concatenated_observation_space = gym.spaces.Dict({'image': concatenated_img})

        if self.disable_firing:
            self.global_action_space = gym.spaces.MultiDiscrete([8] * self.num_agents)
        else:
            self.global_action_space = gym.spaces.MultiDiscrete([9] * self.num_agents)

        if not self.image_obs:
            self.observation_space = gym.spaces.Box(low=np.array([0.0]*(12 + self.num_agents )),
                                                high=np.array([
                                                    len(self.base_map), len(self.base_map[0]), 4, len(self.base_map), len(self.base_map[0]), 4,
                                                    len(self.base_map), len(self.base_map[0]), len(self.base_map), len(self.base_map[0]),
                                                    len(self.apple_points) + 1, self.potential_waste_area + 1] + [np.inf] * self.num_agents))
            
        else: 
            img_space = gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(2 * CLEANUP_VIEW_SIZE + 1, 2 * CLEANUP_VIEW_SIZE + 1, 3),
                    dtype=np.uint8,
                )
            if not self.one_hot_id:
                self.observation_space = gym.spaces.Dict({'image':img_space})
            else :
                self.observation_space = gym.spaces.Dict({'image':img_space, 'features':gym.spaces.Box(low=0, high=1, shape=(self.num_agents,))})

    def custom_reset(self):
        """Initialize the walls and the waste"""
        for waste_start_point in self.waste_start_points:
            self.single_update_map(waste_start_point[0], waste_start_point[1], b"H")
        for river_point in self.river_points:
            self.single_update_map(river_point[0], river_point[1], b"R")
        for stream_point in self.stream_points:
            self.single_update_map(stream_point[0], stream_point[1], b"S")
        self.computed_agent_pos = {key: self.agents[key].pos for key in self.agents.keys()} 
        self.compute_probabilities()
        self.compute_current_apples()
        self.compute_current_wastes()
        self.compute_closest_apples()
        self.compute_closest_wastes()
        self.compute_closest_pos()
        self.metrics = {'total_apples_eaten':0,'raw_env_rewards':0,'transfers':0,'dirt_cleaned':0}
        for i in range(self.num_agents):
            self.metrics['a{}-waste_cleaned'.format(i)] = 0
        self.total_reward_dict = {key: [] for key in self.agents.keys()}

    def reset(self):
        o= super().reset() 
        if not self.image_obs:
            obs = {key: np.array([float(self.agents[key].pos[0]), float(self.agents[key].pos[1]),
                                           float(self.agents[key].int_orientation),
                                           float(self.closest_pos[key][0][0]),float(self.closest_pos[key][0][1]),
                                           float(self.closest_pos[key][1]),
                                           float(self.closest_apples[key][0]), float(self.closest_apples[key][1]),
                                           float(self.closest_wastes[key][0]), float(self.closest_wastes[key][1]),
                                           len(self.current_apple_points), len(self.current_waste_points)]
                                      + [0.0]*self.num_agents)
                            for key in ['a' + str(i) for i in range(self.num_agents)]}
        else : 
            o_img = {key: o[key]['curr_obs']/255 for key in ['a' + str(i) for i in range(self.num_agents)]}
            if not self.one_hot_id:
                obs = { key : {'image':o_img[key]} for key in o_img.keys()} 
            else : 
                obs ={ key: {'image':o_img[key], 'features':self.one_hot(key) } for key in o_img.keys()}
        return obs
    
    def step(self,acts): 
        o,r,d,infos = super().step(acts) 
        for key in infos: 
            infos[key]['eaten_apples'] = 0
            infos[key]['cleaned_squares'] = deepcopy(self.agents[key].cleaned_squares)
            self.metrics['dirt_cleaned'] += deepcopy(self.agents[key].cleaned_squares)
            self.metrics['{}-waste_cleaned'.format(key)] += deepcopy(self.agents[key].cleaned_squares)
            self.agents[key].cleaned_squares = 0 

        for key in self.agents:
            if self.agents[key].list_pos in self.current_apple_points:
                infos[key]['eaten_apples'] += 1
                self.metrics['total_apples_eaten'] +=1 

        for key in self.agents:
            self.total_reward_dict[key].append(r[key])
        
        # Update metrics for raw environment rewards 
        raw_rewards = 0 
        for k,v in r.items() :
            raw_rewards +=v 
        self.metrics['raw_env_rewards'] += raw_rewards

        #Update all variables needed for feature obs and for tracking the metrics
        self.computed_agent_pos = {key: self.agents[key].pos for key in self.agents.keys()} 
        self.compute_current_apples()
        self.compute_current_wastes()
        self.compute_closest_apples()
        self.compute_closest_pos() 
        self.compute_closest_wastes()

        d = {'__all__': self.timesteps == self.horizon, 'a0': self.timesteps == self.horizon, 'a1': self.timesteps == self.horizon}
        feature_obs = {key: np.array([float(self.agents[key].pos[0]), float(self.agents[key].pos[1]),
                                           float(self.agents[key].int_orientation),
                                           float(self.closest_pos[key][0][0]),float(self.closest_pos[key][0][1]),
                                           float(self.closest_pos[key][1]),
                                           float(self.closest_apples[key][0]), float(self.closest_apples[key][1]),
                                           float(self.closest_wastes[key][0]), float(self.closest_wastes[key][1]),
                                           len(self.current_apple_points), len(self.current_waste_points)]
                                      + [infos['a'+str(i)]['cleaned_squares'] if 'a'+str(i) in acts.keys() else 0 for i in range(self.num_agents)])
                            for key in ['a' + str(i) for i in range(self.num_agents)]}
        for key in self.agents:
            infos[key]['feature_obs'] = feature_obs[key] 
        # feature-engineered observations
        if not self.image_obs:
            obs = feature_obs 
        else :
            o_img = {key: o[key]['curr_obs']/255 for key in ['a' + str(i) for i in range(self.num_agents)]}
            if not self.one_hot_id:
                obs = { key : {'image':o_img[key]} for key in o_img.keys()} 
            else : 
                obs ={ key: {'image':o_img[key], 'features':self.one_hot(key) } for key in o_img.keys()}

        if d['__all__']:
            self.metrics['equality'] = self.compute_equality(deepcopy(self.total_reward_dict))
            self.metrics['sustainability'] = self.compute_sustainability(deepcopy(self.total_reward_dict))
        return obs,r,d,infos 

    def custom_action(self, agent, action):
        """Allows agents to take actions that are not move or turn"""
        updates = []
        if action == "FIRE":
            agent.fire_beam(b"F")
            updates = self.update_map_fire(
                agent.pos.tolist(),
                agent.get_orientation(),
                self.all_actions["FIRE"],
                fire_char=b"F",
            )
        elif action == "CLEAN":
            agent.fire_beam(b"C")
            updates = self.update_map_fire(
                agent.pos.tolist(),
                agent.get_orientation(),
                self.all_actions["FIRE"],
                fire_char=b"C",
                cell_types=[b"H"],
                update_char=[b"R"],
                blocking_cells=[b"H"],
            )
            agent.cleaned_squares = len(updates) 
        return updates

    def custom_map_update(self):
        """ "Update the probabilities and then spawn"""
        self.compute_probabilities()
        self.update_map(self.spawn_apples_and_waste())

    def get_global_obs(self):
        return {'image':self.global_view()/255 } 

    def setup_agents(self):
        """Constructs all the agents in self.agent"""
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = "a" + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation()
            # grid = util.return_view(map_with_agents, spawn_point,
            #                         CLEANUP_VIEW_SIZE, CLEANUP_VIEW_SIZE)
            # agent = CleanupAgent(agent_id, spawn_point, rotation, grid)
            agent = CleanupAgent(
                agent_id,
                spawn_point,
                rotation,
                map_with_agents,
                view_len=CLEANUP_VIEW_SIZE,
            )
            self.agents[agent_id] = agent

    def spawn_apples_and_waste(self):
        spawn_points = []
        # spawn apples, multiple can spawn per step
        agent_positions = self.agent_pos
        random_numbers = rand(len(self.apple_points) + len(self.waste_points))
        r = 0
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # don't spawn apples where agents already are
            if [row, col] not in agent_positions and self.world_map[row, col] != b"A":
                rand_num = random_numbers[r]
                r += 1
                if rand_num < self.current_apple_spawn_prob:
                    spawn_points.append((row, col, b"A"))

        # spawn one waste point, only one can spawn per step
        if not np.isclose(self.current_waste_spawn_prob, 0):
            np.random.shuffle(self.waste_points)
            for i in range(len(self.waste_points)):
                row, col = self.waste_points[i]
                # don't spawn waste where it already is
                if self.world_map[row, col] != b"H":
                    rand_num = random_numbers[r]
                    r += 1
                    if rand_num < self.current_waste_spawn_prob:
                        spawn_points.append((row, col, b"H"))
                        break
        return spawn_points

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
        unique, counts = np.unique(self.world_map, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        current_area = counts_dict.get(b"H", 0)
        free_area = self.potential_waste_area - current_area
        return free_area
    
    def compute_current_apples(self):
        # Compute the current apple points in the map
        self.current_apple_points = []
        h,w = self.world_map.shape
        for i in range(h):
            for j in range(w):
                if self.world_map[i,j] == b"A":
                    self.current_apple_points.append([i,j])

    def compute_current_wastes(self):
        # Compute the current apple points in the map
        self.current_waste_points = []
        h,w = self.world_map.shape
        for i in range(h):
            for j in range(w):
                if self.world_map[i,j] == b"H":
                    self.current_waste_points.append([i,j])
    
    def compute_closest_apples(self):
        self.closest_apples = {'a'+str(i): [0, 0] for i in range(self.num_agents)}  # sentinel value if there are no apples
        if self.current_apple_points:
            self.closest_apples = {}
            for key in self.computed_agent_pos.keys():
                distances = np.sum(np.abs(np.array(self.current_apple_points) - np.array(self.computed_agent_pos[key])), axis=1)
                min_dist_idx = np.argmin(distances)
                self.closest_apples[key] = self.current_apple_points[min_dist_idx]
    
    def compute_closest_pos(self):
        self.closest_pos = {'a'+str(i): [0, 0] for i in range(self.num_agents)}  # sentinel value if there is no waste
        for key in self.computed_agent_pos.keys():
            distances = np.sum(np.abs(
                np.array([[np.inf, np.inf] if k == key else self.computed_agent_pos[key] for k in self.computed_agent_pos.keys()]) - np.array(self.computed_agent_pos[key])
            ), axis=1)
            min_dist_idx = np.argmin(distances)
            self.closest_pos[key] = (self.computed_agent_pos['a'+str(min_dist_idx)], self.agents['a'+str(min_dist_idx)].int_orientation)

    def compute_closest_wastes(self):
        self.closest_wastes = {'a'+str(i): [0, 0] for i in range(self.num_agents)}  # sentinel value if there is no waste
        if self.current_waste_points:
            for key in self.computed_agent_pos.keys():
                distances = np.sum(np.abs(np.array(self.current_waste_points) - np.array(self.computed_agent_pos[key])), axis=1)
                min_dist_idx = np.argmin(distances)
                self.closest_wastes[key] = self.current_waste_points[min_dist_idx]

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
   
if __name__ =='__main__':
    import time 
    import matplotlib.pyplot as plt
    from cleanup_features import CleanupFeatures
    import imageio 
    from env_utils import make_video_from_rgb_imgs
    from PIL import Image 
    env = CleanupEnv(num_agents=2,image_obs=1,horizon=100,one_hot_id=1)
    #env = CleanupFeatures(num_agents=2,horizon=100)
    o1 =  env.reset()
    #o2 = env2.reset() 
    print(o1) 
    #print('o2',o2) 
    #print(env.action_space.n)
    input()
    for k in range(1):
        env.reset() 
        imgs = [] 
        for i in range(100):
            a1,a2 = env.action_space.sample(),env.action_space.sample() 
            actions = {'a0':a1,'a1':a2}
        # acts = {key: int(np.argmax(np.random.multinomial(1, scipy.special.softmax(actions[key].astype(np.float64)))))
                    #for key in actions.keys()} 
            o1,r1,d1,i1 = env.step(actions)
        # o2,r2,d2,i2 = env2.step(acts)
            #print('o2',o2)
          #  img = env.full_map_to_colors() 
           # imgs.append(img)
        print(env.metrics)
        height, width, _ = imgs[0].shape
        # Upscale to be more legible
        width *= 20
        height *= 20
        imgs = [image.astype('uint8') for image in imgs]
        make_video_from_rgb_imgs(imgs, '.',resize=(width,height))