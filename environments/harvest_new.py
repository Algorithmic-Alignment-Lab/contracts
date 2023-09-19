import numpy as np
from numpy.random import rand

from environments.Agent import HarvestAgent
from environments.map_env import MapEnv 
import gym 
import scipy 
from copy import deepcopy 

HARVEST_MAP = [
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

# Add custom actions to the agent
_HARVEST_ACTIONS = {"FIRE": 5}  # length of firing range

SPAWN_PROB = [0, 0.005, 0.02, 0.05]

HARVEST_VIEW_SIZE = 7

from gym.spaces import Discrete


class DiscreteWithDType(Discrete):
    def __init__(self, n, dtype):
        assert n >= 0
        self.n = n
        # Skip Discrete __init__ on purpose, to avoid setting the wrong dtype
        super(Discrete, self).__init__((), dtype)

class HarvestEnv(MapEnv):
    def __init__(
        self,
        ascii_map=HARVEST_MAP,
        num_agents=1,
        disable_firing=True,
        image_obs = True, 
        return_agent_actions=False,
        use_collective_reward=False,
        inequity_averse_reward=False,
        alpha=0.0,
        beta=0.0,
        horizon=1000,
        one_hot_id=False,
        **kwargs
    ):
        super().__init__(
            ascii_map,
            _HARVEST_ACTIONS,
            HARVEST_VIEW_SIZE,
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
        self.apple_points = []
        self.one_hot_id = one_hot_id
        for row in range(self.base_map.shape[0]):
            for col in range(self.base_map.shape[1]):
                if self.base_map[row, col] == b"A":
                    self.apple_points.append([row, col])

        if self.disable_firing:
            self.action_space =  gym.spaces.Discrete(7)
            self.continuous_action_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(7,))
        else:
            self.action_space =  gym.spaces.Discrete(8)
            self.continuous_action_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(8,))

        global_img = gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(16, 38, 3),
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
            self.global_action_space = gym.spaces.MultiDiscrete([7] * self.num_agents)
        else:
            self.global_action_space = gym.spaces.MultiDiscrete([8] * self.num_agents)

        if not self.image_obs:
            self.observation_space = gym.spaces.Box(low=np.array([0.0]*(10 + 2 * self.num_agents)),
                                                    high=np.array([
                                                        len(self.base_map), len(self.base_map[0]), 4, len(self.base_map), len(self.base_map[0]), 4,
                                                        len(self.base_map), len(self.base_map[0]),
                                                        len(self.apple_points) + 1,
                                                        len(self.apple_points) + 1] + [1] * (2*self.num_agents)))
        else:
            img_space = gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=(2 * HARVEST_VIEW_SIZE + 1, 2 * HARVEST_VIEW_SIZE + 1, 3),
                    dtype=np.uint8,
                )
            if not self.one_hot_id:
                self.observation_space = gym.spaces.Dict({'image':img_space})
            else :
                self.observation_space = gym.spaces.Dict({'image':img_space, 'features':gym.spaces.Box(low=0, high=1, shape=(self.num_agents,))})
            
    def setup_agents(self):
        map_with_agents = self.get_map_with_agents()

        for i in range(self.num_agents):
            agent_id = "a" + str(i)
            spawn_point = self.spawn_point()
            rotation = self.spawn_rotation() 
            grid = map_with_agents
            agent = HarvestAgent(agent_id, spawn_point, rotation, grid, view_len=HARVEST_VIEW_SIZE)
            self.agents[agent_id] = agent

    def custom_reset(self):
        """Initialize the walls and the apples"""
        for apple_point in self.apple_points:
            self.single_update_map(apple_point[0], apple_point[1], b"A")
        self.computed_agent_pos = {key: self.agents[key].pos for key in self.agents.keys()} 
        self.compute_current_apples()
        self.compute_closest_apples()
        self.compute_closest_pos()
        self.total_close_apples = {key: self.count_apples_in_radius(5, self.agents[key].pos) for key in self.agents} 
        self.metrics = {'total_apples_eaten':0,'low_density_apples_eaten':0,'raw_env_rewards':0,'transfers':0}
        for i in range(self.num_agents):
            self.metrics['a{}-apples_consumed'.format(i)] = 0
            self.metrics['a{}-close_apples_consumed'.format(i)] = 0
        self.total_reward_dict = {key: [] for key in self.agents.keys()}

    def reset(self):
        o= super().reset() 
        if not self.image_obs:
            obs = {key: np.array([float(self.agents[key].pos[0]), float(self.agents[key].pos[1]),
                                           float(self.agents[key].int_orientation),
                                           float(self.closest_pos[key][0][0]),float(self.closest_pos[key][0][1]),
                                           float(self.closest_pos[key][1]),
                                           float(self.closest_apples[key][0]), float(self.closest_apples[key][1]),
                                           float(self.total_close_apples[key]),
                                           float(len(self.current_apple_points))] + [0.0] * (2 * self.num_agents))
                            for key in ['a' + str(i) for i in range(self.num_agents)]}
        else : 
            o_img = {key: o[key]['curr_obs']/255 for key in ['a' + str(i) for i in range(self.num_agents)]}
            if not self.one_hot_id:
                obs = obs = { key : {'image':o_img[key]} for key in o_img.keys()} 
            else : 
                obs ={ key: {'image':o_img[key], 'features':self.one_hot(key) } for key in o_img.keys()}
            
        return obs 

    def get_global_obs(self):
        return {'image':self.global_view()/255 } 
    
    def step(self,acts): 
        o,r,d,infos = super().step(acts) 
        for key in infos: 
            infos[key]['eaten_apples'] = 0
            infos[key]['eaten_close_apples'] = 0

        for key in self.agents:
            self.total_reward_dict[key].append(r[key])

        for key in self.agents:
            if self.agents[key].list_pos in self.current_apple_points:
                infos[key]['eaten_apples'] += 1
                self.metrics['{}-apples_consumed'.format(key)] += 1
                # self.single_update_map(move_squares[key][0], move_squares[key][1], b"0")
                if self.count_apples_in_radius(5, self.agents[key].list_pos) < 4:
                    infos[key]['eaten_close_apples'] += 1
                    self.metrics['low_density_apples_eaten'] +=1
                    self.metrics['{}-close_apples_consumed'.format(key)] += 1
                self.metrics['total_apples_eaten'] +=1 

        # Update metrics for raw environment rewards 
        raw_rewards = 0 
        for k,v in r.items() :
            raw_rewards +=v 
        self.metrics['raw_env_rewards'] += raw_rewards

        #Update all variables needed for feature obs and for tracking the metrics
        self.computed_agent_pos = {key: self.agents[key].pos for key in self.agents.keys()} 
        self.compute_current_apples()
        self.compute_closest_apples()
        self.compute_closest_pos() 
        self.total_close_apples = {key: self.count_apples_in_radius(5, self.agents[key].pos) for key in self.agents} 

        d = {'__all__': self.timesteps == self.horizon, 'a0': self.timesteps == self.horizon, 'a1': self.timesteps == self.horizon}
        feature_obs =  {key: np.array([float(self.agents[key].pos[0]), float(self.agents[key].pos[1]),
                                           float(self.agents[key].int_orientation),
                                           float(self.closest_pos[key][0][0]),float(self.closest_pos[key][0][1]),
                                           float(self.closest_pos[key][1]),
                                           float(self.closest_apples[key][0]), float(self.closest_apples[key][1]),
                                           float(self.total_close_apples[key]),
                                           float(len(self.current_apple_points))] + [0.0] * (2 * self.num_agents))
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
        agent.fire_beam(b"F")  
        updates = self.update_map_fire(
            agent.pos.tolist(),
            agent.get_orientation(),
            self.all_actions["FIRE"],
            fire_char=b"F",
        )
        return updates

    def custom_map_update(self):
        """See parent class"""
        # spawn the apples
        new_apples = self.spawn_apples()
        self.update_map(new_apples)
        
    def compute_current_apples(self):
        # Compute the current apple points in the map
        self.current_apple_points = []
        h,w = self.world_map.shape
        for i in range(h):
            for j in range(w):
                if self.world_map[i,j] == b"A":
                    self.current_apple_points.append([i,j])

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
    
    def spawn_apples(self):
        """Construct the apples spawned in this step.
        Returns
        -------
        new_apple_points: list of 2-d lists
            a list containing lists indicating the spawn positions of new apples
        """

        new_apple_points = []
        agent_positions = self.agent_pos
        random_numbers = rand(len(self.apple_points))
        r = 0
        for i in range(len(self.apple_points)):
            row, col = self.apple_points[i]
            # apples can't spawn where agents are standing or where an apple already is
            if [row, col] not in agent_positions and self.world_map[row, col] != b"A":
                num_apples = 0
                for j in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                    for k in range(-APPLE_RADIUS, APPLE_RADIUS + 1):
                        if j ** 2 + k ** 2 <= APPLE_RADIUS:
                            x, y = self.apple_points[i]
                            if (
                                0 <= x + j < self.world_map.shape[0]
                                and self.world_map.shape[1] > y + k >= 0
                            ):
                                if self.world_map[x + j, y + k] == b"A":
                                    num_apples += 1

                spawn_prob = SPAWN_PROB[min(num_apples, 3)]
                rand_num = random_numbers[r]
                r += 1
                if rand_num < spawn_prob:
                    new_apple_points.append((row, col, b"A"))
        return new_apple_points

    def count_apples(self, window):
        # compute how many apples are in window
        unique, counts = np.unique(window, return_counts=True)
        counts_dict = dict(zip(unique, counts))
        num_apples = counts_dict.get(b"A", 0)
        return num_apples
    
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
    from harvest_features import HarvestFeatures
    import imageio 
    from env_utils import make_video_from_rgb_imgs
    from PIL import Image 
    env = HarvestEnv(num_agents=2,image_obs=1,horizon=100)
    env = HarvestFeatures(num_agents=2,horizon=100)
    o1 =  env.reset()
    #o2 = env2.reset() 
    print('o1',o1['a0']) 
    #print('o2',o2) 
    #print(env.action_space.n)
    for k in range(1):
        env.reset() 
        imgs = [] 
        for i in range(100):
            a1,a2 = env.action_space.sample(),env.action_space.sample() 
            print(a1)
            actions = {'a0':a1,'a1':a2}
        # acts = {key: int(np.argmax(np.random.multinomial(1, scipy.special.softmax(actions[key].astype(np.float64)))))
                    #for key in actions.keys()} 
            o1,r1,d1,i1 = env.step(actions)
        # o2,r2,d2,i2 = env2.step(acts)
            #print('o2',o2)
            #img = env.full_map_to_colors() 
            #imgs.append(img)
        print(env.metrics)
        height, width, _ = imgs[0].shape
        # Upscale to be more legible
        width *= 20
        height *= 20
        imgs = [image.astype('uint8') for image in imgs]
        make_video_from_rgb_imgs(imgs, '.',resize=(width,height))
        
    # plt.imshow(img)
    # plt.show() 