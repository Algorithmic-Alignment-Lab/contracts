import wandb 
import ray 
from ray.tune.logger import LoggerCallback
from ray.rllib.algorithms.callbacks import DefaultCallbacks
import os 
import time 
import numpy as np 
import copy 

class CustomLoggerCallback(LoggerCallback):

    """
    This class defines a custom logger interface used for logging and visualizing experiment metrics during reinforcement learning training. The logger is designed to work with the WandB (Weights and Biases) platform for easy experiment tracking and visualization.

    Classes:
    - `CustomLoggerCallback`: A custom logger interface for tracking and logging experiment metrics.

    Note: The `CustomLoggerCallback` class enhances the standard logger provided by RLlib with custom functionality for more detailed and structured logging. It is designed to work seamlessly with WandB to facilitate experiment tracking and visualization.
    """

    def __init__(self,config=None):
        self.config = config 
        self.curr_step = 0 
        self.stage_1_dur = 0
        self.use_wandb = config['wandb'] 
        self.horizon = config['horizon']
        self.stage = 'stage_1'
        if self.use_wandb : 
            self.wandb = self.setup_wandb() 

    def setup_wandb(self) : 
        global run_wandb
        run_wandb = None 
        name = self.config['parent_tag'] + self.config['exp_name'] +  time.strftime("_%Y-%m-%d") 
        run_wandb = wandb.init(project='Contracting',config=self.config,tags=[self.config['base_env_tag']],name=name,settings=wandb.Settings(start_method="fork",_disable_stats=True))  
        return run_wandb 

    def set_stage(self,stage) : 
        self.stage = 'stage_{}'.format(stage) 
        self.stage_1_dur = copy.deepcopy(self.curr_step) 

    def simple_log(self,info_dict): 
        if self.use_wandb : 
            for k in info_dict.keys() : 
                key_name = self.stage + '/' + k 
                self.wandb.log({key_name:info_dict[k]},step=self.curr_step)
            wandb.log(info_dict,step=self.curr_step)
            self.curr_step += self.horizon 
        
    def log_trial_result(self,it,t,r): 
        if self.use_wandb : 
            self.log = {}  
            eps = r['episodes_this_iter'] 
            if self.stage== 'stage_2':
                self.curr_step += self.horizon * int(eps* np.mean(r['hist_stats']['episode_lengths'][-eps:]))
            else :
                self.curr_step += int(eps* np.mean(r['hist_stats']['episode_lengths'][-eps:]))
            self.log['episode reward'] = r['episode_reward_mean'] 
             
            # log learners and their individual stats 
            keys =[] 
            for k,v in r['info']['learner'].items() : 
                keys.append(k) 
            for k in keys :
                if k not in ['a4','a5','a6','a7','a8']:
                    for k2,v in r['info']['learner'][k]['learner_stats'].items() : 
                        if type(v) != list : 
                            self.log[k+'/'+k2] = v
                        else : 
                            if len(v)!=0 :
                                self.log[k+'/'+k2] = np.mean(v)

            for k,v in r['hist_stats'].items() : 
                if k!='episode_reward' : 
                    if type(v) == list : 
                        if len(v)!=0 : 
                            self.log[k] = float(np.mean(v)) 
                    else : 
                        self.log[k] = v  

            for k,v in r['custom_metrics'].items() : 
                if '_min' not in k and '_max' not in k:
                    self.log['custom_metrics/'+k] = v 
            # append stage to log names 
            self.wandb_log = {} 
            for k,v in self.log.items() : 
                self.wandb_log[self.stage+'/'+k] = v  
         
            if self.stage == 'stage_2' :
                self.wandb_log['stage_2/env_timestep'] = self.curr_step - self.stage_1_dur
                self.wandb.log(self.wandb_log,step =self.curr_step)
            else :
                self.wandb.log(self.wandb_log,step =self.curr_step)

class MetricsCallback(DefaultCallbacks,LoggerCallback):
    """
    A custom callback class for collecting and logging metrics during RLlib training.

    This callback is used to collect custom metrics from the environment and log them
    at the end of each episode. It can be added to RLlib trainers to enable the tracking
    of additional metrics beyond the default ones.

    Note:
    - It collects metrics from the environment, such as those provided by custom Gym environments, and logs them as custom_metrics within the episode object.
    - The collected metrics are logged alongside standard RLlib metrics when using the WandB logger as well as to a csv file.
    """
    
    def __init__(self) :
        super().__init__()   

    def on_episode_start(self, *, worker, base_env,
                         policies, episode,
                         **kwargs): 
        
        episode.custom_metrics = {}   

    def setup(self) :
        pass

    def on_episode_step(self, *, worker, base_env,
                         policies, episode,
                         **kwargs): 
        
        pass

    def on_episode_end(self, *, worker, base_env,
                         policies, episode,
                         **kwargs):

        m1,m2 = None , None 
        try :
            m1 = base_env._unwrapped_env.base_env.metrics
        except :  
            pass
        try :
            m2 = base_env._unwrapped_env.metrics
        except :  
           pass 
        assert (m1 or m2 is not None)

        if m1 is not None :
            metrics = m1 
        else :
            metrics = m2 

        if m1 is not None and m2 is not None :
            metrics.update(m2) 

        if len(metrics) >0 :
            for k,v in metrics.items() : 
                    episode.custom_metrics[k] = v 
        
        