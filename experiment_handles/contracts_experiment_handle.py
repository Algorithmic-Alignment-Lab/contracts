"""
This Python file contains the implementation of the `ContractsExperimentHandle` class, which is an abstract experiment handler for running all MARL experiments in this repository. It leverages the Ray RLlib library for reinforcement learning experiments and calls functions for argument parsing, training, rendering, and solver stages.

Classes:
- `ContractsExperimentHandle`: An abstract experiment handler class for running contract negotiation experiments.
"""

# Rest of your code...


from experiment_handles.experiment_handle import AbstractExperimentHandle
from utils.ray_config_utils import parse_arguments_dict
from run_training import run_experiment
import ray
import os
from  run_render import run_rendering
from run_solver import run_solver
import copy 
# an abstract experiment handle class
class ContractsExperimentHandle(AbstractExperimentHandle):

    def __init__(self, config_dict_list):

        self.config_dict_list = config_dict_list
    
    # logic for argument parsing
    def hook_at_start(self,hook):
        if hook:
            ray.init()  # done once, at the start of the loop

    def hook_at_end(self,hook):
        if hook:
            ray.shutdown()  # done once all experiments are done

    def run_exp(self,hook_start=True,hook_end=True):

        # set up experiment directories, check if experiment has been run
        self.hook_at_start(hook_start)
        i,config_dict = 0, self.config_dict_list[0]
        # Get experiment name
        self.experiment_name = self.get_exp_name(i) 
        # Parse arguments from utils/ray_config_utils.py 
        exp_params = self.argument_parsing(config_dict)
        print('Main Experiment started for ',self.experiment_name)
        # Run training by calling RLlib
        checkpoints = self.main_exp(exp_params) 

        # If using solver for second-stage contracting, run solver with frozen agent weights 
        if config_dict.get('solver') and not config_dict.get('separate'): 
            solver_params = copy.deepcopy(exp_params)
            self.solver_exp(solver_params,checkpoints['weight_directories'],checkpoints['logger'])
        
        # If rendering, generate renders and store in gifs directory
        if config_dict.get('num_renders') :
            render_params = copy.deepcopy(exp_params)
            render_params['second_stage_only'] = checkpoints['weight_directories']
            os.mkdir('gifs/'+self.directory_name)  
            self.gif_store_path = 'gifs/'+self.directory_name 
            self.gif_renders(render_params,config_dict['num_renders'])

        print('Main Experiment ended for ',self.experiment_name)
        self.hook_at_end(hook_end)
        print('Hook ended for ',self.experiment_name)

    def argument_parsing(self, config_dict):
        arguments = parse_arguments_dict(self.experiment_name, config_dict)
        return arguments
    
    def gif_renders(self,exp_params,num_renders) :
        return run_rendering(exp_params,self.gif_store_path,num_renders)

    def solver_exp(self,exp_params,checkpoints,logger) :
        return run_solver(exp_params,checkpoints,logger)
    
    def get_exp_name(self,index) : 
        initial_name = self.config_dict_list[index]['experiment_name']
        new_name = initial_name 
        i= 0 
        while os.path.exists('gifs/'+new_name): 
                new_name = initial_name 
                new_name += '_{}'.format(i) 
                i+=1
        
        arg_dict = self.config_dict_list[index]
        if arg_dict.get("joint"):
            new_name  += '_' + str(arg_dict.get("num_agents")) + 'agents' + '_joint' + '_{}'.format(index) 
        elif arg_dict.get("separate"):
            new_name += '_' + str(arg_dict.get("num_agents")) + 'agents' + '_separate' + '_{}'.format(index)  
        elif arg_dict.get("combined"):
            new_name += '_' + str(arg_dict.get("num_agents")) + 'agents' + '_combined'+ '_{}'.format(index) 
        elif arg_dict.get("first_stage_only"):
            new_name += '_' + str(arg_dict.get("num_agents")) + 'agents' + '_subgame'+ '_{}'.format(index) 
        elif arg_dict.get("second_stage_only"):
            new_name += '_' + str(arg_dict.get("num_agents")) + 'agents' + '_negotiate'+ '_{}'.format(index) 
        else:
            # raise NotImplementedError("Unclear Run Mode") TODO make this ONLY per-stage
            new_name += '_' + str(arg_dict.get("num_agents")) + 'agents'+ '_{}'.format(index) 

        self.config_dict_list[index]['experiment_name'] = new_name  
        
        if self.config_dict_list[index]['parent_name'] is not None :
            self.config_dict_list[index]['directory_name'] = self.config_dict_list[index]['parent_name'] +'/' + new_name
        else :
            self.config_dict_list[index]['directory_name'] = new_name 
        
        self.directory_name = self.config_dict_list[index]['directory_name'] 
        return new_name 

    # main execution loop, running experiments
    def main_exp(self, exp_param):
        return run_experiment(exp_param)


