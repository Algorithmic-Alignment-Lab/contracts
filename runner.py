"""
This Python file defines a main script for running experiments with configurations specified in a JSON file. The script supports both sequential and parallel execution of experiments using multiprocessing. 

Functions:
- `main_run(config, args, hook_start=True, hook_end=True)`: Runs the main experiment with the given configuration.
- `get_args()`: Parses command-line arguments and returns them as an argparse Namespace.
- `sequential_run(num_jobs, args, config_dict_list)`: Runs experiments sequentially with the specified number of jobs and configuration dictionaries.
- `mp_run(num_jobs, total_jobs, args, config_dict_list)`: Runs experiments in parallel using multiprocessing with the specified number of jobs, total jobs, and configuration dictionaries.

The script reads configuration settings from a JSON file and generates combinations of parameters for experiments. It also handles options for using GPUs and multiprocessing.
"""

import multiprocessing,time, os   
import argparse 
import json 
from experiment_handles.contracts_experiment_handle import ContractsExperimentHandle
import copy 
import itertools 

def main_run(config,args,hook_start=True,hook_end=True) :  
    experiment_handle = ContractsExperimentHandle(config)
    experiment_handle.run_exp(hook_start,hook_end)
    
def get_args() :
    parser= argparse.ArgumentParser() 
    parser.add_argument('--name' ,type = str , help= 'Name for campaign to run') 
    parser.add_argument('--mp',action='store_true',default=False,help='Whether to use multiprocesssing')
    parser.add_argument('--workers', type = int , default=1, help = 'Only required if multiprocessing is true: Total number of workers, must be larger than the max worker requirement in the config dict') 
    parser.add_argument('--w_per_job', type = int , default=1 , help = 'Only required if multiprocessing is true: Total number of workers, must be larger than the max worker requirement in the config dict') 
    parser.add_argument('--gpu',action='store_true',default=False,help='Whether to use gpu')
    parser.add_argument('--config_path' ,type = str , default=None , help= 'Name for campaign to run') 
    parser.add_argument('--results_dir',type =str, default = 'results/') 
    parser.add_argument('--load_second_stage',type=str,default=None)
    parser.add_argument('--seeds',type=int,default=1)
    
    args = parser.parse_args() 
    return args

def sequential_run(num_jobs,args,config_dict_list) : 
    for i,c in enumerate(config_dict_list): 
        print('Running Job ',i)
        if i ==0:
            main_run([config_dict_list[i]],args,hook_start=1,hook_end=0) 
        elif i == num_jobs-1 :
            main_run([config_dict_list[i]],args,hook_start=0,hook_end=1)
        else :
            main_run([config_dict_list[i]],args,hook_start=0,hook_end=0)
        print('Finished Job ',i) 
        time.sleep(20) 

def mp_run(num_jobs,total_jobs,args,config_dict_list) :   
    running_jobs = 0 
    index_queue = 0 
    running_processes = [] 
    prev_alive = [] 
    total_done = 0 

    while total_done!= total_jobs : 
        if running_jobs < num_jobs and index_queue <total_jobs : 
            new_process = multiprocessing.Process(target=main_run,args=([config_dict_list[index_queue]],args)) 
            new_process.start() 
            running_processes.append(new_process)  
            running_jobs +=1 
            print('Started Job ',index_queue) 
            index_queue +=1 
        
        running_jobs = 0 
        alive =[] 
        for p in running_processes :
            a = p.is_alive()
            if a : 
                running_jobs+=1  
            alive.append(a)

        for i,_ in enumerate(prev_alive) : 
            if prev_alive[i] != alive[i] : 
                print('Job', i , 'terminated') 
                total_done +=1 
                print('Number of Pending jobs:', total_jobs -total_done)

        prev_alive = alive 
        time.sleep(30) 
if __name__ =='__main__' : 
    # Get arguments 
    args = get_args() 

    assert args.name is not None 

    # Load the config dict list 
    with open(args.config_path, 'r') as f:
            config_dict_list = json.load(f)


    # The first element of the list is the global parameters which all configs will have (this will not overwrite any parameters in the individual configs)
    # The second element of the list is the parameters which will be iterated over (for example hyperparameter tuning) 
    # This code adds the shared parameters to the individual parameters as well as iterates over all combination of params provided 
    global_params = config_dict_list[0] 
    iter_params = config_dict_list[1]
    config_dict_list = config_dict_list[2:]
    for config in config_dict_list:
            for param in global_params:
                if param not in config:
                    config[param] = global_params[param]

    # Create a list of parameter lists
    params_list = [v for v in iter_params.values()]
    # Use itertools.product to get all possible combinations
    combinations = list(itertools.product(*params_list))
    # Print the combinations
    comb_dict =[]
    for i,c in enumerate(combinations):
        c_dict = {} 
        for i,k in enumerate(iter_params.keys()):
            c_dict[k] = c[i] 
        comb_dict.append(c_dict)

    new_config_dict_list = []
    for c in comb_dict:
        for config in config_dict_list:
            new_config_dict_list.append({**config,**c}) 

    config_dict_list = new_config_dict_list

    for config in config_dict_list:
        if 'experiment_name' not in config:
            config['experiment_name'] = config['environment'] + '-' + str(config['num_agents'])+'agents'

    # Generating new identical configs with the number of seeds specified
    # seed multiplier 
    sm = 73907 
    new_config_list = [] 
    for c in config_dict_list :
        for i in range(args.seeds) : 
            c['seed'] = int((i+1)*sm) 
            new_config_list.append(copy.deepcopy(c)) 

    # Do we want to load first stage checkpoint paths from a past experiment, note that for this to work in such an automated manner,
    # the experiment name and the seeding for the past experiment should be identical. I follow a generic naming convention where experiment names
    # are '<env_name>-<num_agents>agents'. If this is not the case, then in the config file, set 'full_path' = True and provide the full first_stage path
    # in the config file itself. This would directly retrieve that and the following loop will not be executed in that case. However, that might be time
    # consuming because a different path would have to be provided for each individual path/experiment

    for i,c in enumerate(new_config_list): 
        if c.get('second_stage_only') or args.load_second_stage is not None: 
            if not c.get('full_path') :
                if args.load_second_stage is not None :
                    c['second_stage_only'] = args.load_second_stage
                checkpoint = c['second_stage_only'] 
                new_checkpoint_file = 'experiment_paths/' + str(checkpoint) + '/'+  c['experiment_name'] + str('s{}'.format(str(int(c['seed']/sm)))) +  '_' +str(i)+ '.json'
                with open(new_checkpoint_file, 'r') as f:
                    new_checkpoint = json.load(f)
                c['second_stage_only'] = new_checkpoint['first_stage_paths'] 

    assert len(config_dict_list) * args.seeds ==len(new_config_list)
    config_dict_list = copy.deepcopy(new_config_list ) 

    if args.name is not None : 
        args.name +=  time.strftime("_%Y-%m-%d")  
        args.results_dir += args.name
        os.mkdir(args.results_dir) 

    for i,c in enumerate(config_dict_list):
        c['experiment_name'] += 's{}'.format(int(c['seed']/sm))
        # The store path is the full file path for the first stage checkpoints to be stored. First stage checkpoints 
        # for all experiments are automatically stored
        c['store_path'] =  'experiment_paths/' + args.name+ '/' + copy.deepcopy(c['experiment_name']) + '_' +str(i)+ '.json'

    os.mkdir('experiment_paths/{}'.format(args.name))
    if config_dict_list[0].get('num_renders'):
        os.mkdir('gifs/{}'.format(args.name))
    
    for c in config_dict_list:
        c['results_dir'] = args.results_dir 
        c['parent_name'] = args.name 
        
    # Total number of jobs
    total_jobs = len(config_dict_list) 

    print('Total Jobs to Run:', total_jobs)
    # Parallel jobs possible 
    num_jobs = args.workers // args.w_per_job 

    if args.mp : 
        if args.gpu:
            for c in config_dict_list:
                c['num_gpus'] = 1/num_jobs 
        else :
            for c in config_dict_list:
                c['num_gpus'] = 0
    else : 
        if args.gpu:
            for c in config_dict_list:
                c['num_gpus'] = 1

    if args.mp :
        mp_run(num_jobs,total_jobs,args,config_dict_list) 
    else :
        sequential_run(total_jobs,args,config_dict_list)

  