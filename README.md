# Get It in Writing: Formal Contracts Mitigate Social Dilemmas in Multi-Agent RL
![logo; Midjourney "A minimalistic geometric logo on white background, a hand of an artificial intelligence signs a contract, visible pen and paper"](./logo.png)
## Installation 
1. Install required packages using [Conda](https://docs.conda.io/en/latest/):
   ```bash
   conda env create -f requirements.yml 
   conda activate contracting
   ```  
2. Create results directories:
   ```bash
   mkdir gifs results experiment_paths
   ```  
3. Test the installation: 
   ```bash
   python runner.py --name "test-v1" --config_path "experiment_configs/test.json"
   ```

## Running experiments
This codebase is built on [RLlib](https://docs.ray.io/en/latest/rllib/index.html) and has a multi-layer structure to enable large benchmarking runs. 
- **Low Level - Config Files:** Users need to create configuration files which specify the parameters for the jobs to be run.
A single config file can be used to specify the parameters for multiple jobs to be run, as detailed in the section below. 
- **High Level - Scheduler:**  If multiple jobs are to be run, users have control over how they are scheduled. They can be run in parallel using python"s [`multiprocessing` library](https://docs.python.org/3/library/multiprocessing.html) or run sequentially. More details can be found in the section below. 

### Configuration Files  
A config file is a `json` file that specifies the parameters for the jobs to be run. Configuration files are defined as a list of dictionaries:

- The first dictionary in the list defines values for all jobs in a file.

- The second dictionary contains hyperparameters. Keys are hyperparameter names, and values are lists of values. All combinations of all parameters will be run.

- The remaining elements of the list can be used to specify any job-specific parameters. These job-specific parameters can also be used to override the global dictionary for a specific job. 

The following image shows an example config file built using the rules described above. 

> **Example**: There are 6 possible permutations from the permutation dictionary. Each of these 6 permutations are then run on both Harvest and Cleanup environments, leading to a total of **12 jobs**. Each job uses 16 workers, 15M timesteps and a batch size of 64000 as defined in the global dictionary. However, all jobs in Harvest use a batch size of 32000 due to the override. More configuration files can be found inside `experiment_configs/` 

```json
[
  {
    "num_workers": 16,
    "num_timesteps": 15000000,
    "batch_size": 64000,
  },
  {
    "num_agents":[8,4,2],
    "separate": [true, false]
  },
  {
    "environment": "cleanup_new",
  },
  {
    "environment": "harvest_new",
    "batch_size" : 32000
  },
]
```

### Scheduler
Config files allow for the specification of multiple jobs where each job requires a specified number of workers $N$. 
There are two ways to run these jobs:

- **Sequentially**: Jobs are run sequentially by default:
```bash
python runner.py --name "cleanup-complete" --config_path "experiment_configs/cleanup-contracting.json" 
```

To run with gpu enabled, add `--gpu` 
```bash
python runner.py --name "cleanup-complete" --config_path "experiment_configs/cleanup-contracting.json" --gpu
```

- **In Parallel**: Jobs in the config can be parallelized by adding the `--mp` flag and specifying the total number of workers available as well as the number of workers required per job. The following code runs 4 jobs parallely. 
```bash
python runner.py --name "cleanup-complete" --config_path "experiment_configs/cleanup-contracting.json" --w_per_job 16 --workers 64 --mp 
```

## Available Environments
The following environments are available to train. The first column can be used in a configuration.

| String Representation   | Class | Description | Contract to Use (if contracting) | 
| ----------------- | ----------- | ----------- | ----------- |
| `cleanup_new`| `CleanupEnv` | image-based cleanup, adapted from [here][harvest] |  `CleanupContract`    |
| `cleanup` | `CleanupFeatures` | feature-based cleanup, manually-designed features | `CleanupContract`    |
| `harvest_new` | `HarvestEnv` |image-based harvest, adapted from [here][harvest]  | `HarvestFeaturemodLocalContract`  |
| `harvest` | `HarvestFeatures` | feature-based harvest, manually-designed features |  `HarvestFeaturemodLocalContract`     |
| `selfdrive` | `SelfAcceleratingCarEnv` | self-driving merge domain |  `SelfdriveContractDistprop`  |

## Configuration Parameters
The following is a partial list of paramters, which are parsed in `utils/ray_config_utils.py`. Additional parameters are defined in [RLLib](https://docs.ray.io/en/latest/rllib/index.html).


| Argument | Type | Description | Default Value | 
| ------------- | ----------- | ----------- | ----------- |
| `num_timesteps` | `int` | Training timesteps | required, 10M recommended |
| `num_workers` | `int` | Number of parallel `ray` workers | required, $1$ turns of parallelism |
| `num_agents` | `int` | Number of agents in the environment ($2 \le n \le 8$) | required |
| `batch_size` | `int` | Number of timesteps to collect before each training update | required |
| `contract` | `str` |Contract space to use | Required for contracting runs, refer `contract/contract_list` for available contracts | required |
| `wandb` | `bool` | Logging on [Weights and Biases](https://wandb.ai/site) | `False` | 
| `separate` | `bool` | No-contracting, separate training | `False` |  
| `joint` | `bool` | Joint training, single controller | `False` |
| `solver` | `bool` | `NegotiationSolver` is used to find the proposed contract | `False` |
| `shared_policy` |  `bool` | All agents share a policy | `False` |
| `num_renders` | `int` | Generate renders at end of training for `HarvestEnv`/`CleanupEnv` | required |
| `horizon` |  `int` |  Episode Length | Required, recommended is 1000 for `HarvestEnv`/`CleanupEnv`/`SelfAcceleratingCarEnv` | required |
| `env_args` |  `dict` | Values passed to base environment | See environment definitions and sample configuration files | required |
| `model_params` | `dict` | Values passed to the RL model | See `run_training.py` | required |

## Sample Configurations
Sample configuration files are in `continuous_domain/experiment_configs`. The use of 8 workers per job in the configuration files is arbitrary and should be adjusted.

| Config   | Description | #Jobs |
| ----------------- | ----------- |  -----|
| `cleanup-baseline-2agents.json` | Runs `CleanupEnv` with 2 agents and no contracting |  1  | 
| `cleanup-contracting-2agents.json` | Runs `CleanupEnv` with 2 agents and contracting |  1  | 
| `cleanup-joint-2agents.json` | Runs the `CleanupEnv` with 2 agents and a joint controller |  1  | 
| `cleanup-contracting.json` | Runs `CleanupEnv` with 2, 4, and 8 agents, and contracting |  3 |  
| `harvest-contracting.json` | Runs `HarvestEnv` 2,4, and 8 agents, and contracting |  3 | 
| `driving-contracting.json` | Runs `SelfAcceleratingCarEnv` with 2, 4, and 8 agents, and contracting | 3 | 
| `contracting-full.json` | Runs `CleanupEnv`, `HarvestEnv`, and `SelfAcceleratingCarEnv` with 2, 4, and 8 agents | 9 | 
| `baseline-full.json`    | Runs `CleanupEnv`, `HarvestEnv`, and `SelfAcceleratingCarEnv` env with 2, 4, and 8 agents and no contracting | 9 | 
| `cleanup-old-contracting-2agents.json` | Runs contracting in cleanup env with 2 agents |  1  | 
| `harvest-old-contracting-2agents.json` | Runs contracting in old feature-based harvest env with 2 agents |  1  | 

> **Example:** The following command runs contracting on all environments in parallel using a GPU and 32 workers, with eight workers per job.

```bash
python runner.py --name "contracting-full-v1" --config_path "experiment_configs/contracting-full.json" --w_per_job 8 --workers 32 --mp --gpu
```

## Overview of Files
The following files are important for the codebase:

- `runner.py`: Initializes jobs.
- `experiments_handles/contracts_experiment_handle.py`: Parses the config, and calls `run_training.py`.
- `run_training.py`: Implements the training pipeline.
- `utils/ray_config_utils.py`: RL model configuration parameters are processed in this file. 
- `contract/contract_list.py`: Implements contract spaces.
- `environments/two_stage_train.py`: Defines the contracting augmentation as wrappers.

## Logging 
[Weights and biases][wandb] is integrated with the codebase and is recommended to visualize training results as well as other performance metrics.

### Finding the optimal contract
In the contracting algorithm presented in the original [Formal Contracting Mitigates Social Dilemmas in Multi-Agent RL][contracting], a reinforcement-learning-based algorithm was implemented. This repository implements a second solver.

1. `SeparateContractNegotiationStage` follows the method described in the paper.
2. `NegotationSolver` The solver uses the agent"s frozen value functions to find the contract that maximizes welfare subject to the constraint that all agents accept the contract. Adding `solver=true` in the config file enables this for contracting.
   
## Citation

If you want to cite this repository accademic work, please use the following citation:

```
@inproceedings{contracts,
author  = {Christophersen, Philip and Haupt, Andreas and Hadfield-Menell, Dylan},
title   = {Getting it in Writing: Formal Contracting Mitigates Social Dilemmas in Multi-Agent Reinforcement Learning},
year    = {2022},
booktitle    = {Proceedings of the 22nd International Conference on Autonomous Agents and Multiagent Systems},
}
```

[wandb]: https://wandb.ai/site
[harvest]: https://github.com/eugenevinitsky/sequential_social_dilemma_games
[contracting]: https://arxiv.org/abs/2208.10469
