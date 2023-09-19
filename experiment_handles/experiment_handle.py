import json
from pathlib import Path


# an abstract experiment handler class
class AbstractExperimentHandle:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config_dict_list = json.load(f)
        self.experiment_name = self.config_dict_list[0]['experiment_name']

    def run_exp(self):
        raise NotImplementedError

    # executed at start, any initialization call
    def hook_at_start(self):
        raise NotImplementedError

    # executed at end, any shutdown after add data
    def hook_at_end(self):
        raise NotImplementedError

    # logic for argument parsing
    def argument_parsing(self, config_dict):
        raise NotImplementedError

    # main execution loop, running experiments
    def main_exp(self, exp_params):
        raise NotImplementedError

