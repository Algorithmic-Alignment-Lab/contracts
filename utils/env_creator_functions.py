from ray.tune.registry import register_env

# import utils.contracts
import contract.contract_list
from environments.cleanup_features import CleanupFeatures
from environments.cleanup_new import CleanupEnv
from environments.harvest_features import HarvestFeatures
from environments.harvest_new import HarvestEnv
from environments.self_driving_car_accelerate import SelfAcceleratingCarEnv
from environments.two_stage_train import SeparateContractNegotiateStage,SeparateContractSubgameStage, SeparateContractCombinedStage, JointEnv, NegotiationSolver

def env_creator(name,config):
    if name == 'SelfDrive':
        return SelfAcceleratingCarEnv(**config)
    elif name == 'Harvest':
        return HarvestFeatures(**config)
    elif name == 'HarvestNew':
        return HarvestEnv(**config)
    elif name == 'Cleanup':
        return CleanupFeatures(**config)
    elif name == 'CleanupNew':
        return CleanupEnv(**config)
    elif name =='ContractWrapperNegotiate':
        return SeparateContractNegotiateStage(**config)
    elif name =='ContractWrapperSubgame':
        return SeparateContractSubgameStage(**config)
    elif name =='ContractWrapperCombined':
        return SeparateContractCombinedStage(**config)
    elif name =='NegotiationSolver':
        return NegotiationSolver(**config)
    elif name =='JointEnv':
        return JointEnv(**config)
    else :
        raise ValueError('Environment not found')

register_env('SelfDrive', lambda config: env_creator('SelfDrive',config))
register_env('Harvest', lambda config: env_creator('Harvest',config))
register_env('HarvestNew', lambda config: env_creator('HarvestNew',config))
register_env('Cleanup', lambda config: env_creator('Cleanup',config))
register_env('CleanupNew', lambda config: env_creator('CleanupNew',config))
register_env('ContractWrapperNegotiate', lambda config: env_creator('ContractWrapperNegotiate',config))
register_env('ContractWrapperSubgame', lambda config: env_creator('ContractWrapperSubgame',config))
register_env('ContractWrapperCombined', lambda config: env_creator('ContractWrapperCombined',config))
register_env('NegotiationSolver', lambda config: env_creator('NegotiationSolver',config))
register_env('JointEnv', lambda config: env_creator('JointEnv',config))

def get_base_env_tag(arg_dict) : 
    if arg_dict.get("environment") == 'selfdrive':
        base_env_tag = "SelfDrive"
    elif arg_dict.get("environment") == 'harvest':
        base_env_tag = "Harvest"
    elif arg_dict.get("environment") == 'harvest_new':
        base_env_tag = 'HarvestNew'
    elif arg_dict.get("environment") == 'cleanup':
        base_env_tag = 'Cleanup'
    elif arg_dict.get("environment") == 'cleanup_new':
        base_env_tag = 'CleanupNew'
    else:
        assert False
    return base_env_tag
