import numpy as np
from typing import Dict, List
import gymnasium as gym

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import (
    normc_initializer,
    same_padding,
    SlimConv2d,
    SlimFC,
)
from ray.rllib.models.utils import get_activation_fn, get_filter_config
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from copy import deepcopy 
from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions

torch, nn = try_import_torch()

import torch.nn as nn

class VisionNetwork(TorchModelV2,nn.Module):
    """Generic vision network.
    This network only has capability of separate value and policy networks,
    please customize accodingly, ray is stupid and getting the shared network 
    setup is left for the future. 
    """

    def __init__(
        self,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):

        if not model_config.get("conv_filters"):
            model_config["conv_filters"] = get_filter_config(obs_space.shape)
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        activation = self.model_config.get("conv_activation")
        filters = self.model_config["conv_filters"]
        assert len(filters) > 0, "Must provide at least 1 entry in `conv_filters`!"

        self.combined_dim = 0
        self.image_space = self.obs_space.original_space['image'] 
        self.i_space,self.c_space,self.f_space = False,False,False
        if 'image' in self.obs_space.original_space.keys():
            self.i_space= True 
            self.image_space = self.obs_space.original_space['image'] 
        if 'contract' in self.obs_space.original_space.keys():
            self.c_space = True
            self.c_dim = self.obs_space.original_space['contract'].shape[0]
            self.combined_dim += self.c_dim
        if 'features' in self.obs_space.original_space.keys():
            self.f_space = True
            self.f_dim = self.obs_space.original_space['features'].shape[0]
            self.combined_dim += self.f_dim
                    
        # Post FC net config.
        post_fcnet_hiddens = model_config.get("post_fcnet_hiddens", [])
        post_fcnet_activation = get_activation_fn(
            model_config.get("post_fcnet_activation"), framework="torch"
        )
        # Whether the last layer is the output of a Flattened (rather than
        # a n x (1,1) Conv2D).
        #self.last_layer_is_flattened = False
        self._logits = None
        
        conv_layers,post_conv_layers = self.get_network(self.image_space,filters,post_fcnet_hiddens,activation,post_fcnet_activation,num_outputs,vf=False)
        self._convs =  nn.Sequential(*conv_layers)
        self._post_convs = nn.Sequential(*post_conv_layers)
        vf_conv_layers,vf_post_conv_layers = self.get_network(self.image_space,filters,post_fcnet_hiddens,activation,post_fcnet_activation,num_outputs=1)
        self._vf_convs = nn.Sequential(*vf_conv_layers)
        self._vf_post_convs = nn.Sequential(*vf_post_conv_layers)
        self._features = None

    def get_network(self,obs_space,filters,post_fcnet_hiddens,activation,post_fcnet_activation,num_outputs=None,vf=True):
        layers = []
        (w, h, in_channels) = obs_space.shape
        in_size = [w, h]
        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = same_padding(in_size, kernel, stride)
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    activation_fn=activation,
                )
            )
            in_channels = out_channels
            in_size = out_size
        
        out_channels, kernel, stride = filters[-1]

        layers.append(
            SlimConv2d(
                in_channels,
                out_channels,
                kernel,
                stride,
                None,  # padding=valid
                activation_fn=activation,
            )
        )
        layers.append(nn.Flatten())
        out_size = self.get_out_size(layers)  
        out_channels = out_size

        if num_outputs:
            in_size = [
                np.ceil((in_size[0] - kernel[0]) / stride),
                np.ceil((in_size[1] - kernel[1]) / stride),
            ]
            padding, _ = same_padding(in_size, [1, 1], [1, 1])
            layers.append(nn.Flatten())
            ## Concatenate contract here 
            in_size = out_channels + self.combined_dim*5 
            conv_layers = deepcopy(layers)
            post_conv_layers = []
            # Add (optional) post-fc-stack after last Conv2D layer.
            for i, out_size in enumerate(post_fcnet_hiddens + [num_outputs]):
                post_conv_layers.append(
                    SlimFC(
                        in_size=in_size,
                        out_size=out_size,
                        activation_fn=post_fcnet_activation
                        if i < len(post_fcnet_hiddens) - 1
                        else None,
                        initializer=normc_initializer(1.0),
                    )
                )
                in_size = out_size
            if not vf:
            # Last layer is logits layer.
                self._logits = post_conv_layers.pop()

        return conv_layers,post_conv_layers 
       


    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType):
        
       # print('a',input_dict["obs"].shape)
        orig_obs = restore_original_dimensions(input_dict["obs"], self.obs_space, "torch")
        self._features = orig_obs["image"].float()
        if self.c_space : 
            self._contracts = orig_obs["contract"].float()
            self._contracts = self._contracts.repeat(1,5)
        if self.f_space : 
            self._feat = orig_obs["features"].float()
            self._feat= self._feat.repeat(1,5)
        # Permuate b/c data comes in as [B, dim, dim, channels]:
        self._features = self._features.permute(0, 3, 1, 2)
        conv_out = self._convs(self._features)
        # Concatenate contract here
        if self.f_space and self.c_space:
            conv_out = torch.concat((conv_out,self._contracts,self._feat),dim=1)
        elif self.f_space:
            conv_out = torch.concat((conv_out,self._feat),dim=1)
        elif self.c_space:
            conv_out = torch.concat((conv_out,self._contracts),dim=1)
        else:
            pass
        conv_out = self._post_convs(conv_out)
        conv_out = self._logits(conv_out)
        logits = conv_out
        return logits, state
     
    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._features is not None, "must call forward() first"
        value = self._vf_convs(self._features)
        # Concatenate contract here
        if self.f_space and self.c_space:
            value = torch.concat((value,self._contracts,self._feat),dim=1)
        elif self.f_space:
            value = torch.concat((value,self._feat),dim=1)
        elif self.c_space:
            value = torch.concat((value,self._contracts),dim=1)
        else:
            pass

        value = self._vf_post_convs(value)
        value = value.squeeze(1)
        return value 
        
    def _hidden_layers(self, obs: TensorType) -> TensorType:
        res = self._convs(obs.permute(0, 3, 1, 2))  # switch to channel-major
        res = res.squeeze(3)
        res = res.squeeze(2)
        return res
    
    def get_out_size(self,layers):
        # Create a B=1 dummy sample and push it through out conv-net.
        dummy_in = (
            torch.from_numpy(self.image_space.sample())
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            )
        dummy_net = nn.Sequential(*layers)
        dummy_out = dummy_net(dummy_in)
        dummy_out = dummy_out.view(dummy_out.size(0),-1)
        return int(dummy_out.size(1))


if __name__=='__main__': 
    from harvest_features import HarvestFeatures
    model_config= {
    # Experimental flag.
    # If True, try to use a native (tf.keras.Model or torch.Module) default
    # model instead of our built-in ModelV2 defaults.
    # If False (default), use "classic" ModelV2 default models.
    # Note that this currently only works for:
    # 1) framework != torch AND
    # 2) fully connected and CNN default networks as well as
    # auto-wrapped LSTM- and attention nets.
    "_use_default_native_models": False,
    # Experimental flag.
    # If True, user specified no preprocessor to be created
    # (via config._disable_preprocessor_api=True). If True, observations
    # will arrive in model as they are returned by the env.
    "_disable_preprocessor_api": False,
    # Experimental flag.
    # If True, RLlib will no longer flatten the policy-computed actions into
    # a single tensor (for storage in SampleCollectors/output files/etc..),
    # but leave (possibly nested) actions as-is. Disabling flattening affects:
    # - SampleCollectors: Have to store possibly nested action structs.
    # - Models that have the previous action(s) as part of their input.
    # - Algorithms reading from offline files (incl. action information).
    "_disable_action_flattening": False,

    # === Built-in options ===
    # FullyConnectedNetwork (tf and torch): rllib.models.tf|torch.fcnet.py
    # These are used if no custom model is specified and the input space is 1D.
    # Number of hidden layers to be used.
    "fcnet_hiddens": [256, 256],
    # Activation function descriptor.
    # Supported values are: "tanh", "relu", "swish" (or "silu"),
    # "linear" (or None).
    "fcnet_activation": "tanh",

    # VisionNetwork (tf and torch): rllib.models.tf|torch.visionnet.py
    # These are used if no custom model is specified and the input space is 2D.
    # Filter config: List of [out_channels, kernel, stride] for each filter.
    # Example:
    # Use None for making RLlib try to find a default filter setup given the
    # observation space.
    "conv_filters": None,
    # Activation function descriptor.
    # Supported values are: "tanh", "relu", "swish" (or "silu"),
    # "linear" (or None).
    "conv_activation": "relu",

    # Some default models support a final FC stack of n Dense layers with given
    # activation:
    # - Complex observation spaces: Image components are fed through
    #   VisionNets, flat Boxes are left as-is, Discrete are one-hot'd, then
    #   everything is concated and pushed through this final FC stack.
    # - VisionNets (CNNs), e.g. after the CNN stack, there may be
    #   additional Dense layers.
    # - FullyConnectedNetworks will have this additional FCStack as well
    # (that's why it's empty by default).
    "post_fcnet_hiddens": [64,64],
    "post_fcnet_activation": "relu",

    # For DiagGaussian action distributions, make the second half of the model
    # outputs floating bias variables instead of state-dependent. This only
    # has an effect is using the default fully connected net.
    "free_log_std": False,
    # Whether to skip the final linear layer used to resize the hidden layer
    # outputs to size `num_outputs`. If True, then the last hidden layer
    # should already match num_outputs.
    "no_final_linear": False,
    # Whether layers should be shared for the value function.
    "vf_share_layers": True,

    # == LSTM ==
    # Whether to wrap the model with an LSTM.
    "use_lstm": False,
    # Max seq len for training the LSTM, defaults to 20.
    "max_seq_len": 20,
    # Size of the LSTM cell.
    "lstm_cell_size": 256,
    # Whether to feed a_{t-1} to LSTM (one-hot encoded if discrete).
    "lstm_use_prev_action": False,
    # Whether to feed r_{t-1} to LSTM.
    "lstm_use_prev_reward": False,
    # Whether the LSTM is time-major (TxBx..) or batch-major (BxTx..).
    "_time_major": False,

    # == Attention Nets (experimental: torch-version is untested) ==
    # Whether to use a GTrXL ("Gru transformer XL"; attention net) as the
    # wrapper Model around the default Model.
    "use_attention": False,
    # The number of transformer units within GTrXL.
    # A transformer unit in GTrXL consists of a) MultiHeadAttention module and
    # b) a position-wise MLP.
    "attention_num_transformer_units": 1,
    # The input and output size of each transformer unit.
    "attention_dim": 64,
    # The number of attention heads within the MultiHeadAttention units.
    "attention_num_heads": 1,
    # The dim of a single head (within the MultiHeadAttention units).
    "attention_head_dim": 32,
    # The memory sizes for inference and training.
    "attention_memory_inference": 50,
    "attention_memory_training": 50,
    # The output dim of the position-wise MLP.
    "attention_position_wise_mlp_dim": 32,
    # The initial bias values for the 2 GRU gates within a transformer unit.
    "attention_init_gru_gate_bias": 2.0,
    # Whether to feed a_{t-n:t-1} to GTrXL (one-hot encoded if discrete).
    "attention_use_n_prev_actions": 0,
    # Whether to feed r_{t-n:t-1} to GTrXL.
    "attention_use_n_prev_rewards": 0,

    # == Atari ==
    # Set to True to enable 4x stacking behavior.
    "framestack": True,
    # Final resized frame dimension
    "dim": 84,
    # (deprecated) Converts ATARI frame to 1 Channel Grayscale image
    "grayscale": False,
    # (deprecated) Changes frame to range from [-1, 1] if true
    "zero_mean": True,

    # === Options for custom models ===
    # Name of a custom model to use
    "custom_model": None,
    # Extra options to pass to the custom classes. These will be available to
    # the Model's constructor in the model_config field. Also, they will be
    # attempted to be passed as **kwargs to ModelV2 models. For an example,
    # see rllib/models/[tf|torch]/attention_net.py.
    "custom_model_config": {},
    # Name of a custom action distribution to use.
    "custom_action_dist": None,
    # Custom preprocessors are deprecated. Please use a wrapper class around
    # your environment instead to preprocess observations.
    "custom_preprocessor": None
}
    env = HarvestFeatures(image_obs=True)
    model_config['conv_filters'] = [[16,[4,4],2],[32,[6,6],2] ]
    o = env.reset() 
