import os

import gym
import torch
from pytest import set_trace
import pytest

from Train_agent_2_0 import calc_network_shape
from a2c_agent import CNNPolicy, A2CAgent

@pytest.mark.parametrize("env_name", ["FFAI-7-v2", "FFAI-v2"])
def test_ac_agent(env_name):

    env = gym.make(env_name)

    num_hidden_nodes = 4
    num_cnn_kernels = [[64, None],
                       [64, 3],
                       [64, 3],
                       [None, 3]]

    shape = calc_network_shape(env)

    ac_agent = CNNPolicy(shape['spat_obs'], shape['non_spat_obs'], hidden_nodes=num_hidden_nodes,
                         kernels=num_cnn_kernels, actions=shape['action_space'],
                         spatial_action_types=shape["num_spat_action_types"],
                         non_spat_actions=shape["num_non_spat_actions"])

    filename = "test_agent_delete"
    torch.save(ac_agent, filename)

    agent = A2CAgent("trainee", env_name=env_name, filename=filename)

    os.remove(filename)

    env.reset()
    done = False
    i = 0
    while not done:

        #(action, action_idx, action_masks, value, spatial_obs, non_spatial_obs) = agent.act(game=None, env=env, obs=obs)
        act_return = agent.act(game=None, env=env)

        try:
            obs, reward, done, info, lect_outcome = env.step(act_return[0])
        except TypeError:
            set_trace()

        i += 1
        #reward_shaped = reward_function(env, info, shaped=True)

