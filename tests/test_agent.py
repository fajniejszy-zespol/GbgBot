import os

import gym
import torch
from pytest import set_trace

from Train_agent_2_0 import calc_network_shape
from a2c_agent import CNNPolicy, A2CAgent


def test_ac_agent():
    env_name = "FFAI-3-v2"
    env = gym.make(env_name)

    num_hidden_nodes = 4
    num_cnn_kernels = [3, 4]

    shape = calc_network_shape(env)

    ac_agent = CNNPolicy(shape['spat_obs'], shape['non_spat_obs'], hidden_nodes=num_hidden_nodes,
                         kernels=num_cnn_kernels, actions=shape['action_space'])

    filename = "test_agent_delete"
    torch.save(ac_agent, filename)

    agent = A2CAgent("trainee", env_name=env_name, filename=filename)

    os.remove(filename)

    obs = env.reset()
    done = False
    i = 0
    while not done:

        #(action, action_idx, action_masks, value, spatial_obs, non_spatial_obs) = agent.act(game=None, env=env, obs=obs)
        act_return = agent.act(game=None, env=env, obs=obs)

        try:
            obs, reward, done, info, lect_outcome = env.step(act_return[0])
        except TypeError:
            set_trace()

        i += 1
        #reward_shaped = reward_function(env, info, shaped=True)

