from ffai.core.load import *
from ffai.ai.bots.random_bot import *

import Lectures as gc
from Curriculum import Academy, LectureOutcome
from VectorEnvironment import VecEnv, Memory, WorkerMemory, reward_function

import itertools as it
import gym
import torch


class A2c_agent_tester(Agent):
    def __init__(self, env):
        super().__init__("a2c_random_tester")

        self.random_agent = RandomBot("Random_bot")
        self.cnn_used = False

        self.action_size = env.get_action_shape()

    def act(self, game, env=None, obs=None):
        if game is None:
            game = env.game

        action = self.random_agent.act(game)
        x = action.position.x if action.position is not None else None
        y = action.position.y if action.position is not None else None

        if action.position is None and action.player is not None:
            pos = action.player.position
            x = pos.x
            y = pos.y
            assert x is not None and y is not None

        action_object = {'action-type': action.action_type,
                         'x': x,
                         'y': y}

        self.cnn_used = self.cnn_used == False  # flips the state

        assert obs is not None
        spatial_obs, non_spatial_obs = self._update_obs(obs)

        if self.cnn_used:
            actions = 1
            action_masks = np.zeros(self.action_size)
            action_masks = torch.tensor(action_masks, dtype=torch.bool)
            value = 0
            return (action_object, actions, action_masks, value, spatial_obs, non_spatial_obs)
        else:
            return (action_object, None, None, None, spatial_obs, non_spatial_obs)

    def _update_obs(self, obs):
        """
        Takes the observation returned by the environment and transforms it to an numpy array that contains all of
        the feature layers and non-spatial info.
        """

        spatial_obs = np.stack(obs['board'].values())

        state = list(obs['state'].values())
        procedures = list(obs['procedures'].values())
        actions = list(obs['available-action-types'].values())

        non_spatial_obs = np.stack(state + procedures + actions)
        non_spatial_obs = np.expand_dims(non_spatial_obs, axis=0)

        return torch.from_numpy(np.stack(spatial_obs)).float(), torch.from_numpy(np.stack(non_spatial_obs)).float()


lectures_to_test = [gc.GameAgainstRandom(), gc.Scoring()]


def test_memories():
    env = gym.make("FFAI-v2")
    ac = Academy(lectures_to_test)
    obs = env.reset(ac.get_next_lecture())
    ag = A2c_agent_tester(env)
    spatial_obs, non_spatial_obs = ag._update_obs(obs)

    mem = Memory(300, env)
    w_mem = WorkerMemory(200, env)

    w_mem.insert_first_obs(spatial_obs, non_spatial_obs)

    while mem.not_full():
        (action, action_idx, action_masks, value, spatial_obs, non_spatial_obs) = ag.act(game=None, env=env, obs=obs)
        obs, reward, done, info, lect_outcome = env.step(action)
        reward_shaped = reward_function(env, info, shaped=True)

        if action_idx is None or action_masks is None or value is None:
            w_mem.insert_scripted_step(done, spatial_obs, non_spatial_obs, reward_shaped)
        else:
            w_mem.insert_network_step(done, spatial_obs, non_spatial_obs, action_idx, reward_shaped, action_masks)

        if done:
            w_mem.insert_epside_end(0.5)
            mem.insert_worker_memory(w_mem)

            obs = env.reset(ac.get_next_lecture())
            spatial_obs, non_spatial_obs = ag._update_obs(obs)
            w_mem.insert_first_obs(spatial_obs, non_spatial_obs)


def test_vec_env_small_board():
    N = 2
    envs = [gym.make("FFAI-5-v2") for i in range(N)]
    vec_env = VecEnv(envs, Academy(lectures_to_test), A2c_agent_tester(envs[0]), 10)
    vec_env.step()
    vec_env.close()


def test_vec_env_fullsize_board():
    N = 2
    envs = [gym.make("FFAI-v2") for i in range(N)]
    vec_env = VecEnv(envs, Academy(lectures_to_test), A2c_agent_tester(envs[0]), 10)
    vec_env.step()
    vec_env.close()
