from pytest import set_trace

import Curriculum as gc
from multiprocessing import Process, Pipe
import torch
import torch.nn as nn
import torch.nn.functional as F

from ffai.core.table import OutcomeType

from time import sleep

worker_max_steps = 200

# --- Reward function ---
rewards_own = {
    OutcomeType.TOUCHDOWN: 1,
    OutcomeType.CATCH: 0.1,
    OutcomeType.INTERCEPTION: 0.2,
    OutcomeType.SUCCESSFUL_PICKUP: 0.1,
    OutcomeType.FUMBLE: -0.1,
    OutcomeType.KNOCKED_DOWN: -0.1,
    OutcomeType.KNOCKED_OUT: -0.2,
    OutcomeType.CASUALTY: -0.5
}
rewards_opp = {
    OutcomeType.TOUCHDOWN: -1,
    OutcomeType.CATCH: -0.1,
    OutcomeType.INTERCEPTION: -0.2,
    OutcomeType.SUCCESSFUL_PICKUP: -0.1,
    OutcomeType.FUMBLE: 0.1,
    OutcomeType.KNOCKED_DOWN: 0.1,
    OutcomeType.KNOCKED_OUT: 0.2,
    OutcomeType.CASUALTY: 0.5
}
ball_progression_reward = 0.005


def reward_function(env, info, shaped=False):
    r = 0
    for outcome in env.get_outcomes():
        if not shaped and outcome.outcome_type != OutcomeType.TOUCHDOWN:
            continue
        team = None
        if outcome.player is not None:
            team = outcome.player.team
        elif outcome.team is not None:
            team = outcome.team
        if team == env.own_team and outcome.outcome_type in rewards_own:
            r += rewards_own[outcome.outcome_type]
        if team == env.opp_team and outcome.outcome_type in rewards_opp:
            r += rewards_opp[outcome.outcome_type]
    if info['ball_progression'] > 0:
        r += info['ball_progression'] * ball_progression_reward
    return r


class Memory(object):
    def __init__(self, steps_per_update, env):
        spatial_obs_shape = env.get_spatial_obs_shape()
        non_spatial_obs_shape = env.get_non_spatial_obs_shape()
        action_space = env.get_action_shape()

        self.step = 0
        self.max_steps = steps_per_update

        self.spatial_obs = torch.zeros(steps_per_update, *spatial_obs_shape)
        self.non_spatial_obs = torch.zeros(steps_per_update, *non_spatial_obs_shape)
        self.rewards = torch.zeros(steps_per_update, 1)
        self.returns = torch.zeros(steps_per_update, 1)
        self.td_outcome = torch.zeros(steps_per_update, 1)

        action_shape = 1
        self.actions = torch.zeros(steps_per_update, action_shape)
        self.actions = self.actions.long()
        self.action_masks = torch.zeros(steps_per_update, action_space, dtype=torch.uint8)

    def cuda(self):
        pass

    def insert_worker_memory(self, worker_mem):
        steps_to_copy = worker_mem.get_steps_to_copy()

        # check that there's space left. 
        if self.max_steps - self.step < steps_to_copy:
            steps_to_copy = self.max_steps - self.step

        begin = self.step
        end = self.step + steps_to_copy

        self.spatial_obs[begin:end].copy_(worker_mem.spatial_obs[:steps_to_copy])
        self.non_spatial_obs[begin:end].copy_(worker_mem.non_spatial_obs[:steps_to_copy])
        self.rewards[begin:end].copy_(worker_mem.rewards[:steps_to_copy])
        self.returns[begin:end].copy_(worker_mem.returns[:steps_to_copy])
        self.td_outcome[begin:end].copy_(worker_mem.td_outcome[:steps_to_copy])

        self.actions[begin:end].copy_(worker_mem.actions[:steps_to_copy])
        self.action_masks[begin:end].copy_(worker_mem.action_masks[:steps_to_copy])

        self.step += steps_to_copy

    def not_full(self):
        return 0.9 * self.max_steps > self.step


class WorkerMemory(object):
    def __init__(self, max_steps, env):

        spatial_obs_shape = env.get_spatial_obs_shape()
        non_spatial_obs_shape = env.get_non_spatial_obs_shape()
        action_space = env.get_action_shape()

        self.max_steps = max_steps
        self.looped = False
        self.step = 0

        self.spatial_obs = torch.zeros(max_steps, *spatial_obs_shape)
        self.non_spatial_obs = torch.zeros(max_steps, *non_spatial_obs_shape)
        self.rewards = torch.zeros(max_steps, 1)
        self.returns = torch.zeros(max_steps, 1)
        self.td_outcome = torch.zeros(max_steps, 1)

        self.actions = torch.zeros(max_steps, 1).long()  # action_shape = 1
        self.action_masks = torch.zeros(max_steps, action_space, dtype=torch.uint8)

    def cuda(self):
        pass

    def insert_first_obs(self, spatial_obs, non_spatial_obs):
        # consider clearing the variables
        self.step = 0
        self.looped = False

        self.spatial_obs[0].copy_(spatial_obs)
        self.non_spatial_obs[0].copy_(non_spatial_obs)

    def insert_network_step(self, done, spatial_obs, non_spatial_obs, action, reward, action_masks):

        self.actions[self.step] = action
        self.rewards[self.step] = reward
        self.action_masks[self.step].copy_(action_masks)

        self.step += 1
        if self.step == self.max_steps:
            self.step = 0
            self.looped = True

        if not done:
            self.spatial_obs[self.step].copy_(spatial_obs)
            self.non_spatial_obs[self.step].copy_(non_spatial_obs)

    def insert_scripted_step(self, done, spatial_obs, non_spatial_obs, reward):
        # observation overwrites the previously inserted observations 
        if not done:
            self.spatial_obs[self.step].copy_(spatial_obs)
            self.non_spatial_obs[self.step].copy_(non_spatial_obs)

        # reward is added to the previously inserted reward 
        prev_step = self.step - 1 if self.step > 0 else self.max_steps - 1
        self.rewards[prev_step] += reward

    def insert_epside_end(self, td_outcome):
        gamma = 0.99
        self.td_outcome[:] = td_outcome

        # Compute returns 
        assert not (self.step == 0 and self.looped == False)
        if self.step != 0:

            self.returns[self.step - 1] = self.rewards[self.step - 1]
            for i in reversed(range(self.step - 1)):
                self.returns[i] = self.returns[i + 1] * gamma + self.rewards[i]

        if self.looped:
            self.returns[-1] = gamma * self.returns[0] + self.rewards[-1]
            for i in reversed(range(self.step + 1, self.max_steps - 1)):
                self.returns[i] = self.returns[i + 1] * gamma + self.rewards[i]

    def get_steps_to_copy(self):
        return self.max_steps if self.looped else self.step


class VecEnv():
    def __init__(self, envs, academy, starting_agent, memory_size):
        """
        envs: list of FFAI environments to run in subprocesses
        """
        self.closed = False
        self.academy = academy
        nenvs = len(envs)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        lectures = [academy.get_next_lecture() for i in range(nenvs)]

        self.ps = [Process(target=worker, args=(work_remote, remote, env, envs.index(env), [lect], starting_agent))
                   for (work_remote, remote, env, lect) in zip(self.work_remotes, self.remotes, envs, lectures)]

        for p in self.ps:
            p.daemon = True  # If the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.memory = Memory(memory_size, envs[0])

    def step(self):

        while self.memory.not_full():
            for remote in self.remotes:
                if remote.poll():
                    data = remote.recv()
                    self.memory.insert_worker_memory(data[0])
                    self.academy.report(data[1])

                    # TODO: queue another lecture?

            sleep(0.01)

        return True

    def update_trainee(self, agent):

        for remote in self.remotes:
            remote.send(('swap trainee', agent))

    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    @property
    def num_envs(self):
        return len(self.remotes)


def worker(remote, parent_remote, env, worker_id, lectures, trainee):
    parent_remote.close()

    assert len(lectures) > 0
    worker_running = True
    steps = 0

    reset_steps = 2000

    with torch.no_grad():

        lect = lectures.pop()
        obs = env.reset(lect)
        spatial_obs, non_spatial_obs = trainee._update_obs(obs)
        memory = WorkerMemory(worker_max_steps, env)
        memory.insert_first_obs(spatial_obs, non_spatial_obs)

        while worker_running:

            # Updates from master process?
            while remote.poll():
                command, data = remote.recv()
                if command == 'swap trainee':
                    trainee = data
                elif command == 'queue lecture':
                    lectures.append(data)
                elif command == 'close':
                    worker_running = False
                    break
                else:
                    raise Exception(f"Unknown command to worker: {command}")

            # Agent takes step  and insert to memory
            steps += 1
            (action, action_idx, action_masks, value, spatial_obs, non_spatial_obs) = trainee.act(game=None, env=env,
                                                                                                  obs=obs)
            obs, reward, done, info, lect_outcome = env.step(action)
            reward_shaped = reward_function(env, info, shaped=True)

            if action_idx is None or action_masks is None or value is None:
                memory.insert_scripted_step(done, spatial_obs, non_spatial_obs, reward_shaped)
            else:
                memory.insert_network_step(done, spatial_obs, non_spatial_obs, action_idx, reward_shaped, action_masks)

            # Check progress and report back
            if done:
                td_outcome = 0.5 * (1 + info['touchdowns'] - info['opp_touchdowns'])
                assert td_outcome in [0, 0.5, 1]
                assert type(lect_outcome) == gc.LectureOutcome
                lect_outcome.steps = steps

                memory.insert_epside_end(td_outcome)

                print(f"Worker {worker_id} - on step {steps} - sending")
                remote.send((memory, lect_outcome))
                print(f"Worker {worker_id} - on step {steps} - send complete")

                if len(lectures) > 0:
                    lect = lectures.pop()

                obs = env.reset(lecture=lect)
                spatial_obs, non_spatial_obs = trainee._update_obs(obs)
                memory.insert_first_obs(spatial_obs, non_spatial_obs)

                steps = 0

            if steps >= reset_steps:
                # If we  get stuck or something - reset the environment
                print("Max. number of steps exceeded! Consider increasing the number.")
                obs = env.reset(lecture=lect)
                spatial_obs, non_spatial_obs = trainee._update_obs(obs)
                memory.insert_first_obs(spatial_obs, non_spatial_obs)

                steps = 0