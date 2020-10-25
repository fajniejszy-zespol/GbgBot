import os
import gym
from ffai import FFAIEnv
from torch.autograd import Variable
import torch.optim as optim
from multiprocessing import Process, Pipe
from ffai.ai.layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
from a2c_agent import A2CAgent, CNNPolicy
import ffai
import random

from pytest import set_trace 

# Training configuration
num_steps = 1000000
num_processes = 8
steps_per_update = 20 
learning_rate = 0.001
gamma = 0.99
entropy_coef = 0.01
value_loss_coef = 0.5
prediction_loss_coeff = 0.25
max_grad_norm = 0.05
log_interval = 20
save_interval = 500
ppcg = False
assert not ppcg 

# Environment
env_name = "FFAI-1-v2"
#env_name = "FFAI-3-v2"
#num_steps = 10000000 # Increase training time
#log_interval = 100
#env_name = "FFAI-5-v2"
#num_steps = 100000000 # Increase training time
#log_interval = 1000
#save_interval = 5000
# env_name = "FFAI-v2"
reset_steps = 5000  # The environment is reset after this many steps it gets stuck

# Self-play
selfplay = False  # Use this to enable/disable self-play
selfplay_window = 1
selfplay_save_steps = int(num_steps / 10)
selfplay_swap_steps = selfplay_save_steps

# Architecture
num_hidden_nodes = 128
num_cnn_kernels = [32, 64]

model_name = env_name
log_filename = "logs/" + model_name + ".dat"


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

ensure_dir("logs/")
ensure_dir("models/")
ensure_dir("plots/")

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

def calc_network_shape(env): 
    spatial_obs_space = env.observation_space.spaces['board'].shape
    board_dim = (spatial_obs_space[1], spatial_obs_space[2])
    board_squares = spatial_obs_space[1] * spatial_obs_space[2]

    non_spatial_obs_space = env.observation_space.spaces['state'].shape[0] + env.observation_space.spaces['procedures'].shape[0] + env.observation_space.spaces['available-action-types'].shape[0]
    non_spatial_action_types = FFAIEnv.simple_action_types + FFAIEnv.defensive_formation_action_types + FFAIEnv.offensive_formation_action_types
    num_non_spatial_action_types = len(non_spatial_action_types)
    spatial_action_types = FFAIEnv.positional_action_types
    num_spatial_action_types = len(spatial_action_types)
    num_spatial_actions = num_spatial_action_types * spatial_obs_space[1] * spatial_obs_space[2]
    action_space = num_non_spatial_action_types + num_spatial_actions

    
    
    shape = {'spat_obs': spatial_obs_space, 
             'non_spat_obs': non_spatial_obs_space, 
             'board_dim': board_dim, 
             'num_spat_action_types': num_spatial_action_types, 
             'num_spat_actions': num_spatial_actions, 
             'num_non_spat_actions': num_non_spatial_action_types}
    return shape 
    
def main():
    if True: 
        es = [make_env(i) for i in range(num_processes)]
        envs = VecEnv([es[i] for i in range(num_processes)])

        spatial_obs_space = es[0].observation_space.spaces['board'].shape
        board_dim = (spatial_obs_space[1], spatial_obs_space[2])
        board_squares = spatial_obs_space[1] * spatial_obs_space[2]

        non_spatial_obs_space = es[0].observation_space.spaces['state'].shape[0] + es[0].observation_space.spaces['procedures'].shape[0] + es[0].observation_space.spaces['available-action-types'].shape[0]
        non_spatial_action_types = FFAIEnv.simple_action_types + FFAIEnv.defensive_formation_action_types + FFAIEnv.offensive_formation_action_types
        num_non_spatial_action_types = len(non_spatial_action_types)
        spatial_action_types = FFAIEnv.positional_action_types
        num_spatial_action_types = len(spatial_action_types)
        num_spatial_actions = num_spatial_action_types * spatial_obs_space[1] * spatial_obs_space[2]
        action_space = num_non_spatial_action_types + num_spatial_actions

        
        # Clear log file
        try: os.remove(log_filename)
        except OSError: pass

        # MODEL
        ac_agent = CNNPolicy(spatial_obs_space, non_spatial_obs_space, hidden_nodes=num_hidden_nodes, kernels=num_cnn_kernels, actions=action_space)

        # OPTIMIZER
        optimizer = optim.RMSprop(ac_agent.parameters(), learning_rate)

        # Create the agent 
        torch.save(ac_agent, "models/" + model_name)
        agent = A2CAgent("trainee", env_name=env_name, filename= "models/" + model_name )
        
        # send agent to environments 
        envs.update_trainee(agent) 
    
    
    while all_steps < num_steps:

        # Step until memory is filled, access memory through envs.memory 
        envs.step() 
        memory = envs.memory 
        
        # ### Evaluate the actions taken ### 
        spatial = Variable(memory.spatial_obs)
        #spatial = spatial.view(-1, *spatial_obs_space)
        non_spatial = Variable(memory.non_spatial_obs)
        #non_spatial = non_spatial.view(-1, non_spatial.shape[-1]) 
        
        
        actions = Variable(torch.LongTensor(memory.actions.view(-1, 1)))
        actions_mask = Variable(memory.action_masks)

        action_log_probs, values, dist_entropy, td_pred = ac_agent.evaluate_actions(spatial, non_spatial, actions, actions_mask)

        # ### Compute loss and back propagate ### 
        #values = values.view(steps_per_update, num_processes, 1)
        #action_log_probs = action_log_probs.view(steps_per_update, num_processes, 1)

        advantages = Variable(memory.returns) - values
        value_loss = advantages.pow(2).mean()
        
        outcome_prediction_error = Variable(memory.td_outcome) - td_pred
        prediction_loss = outcome_prediction_error.pow(2).mean() 
        
        action_loss = -(Variable(advantages.data) * action_log_probs).mean()
        
        
        
        optimizer.zero_grad()

        total_loss = (value_loss * value_loss_coef + prediction_loss * prediction_loss_coeff + action_loss - dist_entropy * entropy_coef)
        total_loss.backward()

        nn.utils.clip_grad_norm_(ac_agent.parameters(), max_grad_norm)

        optimizer.step()
        
        # Updates
        all_updates += 1
        # Episodes
        all_episodes += episodes
        episodes = 0
        # Steps
        all_steps += num_processes * steps_per_update

        # Self-play save
        if selfplay and all_steps >= selfplay_next_save:
            selfplay_next_save = max(all_steps+1, selfplay_next_save+selfplay_save_steps)
            model_path = f"models/{model_name}_selfplay_{selfplay_models}"
            print(f"Saving {model_path}")
            torch.save(ac_agent, model_path)
            selfplay_models += 1
            
            # TODO: add lecture of this agent. 

        
        # Save
        if all_updates % save_interval == 0 and len(episode_rewards) >= num_processes:
            # Save to files
            with open(log_filename, "a") as myfile:
                myfile.write(log_to_file)

        # Logging
        if all_updates % log_interval == 0 and len(episode_rewards) >= num_processes:
            td_rate = np.mean(episode_tds)
            td_rate_opp = np.mean(episode_tds_opp)
            episode_tds.clear()
            episode_tds_opp.clear()
            mean_reward = np.mean(episode_rewards)
            episode_rewards.clear()
            win_rate = np.mean(wins)
            wins.clear()
            #mean_value_loss = np.mean(value_losses)
            #mean_policy_loss = np.mean(policy_losses)    
            
            log_updates.append(all_updates)
            log_episode.append(all_episodes)
            log_steps.append(all_steps)
            log_win_rate.append(win_rate)
            log_td_rate.append(td_rate)
            log_td_rate_opp.append(td_rate_opp)
            log_mean_reward.append(mean_reward)
            log_difficulty.append(difficulty)

            log = "Updates: {}, Episodes: {}, Timesteps: {}, Win rate: {:.2f}, TD rate: {:.2f}, TD rate opp: {:.2f}, Mean reward: {:.3f}, Difficulty: {:.2f}" \
                .format(all_updates, all_episodes, all_steps, win_rate, td_rate, td_rate_opp, mean_reward, difficulty)

            log_to_file = "{}, {}, {}, {}, {}, {}, {}\n" \
                .format(all_updates, all_episodes, all_steps, win_rate, td_rate, td_rate_opp, mean_reward, difficulty)

            print(log)

            episodes = 0
            value_losses.clear()
            policy_losses.clear()

            # Save model
            torch.save(ac_agent, "models/" + model_name)
            
            # plot
            n = 3
            if ppcg:
                n += 1
            fig, axs = plt.subplots(1, n, figsize=(4*n, 5))
            axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            axs[0].plot(log_steps, log_mean_reward)
            axs[0].set_title('Reward')
            #axs[0].set_ylim(bottom=0.0)
            axs[0].set_xlim(left=0)
            axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            axs[1].plot(log_steps, log_td_rate, label="Learner")
            axs[1].set_title('TD/Episode')
            axs[1].set_ylim(bottom=0.0)
            axs[1].set_xlim(left=0)
            if selfplay:
                axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                axs[1].plot(log_steps, log_td_rate_opp, color="red", label="Opponent")
            axs[2].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            axs[2].plot(log_steps, log_win_rate)
            axs[2].set_title('Win rate')            
            axs[2].set_yticks(np.arange(0, 1.001, step=0.1))
            axs[2].set_xlim(left=0)
            if ppcg:
                axs[3].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                axs[3].plot(log_steps, log_difficulty)
                axs[3].set_title('Difficulty')
                axs[3].set_yticks(np.arange(0, 1.001, step=0.1))
                axs[3].set_xlim(left=0)
            fig.tight_layout()
            fig.savefig(f"plots/{model_name}{'_selfplay' if selfplay else ''}.png")
            plt.close('all')

        #send updated agent to workers
        agent.policy = ac_agent 
        envs.update_trainee(agent) 
         
    torch.save(ac_agent, "models/" + model_name)
    envs.close()


def make_env(worker_id):
    print("Initializing worker", worker_id, "...")
    env = gym.make(env_name)
    return env


if __name__ == "__main__":
    main()
