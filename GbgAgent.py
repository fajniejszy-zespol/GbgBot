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

from pytest import set_trace 

# Architecture
model_name = 'FFAI-v2'
env_name = 'FFAI-v2'
model_filename = "models/" + model_name
log_filename = "logs/" + model_name + ".dat"


class CNNPolicy(nn.Module):
    def __init__(self, spatial_shape, non_spatial_inputs, hidden_nodes, kernels, actions, spatial_action_types, non_spat_actions):
        super(CNNPolicy, self).__init__()
        
        # Spatial input stream
        self.conv1 = nn.Conv2d(spatial_shape[0],        out_channels=kernels[0],            kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=kernels[0],  out_channels=kernels[1],            kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=kernels[1],  out_channels=kernels[2],            kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=kernels[2],  out_channels=spatial_action_types,  kernel_size=7, stride=1, padding=3)
        
        # Non-spatial input stream
        self.linear0 = nn.Linear(non_spatial_inputs, non_spatial_inputs*2)

        # Linear layers
        stream_size = kernels[2] * spatial_shape[1] * spatial_shape[2]
        stream_size += non_spatial_inputs*2
        self.linear1 = nn.Linear(stream_size, hidden_nodes)

        # The outputs
        self.non_spat_actor = nn.Linear(hidden_nodes, non_spat_actions) 
        
        # Critic stream 
        critic_stream_size = actions + stream_size 
        self.critic1 = nn.Linear(critic_stream_size, hidden_nodes)
        self.critic2 = nn.Linear(hidden_nodes, 1)
        
        
        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        print("parameters reset")
        relu_gain = nn.init.calculate_gain('relu')
        
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.conv4.weight.data.mul_(relu_gain)
        
        self.linear0.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)
        self.non_spat_actor.weight.data.mul_(relu_gain)
        self.critic1.weight.data.mul_(relu_gain)
        self.critic2.weight.data.mul_(relu_gain)

    def forward(self, spatial_input, non_spatial_input):
        """
        The forward functions defines how the data flows through the graph (layers)
        """
        # Spatial input through convolutional layers
        x1 = F.relu(self.conv1(spatial_input))
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        
        spat_actions = self.conv4(x1)
        spat_actions = spat_actions.flatten(start_dim=1)
        
        #Non spatial input
        x2 = self.linear0(non_spatial_input)
        x2 = F.relu(x2)
        
        # Concatenate the input streams for non spat actions 
        flatten_x1 = x1.flatten(start_dim=1)
        flatten_x2 = x2.flatten(start_dim=1)
        concatenated = torch.cat( (flatten_x1, flatten_x2), dim=1)
        
        # Fully-connected layers
        x3 = self.linear1(concatenated)
        x3 = F.relu(x3)
        no_spat_actions = self.non_spat_actor(x3) 
        
        # Output actions         
        actor = torch.cat( (no_spat_actions, spat_actions) , dim=1) 
        
        #Calculate the critic
        x_critic_stream = torch.cat( (actor, concatenated), dim=1) 
        
        x_critic_stream = self.critic1(x_critic_stream)
        x_critic_stream = F.relu(x_critic_stream)
        value = self.critic2(x_critic_stream)
        
        return value, actor

    def act(self, spatial_inputs, non_spatial_input, action_mask):
        values, action_probs = self.get_action_probs(spatial_inputs, non_spatial_input, action_mask=action_mask)
        actions = action_probs.multinomial(1)
        return values, actions

    def evaluate_actions(self, spatial_inputs, non_spatial_input, actions, actions_mask):
        value, policy = self(spatial_inputs, non_spatial_input)
        actions_mask = actions_mask.view(-1, 1, actions_mask.shape[2]).squeeze().bool()
        policy[~actions_mask] = float('-inf')
        log_probs = F.log_softmax(policy, dim=1)
        probs = F.softmax(policy, dim=1)
        action_log_probs = log_probs.gather(1, actions)
        log_probs = torch.where(log_probs[None, :] == float('-inf'), torch.tensor(0.), log_probs)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, value, dist_entropy

    def get_action_probs(self, spatial_input, non_spatial_input, action_mask):
        values, actions = self(spatial_input, non_spatial_input)
        # Masking step: Inspired by: http://juditacs.github.io/2018/12/27/masked-attention.html
        if action_mask is not None:
            actions[~action_mask] = float('-inf')
        action_probs = F.softmax(actions, dim=1)
        return values, action_probs

        
def update_obs(observations):
    """
    Takes the observation returned by the environment and transforms it to an numpy array that contains all of
    the feature layers and non-spatial info
    """
    spatial_obs = []
    non_spatial_obs = []

    for obs in observations:
        '''
        for k, v in obs['board'].items():
            print(k)
            print(v)
        '''
        spatial_ob = np.stack(obs['board'].values())

        state = list(obs['state'].values())
        procedures = list(obs['procedures'].values())
        actions = list(obs['available-action-types'].values())
        

        #set_trace() 
        non_spatial_ob = np.stack(state+procedures+actions)
        
        # feature_layers = np.expand_dims(feature_layers, axis=0)
        non_spatial_ob = np.expand_dims(non_spatial_ob, axis=0)

        spatial_obs.append(spatial_ob)
        non_spatial_obs.append(non_spatial_ob)

    return torch.from_numpy(np.stack(spatial_obs)).float(), torch.from_numpy(np.stack(non_spatial_obs)).float()



class A2CAgent(Agent):

    def __init__(self, name, env_name=env_name, filename=model_filename):
        super().__init__(name)
        self.my_team = None
        self.env = self.make_env(env_name)

        self.spatial_obs_space = self.env.observation_space.spaces['board'].shape
        self.board_dim = (self.spatial_obs_space[1], self.spatial_obs_space[2])
        self.board_squares = self.spatial_obs_space[1] * self.spatial_obs_space[2]

        self.non_spatial_obs_space = self.env.observation_space.spaces['state'].shape[0] + \
                                self.env.observation_space.spaces['procedures'].shape[0] + \
                                self.env.observation_space.spaces['available-action-types'].shape[0]
        self.non_spatial_action_types = FFAIEnv.simple_action_types + FFAIEnv.defensive_formation_action_types + FFAIEnv.offensive_formation_action_types
        self.num_non_spatial_action_types = len(self.non_spatial_action_types)
        self.spatial_action_types = FFAIEnv.positional_action_types
        self.num_spatial_action_types = len(self.spatial_action_types)
        self.num_spatial_actions = self.num_spatial_action_types * self.spatial_obs_space[1] * self.spatial_obs_space[2]
        self.action_space = self.num_non_spatial_action_types + self.num_spatial_actions
        self.is_home = True

        # MODEL
        self.policy = torch.load(filename)
        self.policy.eval()
        self.end_setup = False

    def new_game(self, game, team):
        self.my_team = team
        self.is_home = self.my_team == game.state.home_team

    def _flip(self, board):
        flipped = {}
        for name, layer in board.items():
            flipped[name] = np.flip(layer, 1)
        return flipped

    def act(self, game):

        if self.end_setup:
            self.end_setup = False
            return ffai.Action(ActionType.END_SETUP)

        # Get observation
        self.env.game = game
        observation = self.env.get_observation()

        # Flip board observation if away team - we probably only trained as home team
        if not self.is_home:
            observation['board'] = self._flip(observation['board'])

        obs = [observation]
        spatial_obs, non_spatial_obs = self._update_obs(obs)

        action_masks = self._compute_action_masks(obs)
        action_masks = torch.tensor(action_masks, dtype=torch.bool)

        values, actions = self.policy.act(
            Variable(spatial_obs),
            Variable(non_spatial_obs),
            Variable(action_masks))

        # Create action from output
        action = actions[0]
        value = values[0]
        action_type, x, y = self._compute_action(action.numpy()[0])
        position = Square(x, y) if action_type in FFAIEnv.positional_action_types else None

        # Flip position
        if not self.is_home and position is not None:
            position = Square(game.arena.width - 1 - position.x, position.y)

        action = ffai.Action(action_type, position=position, player=None)

        # Let's just end the setup right after picking a formation
        if action_type.name.lower().startswith('setup'):
            self.end_setup = True

        # Return action to the framework
        return action

    def end_game(self, game):
        pass

    def _compute_action_masks(self, observations):
        masks = []
        m = False
        for ob in observations:
            mask = np.zeros(self.action_space)
            i = 0
            for action_type in self.non_spatial_action_types:
                mask[i] = ob['available-action-types'][action_type.name]
                i += 1
            for action_type in self.spatial_action_types:
                if ob['available-action-types'][action_type.name] == 0:
                    mask[i:i+self.board_squares] = 0
                elif ob['available-action-types'][action_type.name] == 1:
                    position_mask = ob['board'][f"{action_type.name.replace('_', ' ').lower()} positions"]
                    position_mask_flatten = np.reshape(position_mask, (1, self.board_squares))
                    for j in range(self.board_squares):
                        mask[i + j] = position_mask_flatten[0][j]
                i += self.board_squares
            assert 1 in mask
            if m:
                print(mask)
            masks.append(mask)
        return masks

    def _compute_action(self, action_idx):
        if action_idx < len(self.non_spatial_action_types):
            return self.non_spatial_action_types[action_idx], 0, 0
        spatial_idx = action_idx - self.num_non_spatial_action_types
        spatial_pos_idx = spatial_idx % self.board_squares
        spatial_y = int(spatial_pos_idx / self.board_dim[1])
        spatial_x = int(spatial_pos_idx % self.board_dim[1])
        spatial_action_type_idx = int(spatial_idx / self.board_squares)
        spatial_action_type = self.spatial_action_types[spatial_action_type_idx]
        return spatial_action_type, spatial_x, spatial_y

    def _update_obs(self, observations):
        return update_obs(observations)

    def make_env(self, env_name):
        env = gym.make(env_name)
        return env


# Register the bot to the framework
ffai.register_bot('my-a2c-bot', A2CAgent)

'''
import ffai.web.server as server

if __name__ == "__main__":
    server.start_server(debug=True, use_reloader=False)
'''

if __name__ == "__main__":

    # Load configurations, rules, arena and teams
    config = ffai.load_config("ff-1")
    config.competition_mode = False
    ruleset = ffai.load_rule_set(config.ruleset)
    arena = ffai.load_arena(config.arena)
    home = ffai.load_team_by_filename("human", ruleset)
    away = ffai.load_team_by_filename("human", ruleset)
    config.competition_mode = False
    config.debug_mode = False

    # Play 100 games
    game_times = []
    wins = 0
    draws = 0
    n = 100
    is_home = True
    tds_away = 0
    tds_home = 0
    for i in range(n):

        if is_home:
            away_agent = ffai.make_bot('random')
            home_agent = ffai.make_bot('my-a2c-bot')
        else:
            away_agent = ffai.make_bot('my-a2c-bot')
            home_agent = ffai.make_bot("random")
        game = ffai.Game(i, home, away, home_agent, away_agent, config, arena=arena, ruleset=ruleset)
        game.config.fast_mode = True

        print("Starting game", (i+1))
        game.init()
        print("Game is over")

        winner = game.get_winner()
        if winner is None:
            draws += 1
        elif winner == home_agent and is_home:
            wins += 1
        elif winner == away_agent and not is_home:
            wins += 1

        tds_home += game.get_agent_team(home_agent).state.score
        tds_away += game.get_agent_team(away_agent).state.score

    print(f"Home/Draws/Away: {wins}/{draws}/{n-wins-draws}")
    print(f"Home TDs per game: {tds_home/n}")
    print(f"Away TDs per game: {tds_away/n}")
