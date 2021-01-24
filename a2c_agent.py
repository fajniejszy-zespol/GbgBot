import os
import gym
from ffai import FFAIEnv
from pytest import set_trace
from torch.autograd import Variable
import torch.optim as optim
from multiprocessing import Process, Pipe
from ffai.ai.layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys

# Architecture
model_name = 'FFAI-v2'
env_name = 'FFAI-v2'
model_filename = "models/" + model_name
log_filename = "logs/" + model_name + ".dat"


class CNNPolicy(nn.Module):

    def __init__(self, spatial_shape, non_spatial_inputs, hidden_nodes, kernels, actions, spatial_action_types,
                 non_spat_actions):
        super(CNNPolicy, self).__init__()

        # Spatial input stream
        self.num_convs = len(kernels)

        self.convs = [nn.Conv2d(in_channels=spatial_shape[0], out_channels=kernels[0][0], kernel_size=3, stride=1, padding=1)]

        #self.convs = [nn.Conv2d(in_channels=43, out_channels=32, kernel_size=3, stride=1, padding=1)]

        in_channels = kernels[0][0]

        for kernel in kernels[1:-1]:
            out_channels = kernel[0]
            kernel_size = kernel[1]
            padding = (kernel_size - 1) // 2
            assert out_channels > 0
            assert kernel_size in list(range(3, 27, 2))

            self.convs.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=1, padding=padding))

            in_channels = out_channels

        out_channels = spatial_action_types
        kernel_size = kernels[-1][1]
        padding = (kernel_size - 1) // 2
        self.convs.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=1, padding=padding))

        self.conv_non_spat1 = nn.Conv2d(in_channels=in_channels, out_channels=64,
                                        kernel_size=7, stride=1, padding=1)
        self.conv_non_spat2 = nn.Conv2d(in_channels=64, out_channels=32,
                                        kernel_size=7, stride=1, padding=1)

        stream_size = 2240  # assert False #TODO. How do calculate this dynamically for different pitch sizes.

        self.linear_non_spat = nn.Linear(stream_size, hidden_nodes)

        self.linear_non_spat_action = nn.Linear(hidden_nodes, non_spat_actions)

        # The outputs
        self.critic = nn.Linear(hidden_nodes, 1)
        self.outcome_pred = nn.Linear(hidden_nodes, 1)

        self.reset_parameters()

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')

        for conv in self.convs:
            conv.weight.data.mul_(relu_gain)
        self.conv_non_spat1.weight.data.mul_(relu_gain)
        self.conv_non_spat2.weight.data.mul_(relu_gain)

        self.linear_non_spat.weight.data.mul_(relu_gain)

        self.linear_non_spat_action.weight.data.mul_(relu_gain)
        self.critic.weight.data.mul_(relu_gain)
        self.outcome_pred.weight.data.mul_(relu_gain)

    def forward(self, spatial_input, non_spatial_input):
        """
        The forward functions defines how the data flows through the graph (layers)
        """
        # Spatial input the convolutional layers
        spat_x = spatial_input
        for conv in self.convs[:-1]:
            spat_x = F.relu(conv(spat_x))

        spat_actions = self.convs[-1](spat_x)
        spat_actions = spat_actions.flatten(start_dim=1)

        # Two conv layers and one linear - For game understanding
        non_spat_output = F.relu(self.conv_non_spat1(spat_x))
        non_spat_output = F.relu(self.conv_non_spat2(non_spat_output))

        non_spat_output = non_spat_output.flatten(start_dim=1)

        non_spat_output = F.relu(self.linear_non_spat(non_spat_output))


            # Non-spat actions
        non_spat_actions = self.linear_non_spat_action(non_spat_output)

        # Output streams


        value = self.critic(non_spat_output)
        actor = torch.cat((non_spat_actions, spat_actions), dim=1)
        outcome = self.outcome_pred(non_spat_output)

        # return value, policy
        return value, actor, outcome

    def act(self, spatial_inputs, non_spatial_input, action_mask):
        values, action_probs = self.get_action_probs(spatial_inputs, non_spatial_input, action_mask=action_mask)
        actions = action_probs.multinomial(1)
        return values, actions

    def evaluate_actions(self, spatial_inputs, non_spatial_input, actions, actions_mask):
        value, policy, pred = self(spatial_inputs, non_spatial_input)
        # actions_mask = actions_mask.view(-1, 1, actions_mask.shape[1]).squeeze().bool()
        policy[~actions_mask.bool()] = float('-inf')
        log_probs = F.log_softmax(policy, dim=1)
        probs = F.softmax(policy, dim=1)
        action_log_probs = log_probs.gather(1, actions)
        log_probs = torch.where(log_probs[None, :] == float('-inf'), torch.tensor(0.), log_probs)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, value, dist_entropy, pred

    def get_action_probs(self, spatial_input, non_spatial_input, action_mask):

        values, actions, pred = self(spatial_input, non_spatial_input)
        # Masking step: Inspired by: http://juditacs.github.io/2018/12/27/masked-attention.html
        if action_mask is not None:
            actions[~action_mask] = float('-inf')
        action_probs = F.softmax(actions, dim=1)
        return values, action_probs


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
        self.cnn_used = False

    def create_action_object(self, action_type, x=None, y=None):
        if action_type is None:
            return None

        if self.not_training:
            # position = Square(x, y) if action_type in FFAIEnv.positional_action_types else None
            # return ffai.Action(action_type, position=position, player=None)

            if action_type in FFAIEnv.positional_action_types:
                assert x is not None and y is not None
                return ffai.Action(action_type, position=Square(x, y), player=None)
            else:
                return ffai.Action(action_type, position=None, player=None)
        else:
            return {'action-type': action_type,
                    'x': x,
                    'y': y}

    def new_game(self, game, team):
        self.my_team = team
        self.is_home = self.my_team == game.state.home_team

    def _flip(self, board):
        flipped = {}
        for name, layer in board.items():
            flipped[name] = np.flip(layer, 1)
        return flipped

    def act(self, game, env=None):

        '''todo - Update code to get observations that are torch tensors. 
                - because they are either way converted for the policy optimization. 
        '''



        self.not_training = env is None


        if game is None:
            game = env.game


        if self.my_team is None:
            if self.not_training:
                assert False
            else:
                self.my_team = game.state.home_team

        self.cnn_used = False
        action = self.scripted_act(game)
        if action is not None:
            if self.not_training:
                return action
            else:
                return action, None, None, None, None, None

        self.cnn_used = True

        # Get observation
        if env is None:
            assert game is not None
            self.env.game = game
            env = self.env
        obs = env.get_observation()

        # Flip board observation if away team - we probably only trained as home team
        if not self.is_home:
            obs['board'] = self._flip(obs['board'])

        spatial_obs, non_spatial_obs = self._update_obs(obs)

        action_masks = self._compute_action_masks(obs)
        action_masks = torch.tensor(action_masks, dtype=torch.bool)

        values, actions = self.policy.act(
            Variable(spatial_obs.unsqueeze(0)),
            Variable(non_spatial_obs.unsqueeze(0)),
            Variable(action_masks.unsqueeze(0)))

        values.detach()
        # Create action from output
        action = actions[0]
        value = values[0]
        value.detach()
        action_type, x, y = self._compute_action(action.numpy()[0])

        # Flip position if playing as away and x>0 (x>0 means it's a positional action)
        if not self.is_home and x > 0:
            x = game.arena.width - 1 - x

        # Let's just end the setup right after picking a formation
        if action_type.name.lower().startswith('setup'):
            self.end_setup = True

        action_object = self.create_action_object(action_type, x, y)
        if self.not_training:
            return action_object
        else:
            return action_object, actions, action_masks, value, spatial_obs, non_spatial_obs

    def scripted_act(self, game):
        if self.end_setup:
            self.end_setup = False
            return self.create_action_object(ActionType.END_SETUP)

        available_action_types = [a.action_type for a in game.get_available_actions()]

        for a in [ActionType.STAND_UP, ActionType.USE_BRIBE, ActionType.START_GAME, ActionType.HEADS,
                  ActionType.RECEIVE]:
            if a in available_action_types:
                return self.create_action_object(a)

        if ActionType.PLACE_BALL in available_action_types:
            board_x_max = len(game.state.pitch.board[0]) - 2
            board_y_max = len(game.state.pitch.board) - 2

            if self.is_home:
                x = board_x_max // 4 + 1
            else:
                x = 3 * board_x_max // 4 + 1
            y = board_y_max // 2 + 1

            return self.create_action_object(ActionType.PLACE_BALL, x, y)

        proc = game.get_procedure()
        if isinstance(proc, Block):
            action_type = self.choose_block_dice(game, available_action_types)
            return self.create_action_object(action_type)

        return None

    def choose_block_dice(self, game, actions):

        # Block dice choice:
        # Get attacker and defender
        attacker = game.get_procedure().attacker
        defender = game.get_procedure().defender

        if attacker in self.my_team.players:

            # 1. DEFENDER DOWN
            if ActionType.SELECT_DEFENDER_DOWN in actions:
                return ActionType.SELECT_DEFENDER_DOWN

            if ActionType.SELECT_DEFENDER_STUMBLES in actions and not (
                    defender.has_skill(Skill.DODGE) and not attacker.has_skill(Skill.TACKLE)):
                return ActionType.SELECT_DEFENDER_STUMBLES

            if ActionType.SELECT_BOTH_DOWN in actions and not defender.has_skill(Skill.BLOCK) and attacker.has_skill(
                    Skill.BLOCK):
                return ActionType.SELECT_BOTH_DOWN

            # 2. No one down
            if ActionType.SELECT_DEFENDER_STUMBLES in actions:
                return ActionType.SELECT_DEFENDER_STUMBLES

            if ActionType.SELECT_PUSH in actions:
                return ActionType.SELECT_PUSH

            if ActionType.SELECT_BOTH_DOWN in actions and attacker.has_skill(Skill.BLOCK):
                return ActionType.SELECT_BOTH_DOWN

            # 3. We're going down!
            # If reroll available, ask the Neural Network
            if ActionType.USE_REROLL in actions:
                return None

            if ActionType.SELECT_BOTH_DOWN in actions:
                return ActionType.SELECT_BOTH_DOWN

            if ActionType.SELECT_ATTACKER_DOWN in actions:
                return ActionType.SELECT_ATTACKER_DOWN

        else:  # Opponent made uphill block.
            # 1. ATTACKER DOWN
            if ActionType.SELECT_ATTACKER_DOWN in actions:
                return ActionType.SELECT_ATTACKER_DOWN

            # 2. BOTH DOWN
            if ActionType.SELECT_BOTH_DOWN in actions and defender.has_skill(Skill.BLOCK):
                return ActionType.SELECT_BOTH_DOWN

            # 3. PUSH
            if ActionType.SELECT_PUSH in actions:
                return ActionType.SELECT_PUSH

            # 4. PUSH by Dodge
            if ActionType.SELECT_DEFENDER_STUMBLES in actions and defender.has_skill(
                    Skill.DODGE) and not attacker.has_skill(Skill.TACKLE):
                return ActionType.SELECT_DEFENDER_STUMBLES

            # OK, we're going down!
            # 5. BOTH DOWN
            if ActionType.SELECT_BOTH_DOWN in actions:
                return ActionType.SELECT_BOTH_DOWN

            if ActionType.SELECT_DEFENDER_STUMBLES in actions:
                return ActionType.SELECT_DEFENDER_STUMBLES

            if ActionType.SELECT_DEFENDER_DOWN in actions:
                return ActionType.SELECT_DEFENDER_DOWN

    def cnn_used_for_latest_action(self):
        return self.cnn_used

    def end_game(self, game):
        pass

    def _compute_action_masks(self, ob):
        mask = np.zeros(self.action_space)
        i = 0
        for action_type in self.non_spatial_action_types:
            mask[i] = ob['available-action-types'][action_type.name]
            i += 1
        for action_type in self.spatial_action_types:
            if ob['available-action-types'][action_type.name] == 0:
                mask[i:i + self.board_squares] = 0
            elif ob['available-action-types'][action_type.name] == 1:
                position_mask = ob['board'][f"{action_type.name.replace('_', ' ').lower()} positions"]
                position_mask_flatten = np.reshape(position_mask, (1, self.board_squares))
                for j in range(self.board_squares):
                    mask[i + j] = position_mask_flatten[0][j]
            i += self.board_squares
        assert 1 in mask
        return mask

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

        print("Starting game", (i + 1))
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

    print(f"Home/Draws/Away: {wins}/{draws}/{n - wins - draws}")
    print(f"Home TDs per game: {tds_home / n}")
    print(f"Away TDs per game: {tds_away / n}")
