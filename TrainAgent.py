import os
import gym
from ffai import FFAIEnv
from torch.autograd import Variable
import torch.optim as optim
from multiprocessing import Process, Pipe 
from ffai.ai.layers import *
#import ffai.ai.pathfinding as pf 
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
from GbgAgent import A2CAgent
import ffai
import random
from pdb import set_trace

import Curriculum as gc
from GbgAgent import CNNPolicy, update_obs

debug_thread = False 

# Training configuration
#num_steps = 10000000
learning_rate = 0.001 #0.001
gamma = 0.95
entropy_coef = 0.01
value_loss_coef = 0.5
max_grad_norm = 0.05
reset_steps = 20000  # The environment is reset after this many steps it gets stuck

# Environment
#env_name = "FFAI-7-v2"
env_name = "FFAI-v2"

num_processes = 10
match_processes = 6
num_steps = 10000000
steps_per_update = 60

log_interval = 50 
save_interval = 1000

ppcg = False 

# Self-play
selfplay = False   # Use this to enable/disable self-play
selfplay_window = 8
selfplay_save_steps = int(num_steps / 25)
selfplay_swap_steps = selfplay_save_steps

# Architecture
num_hidden_nodes = 512
num_cnn_kernels = [48, 32, 24]

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
    #Scoring 
    OutcomeType.TOUCHDOWN:          2,
    OutcomeType.REROLL_USED:        -0.05, #to discourage unnecessary re-rolling 
    
    #Ball handling 
    OutcomeType.CATCH:              0.0,
    OutcomeType.INTERCEPTION:       0.2,
    OutcomeType.SUCCESSFUL_PICKUP:  0.1,
    OutcomeType.FUMBLE:            -0.3,
    OutcomeType.FAILED_CATCH:      -0.1, 
    OutcomeType.INACCURATE_PASS:   -0.1,
    
    #Fighting 
    OutcomeType.KNOCKED_DOWN:      -0.1, #always reported when knocked down. Add Stun/KO/cas after
    OutcomeType.STUNNED:           -0.1, 
    OutcomeType.KNOCKED_OUT:       -0.2,
    OutcomeType.CASUALTY:          -0.5,
    OutcomeType.PLAYER_EJECTED:    -0.5, 
    OutcomeType.PUSHED_INTO_CROWD: -0.25, 
    
}
rewards_opp = {
    #Scoring 
    OutcomeType.TOUCHDOWN:         -2,
    
    #Ball handling 
    OutcomeType.CATCH:             -0.0,
    OutcomeType.INTERCEPTION:      -0.2,
    OutcomeType.SUCCESSFUL_PICKUP: -0.1,
    OutcomeType.FUMBLE:             0.5,#0.1,
    OutcomeType.FAILED_CATCH:       0.1,
    OutcomeType.INACCURATE_PASS:    0.1,
    OutcomeType.TOUCHBACK:         -0.4,
    
    #Fighting 
    OutcomeType.KNOCKED_DOWN:       0.1, #always reported when knocked down. Add Stun/KO/cas after
    OutcomeType.STUNNED:            0.1,
    OutcomeType.KNOCKED_OUT:        0.2,
    OutcomeType.CASUALTY:           0.5,
    OutcomeType.PUSHED_INTO_CROWD:  0.25, 
    OutcomeType.PLAYER_EJECTED:     0.5,
    
}
ball_progression_reward = 0.005


class Memory(object):
    def __init__(self, steps_per_update, num_processes, spatial_obs_shape, non_spatial_obs_shape, action_space):
        self.spatial_obs = torch.zeros(steps_per_update + 1, num_processes, *spatial_obs_shape)
        self.non_spatial_obs = torch.zeros(steps_per_update + 1, num_processes, *non_spatial_obs_shape)
        self.rewards = torch.zeros(steps_per_update, num_processes, 1)
        self.value_predictions = torch.zeros(steps_per_update + 1, num_processes, 1)
        self.returns = torch.zeros(steps_per_update + 1, num_processes, 1)
        action_shape = 1
        self.actions = torch.zeros(steps_per_update, num_processes, action_shape)
        self.actions = self.actions.long()
        self.masks = torch.ones(steps_per_update + 1, num_processes, 1)
        self.action_masks = torch.zeros(steps_per_update + 1, num_processes, action_space, dtype=torch.uint8)

    def cuda(self):
        self.spatial_obs = self.spatial_obs.cuda()
        self.non_spatial_obs = self.non_spatial_obs.cuda()
        self.rewards = self.rewards.cuda()
        self.value_predictions = self.value_predictions.cuda()
        self.returns = self.returns.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()
        self.action_masks = self.action_masks.cuda()

    def insert(self, step, spatial_obs, non_spatial_obs, action, value_pred, reward, mask, action_masks):
        self.spatial_obs[step + 1].copy_(spatial_obs)
        self.non_spatial_obs[step + 1].copy_(non_spatial_obs)
        self.actions[step].copy_(action)
        self.value_predictions[step].copy_(value_pred)
        self.rewards[step].copy_(reward)
        self.masks[step].copy_(mask)
        self.action_masks[step].copy_(action_masks)

    def compute_returns(self, next_value, gamma):
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.size(0))):
            self.returns[step] = self.returns[step + 1] * gamma * self.masks[step] + self.rewards[step]

    
def reward_function(env, info, shaped=False, obs=None, prev_super_shaped=None, debug=False ):
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
    
    super_shaped = 0
    if obs is not None: 
        super_shaped = 0
        
        # Reward players and tz the 5-by-5 square with ball in middle 
        if True: 
            b = env.game.get_ball()
            if b is None: 
                return r, None
            
            x = b.position.x
            y = b.position.y 
            
            x_max = len(env.game.state.pitch.board[0]) -2  
            y_max = len(env.game.state.pitch.board) -2
            
            dx_neg = min(2, x-1) 
            dx_pos = min(2, x_max-x)
            dy_neg = min(2, y-1)
            dy_pos = min(2, y_max-y)
            
            x_start = x - dx_neg
            x_end   = x + dx_pos +1
            y_start = y - dy_neg
            y_end   = y + dy_pos +1
            
            
            
            player_in_ballzone_reward = 0.1 # min arbitrated with 6
            tz_ballzone_reward = 0.01
            tz_on_ball = 0.1 #min arbitrated with 4 
            
            # Opp players in ball zone
            opp_layer = obs['board']["opp players"][y_start:y_end , x_start:x_end ]
            up_layer = obs['board']["standing players"][ y_start:y_end , x_start:x_end ]
            opp_close_to_ball = np.multiply( opp_layer, up_layer).sum() 
            opp_close_to_ball = min( opp_close_to_ball, 6) * player_in_ballzone_reward
            #REWARD - Opp players in ball zone  
            super_shaped -= opp_close_to_ball
            
            # Opp tz in ball zone 
            opp_tz_layer = obs['board']["opp tackle zones"][ y_start:y_end , x_start:x_end ]
            opp_nbr_of_tz = (opp_tz_layer>0).sum()   
            opp_nbr_of_tz *= tz_ballzone_reward
            #REWARD - Opp tz in ball zone  
            super_shaped -= opp_nbr_of_tz 
            
            opp_tz_ball = obs['board']["opp tackle zones"][ y,x] / 0.125
            opp_tz_ball = min(opp_tz_ball, 4) * tz_on_ball
            #REWARD - Opp TZ on ball 
            super_shaped -= opp_tz_ball
            if debug: print("Reward - Away tz: {} - {} -  {} ".format(opp_close_to_ball, opp_nbr_of_tz, opp_tz_ball ) )
            
            # Own players in ball zone
            own_layer = obs['board']["own players"][y_start:y_end , x_start:x_end ]
            own_in_ballzone = np.multiply( own_layer, up_layer).sum() 
            own_in_ballzone = min(own_in_ballzone, 6) * player_in_ballzone_reward
            #REWARD - Own players in ball zone  
            super_shaped += own_in_ballzone
            
            # Own tz in ball zone 
            own_tz_layer = obs['board']["own tackle zones"][ y_start:y_end , x_start:x_end ]
            own_tz_ballzone = (own_tz_layer>0).sum()
            own_tz_ballzone *= tz_ballzone_reward
            #REWARD - Own tz in ball zone  
            super_shaped += own_tz_ballzone
            
            own_tz_ball = obs['board']["own tackle zones"][ y,x] / 0.125
            own_tz_ball = min(own_tz_ball, 4) * tz_on_ball 
            #REWARD - Own TZ on ball 
            super_shaped += own_tz_ball 
            
            if debug: print("Reward - Home tz: {} - {} - {}".format(own_in_ballzone, own_tz_ballzone, own_tz_ball)   )
            
        # Reward for having two scoring threats 
        if True: 
            home_players = gc.get_home_players(env.game) 
            away_players = gc.get_away_players(env.game)
            
            home_score_threat = 0
            away_score_threat = 0 
            
            board_x_max = len(env.game.state.pitch.board[0]) -2  
            
            for player in home_players: 
                moves_to_td = player.position.x  -1 
                tz = env.game.num_tackle_zones_in(player)
                
                tz = 1 - max(tz,1)/2 
                home_score_threat += ( moves_to_td <= player.get_ma() ) * tz  

            for player in away_players: 
                moves_to_td = board_x_max - player.position.x  
                tz = env.game.num_tackle_zones_in(player)
                
                tz = 1 - max(tz,1)/2 
                away_score_threat += ( moves_to_td <= player.get_ma() ) * tz  


              
            home_score_threat = max(home_score_threat, 2)    * 0.3
            away_score_threat = max(home_score_threat, 2)    * 0.3
            
            
            super_shaped += home_score_threat
            super_shaped -= away_score_threat
        
            if debug: print("Reward scoring threat, home vs. away: {} - {}".format(home_score_threat, away_score_threat))
        
        # Reward screening 
        if True: 
            screening_reward = 0.3
            
            away_ps = gc.get_away_players(env.game)
            home_players = gc.get_home_players(env.game)
            if env.game.get_ball_carrier() in home_players or len(away_ps)==0: 
                unmarked_tz_reward = screening_reward #Give max if screen is not needed. 
                
            else: 
                #Only consider relevant tacklezones between ball and own td 
                
                away_mean_pos = np.mean( [ [p.position.x, p.position.y] for p in away_ps], axis=0)
                away_mean_x = int( round(away_mean_pos[0]) ) 
                away_mean_y = int( round(away_mean_pos[1]) )
                
                tz = obs['board']["own unmarked tackle zones"][:, :away_mean_x] > 0
                
                covered_lanes = (np.multiply( tz[:-1], tz[1:]).sum(axis=1)>0).astype(float)
                
                weight = np.array( [1,1,1,1,.8,.6, .4, .1, 0,0,0,0,0,0,0,0,0,0,0] )

                weight_mask = np.zeros( covered_lanes.shape ) 
                weight_mask[ away_mean_y::-1] = weight[:away_mean_y+1] 
                weight_mask[ away_mean_y:] = weight[: len(weight_mask[ away_mean_y:]) ] 
                
                #REWARD - Screening with unmarked tacklezones 
                unmarked_tz_reward = np.dot( covered_lanes, weight_mask )  
                
                normalization = np.ones( covered_lanes.shape) 
                unmarked_tz_reward /= np.dot( normalization, weight_mask ) 
                unmarked_tz_reward *= screening_reward
                if debug: print("unmarked single tz screen: {}".format( unmarked_tz_reward ) ) 
            
            
            super_shaped += unmarked_tz_reward
            
            
        
        if prev_super_shaped is not None:
            r += super_shaped - prev_super_shaped 
    
    return r, super_shaped

def worker(remote, parent_remote, env, worker_id):
    parent_remote.close()

    steps = 0
    tds = 0
    tds_opp = 0
    next_opp = ffai.make_bot('random')

    prev_super_shaped = None
    
    while True:

        if debug_thread: print(f"worker {worker_id} - stuck on receive")
                        
        command, data = remote.recv()
        if debug_thread: print(f"worker {worker_id} - after receive: '{command}'")
        if command == 'step':
            steps += 1
            action, dif, lecture = data[0], data[1], data[2]
            
            
           # s = "in worker, action is " + str(action)
            #print(s) 
            

            obs, reward, done, info = env.step(action)
            tds_scored = info['touchdowns'] - tds
            tds = info['touchdowns']
            tds_opp_scored = info['opp_touchdowns'] - tds_opp
            tds_opp = info['opp_touchdowns']
            reward_shaped, prev_super_shaped = reward_function(env, info, shaped=True, obs=obs, prev_super_shaped = prev_super_shaped)
            ball_carrier = env.game.get_ball_carrier()
            # PPCG
            if dif < 1.0 and env.lecture is None:
                if ball_carrier and ball_carrier.team == env.game.state.home_team:
                    extra_endzone_squares = int((1.0 - dif) * 25.0)
                    distance_to_endzone = ball_carrier.position.x - 1
                    if distance_to_endzone <= extra_endzone_squares:
                        #reward_shaped += rewards_own[OutcomeType.TOUCHDOWN]
                        env.game.state.stack.push(Touchdown(env.game, ball_carrier))
            if done or steps >= reset_steps:
                # If we  get stuck or something - reset the environment
                if steps >= reset_steps:
                    print("Max. number of steps exceeded! Consider increasing the number.")
                done = True
                env.opp_actor = next_opp
                obs = env.reset(lecture)
                steps = 0
                tds = 0
                tds_opp = 0
            
            if debug_thread: print(f"worker {worker_id} - stuck on send") 
            remote.send((obs, reward, reward_shaped, tds_scored, tds_opp_scored, done, info))
            if debug_thread: print(f"worker {worker_id}  - stuck somewhere else")

        elif command == 'reset':
            dif, lecture = data[0], data[1]
            steps = 0
            tds = 0
            tds_opp = 0
            env.opp_actor = next_opp
            prev_super_shaped = None 
            obs = env.reset(lecture)
            # set_difficulty(env, dif)
            remote.send(obs)
        elif command == 'render':
            #env.render()
            pass 
        elif command == 'swap':
            next_opp = data
        elif command == 'close':
            break

            
class VecEnv():
    def __init__(self, envs):
        """
        envs: list of FFAI environments to run in subprocesses
        """
        self.closed = False
        nenvs = len(envs)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        self.ps = [Process(target=worker, args=(work_remote, remote, env, envs.index(env)))
                   for (work_remote, remote, env) in zip(self.work_remotes, self.remotes, envs)]

        for p in self.ps:
            p.daemon = True  # If the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions, difficulty=1.0, lectures = None ):
        cumul_rewards = None
        cumul_shaped_rewards = None
        cumul_tds_scored = None
        cumul_tds_opp_scored = None
        cumul_dones = None
        #set_trace() 
        if lectures == None: 
            lectures = [None] * len(self.remotes)
        
        for remote, action, lecture in zip(self.remotes, actions, lectures):
            remote.send(('step', [action, difficulty, lecture]))

        results = [remote.recv() for remote in self.remotes]
        
        obs, rews, rews_shaped, tds, tds_opp, dones, infos = zip(*results)
        if cumul_rewards is None:
            cumul_rewards = np.stack(rews)
            cumul_shaped_rewards = np.stack(rews_shaped)
            cumul_tds_scored = np.stack(tds)
            cumul_tds_opp_scored = np.stack(tds_opp)
        else:
            cumul_rewards += np.stack(rews)
            cumul_shaped_rewards += np.stack(rews_shaped)
            cumul_tds_scored += np.stack(tds)
            cumul_tds_opp_scored += np.stack(tds_opp)
        if cumul_dones is None:
            cumul_dones = np.stack(dones)
        else:
            cumul_dones |= np.stack(dones)
        return np.stack(obs), cumul_rewards, cumul_shaped_rewards, cumul_tds_scored, cumul_tds_opp_scored, cumul_dones, infos

    def reset(self, difficulty=1.0, lectures = None ):
        if lectures == None: lectures = [None for _ in range(len(self.remotes))]
        
        for remote, lecture in zip(self.remotes, lectures):
            remote.send(('reset', [difficulty, lecture] ))
        return np.stack([remote.recv() for remote in self.remotes])

    def render(self):
        for remote in self.remotes:
            remote.send(('render', None))

    def swap(self, agent):
        for remote in self.remotes:
            remote.send(('swap', agent))

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

        
def main():
    if True: #GbgBot config 
        #academy = gc.Academy( [gc.CrowdSurf(), gc.BlockBallCarrier(), gc.PickupAndScore(), gc.Scoring(), gc.HandoffAndScore()] )
        planned_lectures = [gc.Scoring(), 
                            gc.PassAndScore(handoff=True), 
                            gc.PassAndScore(handoff=False), 
                            gc.PickupAndScore(), 
                            gc.BlockBallCarrier(),
                            #gc.CrowdSurf(), 
                            gc.ChooseBlockDie()
                            ]
        academy = gc.Academy( planned_lectures , num_processes,    match_processes=match_processes )
        
        lectures = academy.get_next_lectures( num_processes )
        
    es = [make_env(i) for i in range(num_processes)]
    

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

    def compute_action_masks(observations):
        masks = []
        m = False
        for ob in observations:
            mask = np.zeros(action_space)
            i = 0
            for action_type in non_spatial_action_types:
                mask[i] = ob['available-action-types'][action_type.name]
                i += 1
            for action_type in spatial_action_types:
                if ob['available-action-types'][action_type.name] == 0:
                    mask[i:i+board_squares] = 0
                elif ob['available-action-types'][action_type.name] == 1:
                    position_mask = ob['board'][f"{action_type.name.replace('_', ' ').lower()} positions"]
                    position_mask_flatten = np.reshape(position_mask, (1, board_squares))
                    for j in range(board_squares):
                        mask[i + j] = position_mask_flatten[0][j]
                i += board_squares
            try: 
                assert 1 in mask
            except: 
                print("assert 1 in mask")
                pdb.set_trace()
                exit() 
            #if ob["procedures"]["PlaceBall"]>0: 
            #    print("tracing placeball")
            #    pdb.set_trace()
            #    exit() 
            
            if m:
                print(mask)
            masks.append(mask)
        return masks

    def compute_action(action_idx):
        if action_idx < len(non_spatial_action_types):
            return non_spatial_action_types[action_idx], 0, 0
        spatial_idx = action_idx - num_non_spatial_action_types
        spatial_pos_idx = spatial_idx % board_squares
        spatial_y = int(spatial_pos_idx / board_dim[1])
        spatial_x = int(spatial_pos_idx % board_dim[1])
        spatial_action_type_idx = int(spatial_idx / board_squares)
        spatial_action_type = spatial_action_types[spatial_action_type_idx]
        return spatial_action_type, spatial_x, spatial_y

    # Clear log file
    try:
        os.remove(log_filename)
    except OSError:
        pass

    try: 
        os.remove("logs/Gbg_log.txt")
    except OSError:
        pass    
    # MODEL
    try: 
        ac_agent = torch.load("models/" + model_name)
        print("load successful")
    except: 
       ac_agent = CNNPolicy(spatial_obs_space, non_spatial_obs_space, hidden_nodes=num_hidden_nodes, kernels=num_cnn_kernels, actions=action_space, spatial_action_types = num_spatial_action_types, non_spat_actions=num_non_spatial_action_types)
       print("load failed - new agent created")
        
    #exit()
    
    envs = VecEnv([es[i] for i in range(num_processes)])
    
    # OPTIMIZER
    optimizer = optim.RMSprop(ac_agent.parameters(), learning_rate)

    # MEMORY STORE
    memory = Memory(steps_per_update, num_processes, spatial_obs_space, (1, non_spatial_obs_space), action_space)

    # PPCG
    difficulty = 0.0
    dif_delta = 0.01

    # Reset environments
    obs = envs.reset(difficulty, lectures)
    spatial_obs, non_spatial_obs = update_obs(obs)

     
    # Add obs to memory
    memory.spatial_obs[0].copy_(spatial_obs)
    memory.non_spatial_obs[0].copy_(non_spatial_obs)

    if True: # Variables for storing stats
        all_updates = 0
        all_episodes = 0
        all_steps = 0
        episodes = 0
        proc_rewards = np.zeros(num_processes)
        proc_tds = np.zeros(num_processes)
        proc_tds_opp = np.zeros(num_processes)
        episode_rewards = []
        episode_tds = []
        episode_tds_opp = []
        wins = []
        #value_losses = []
        #policy_losses = []
        log_updates = []
        log_episode = []
        log_steps = []
        log_win_rate = []
        log_td_rate = []
        log_td_rate_opp = []
        log_mean_reward = []
        log_difficulty = []

    # self-play
    selfplay_next_save = selfplay_save_steps
    selfplay_next_swap = selfplay_swap_steps
    selfplay_models = 0
    if selfplay:
        model_path = f"models/{model_name}_selfplay_0"
        torch.save(ac_agent, model_path)
        envs.swap(A2CAgent(name=f"selfplay-0", env_name=env_name, filename=model_path))
        selfplay_models += 1

    print("Training started!")
    while all_steps < num_steps:
        
        
        
        for step in range(steps_per_update):

            action_masks = compute_action_masks(obs)
            action_masks = torch.tensor(action_masks, dtype=torch.bool)

            values, actions = ac_agent.act(
                Variable(memory.spatial_obs[step]),
                Variable(memory.non_spatial_obs[step]),
                Variable(action_masks))

            action_objects = []

            for action in actions:
                action_type, x, y = compute_action(action.numpy()[0])
                action_object = {
                    'action-type': action_type,
                    'x': x,
                    'y': y
                }
                action_objects.append(action_object)

            lectures = academy.get_next_lectures( num_processes )
            obs, env_reward, shaped_reward, tds_scored, tds_opp_scored, done, info = envs.step(action_objects, difficulty=difficulty,lectures=lectures)
            
            #print(f"Step - {all_steps + step}" ) 
            
            reward = torch.from_numpy(np.expand_dims(np.stack(env_reward), 1)).float()
            shaped_reward = torch.from_numpy(np.expand_dims(np.stack(shaped_reward), 1)).float()
            r = reward.numpy()
            sr = shaped_reward.numpy()
            for i in range(num_processes):
                if lectures[i] == None:  
                    proc_rewards[i] += sr[i]
                    proc_tds[i] += tds_scored[i]
                    proc_tds_opp[i] += tds_opp_scored[i]

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ or info_["reset_reward"] else [1.0] for done_, info_ in zip(done, info) ])
            dones = masks.squeeze()
            #episodes += num_processes - int(dones.sum().item())
            for i in range(num_processes):
                if done[i]:
                    if "lecture" in info[i].keys(): 
                        academy.log_training( info[i]["lecture"])
                        proc_rewards[i] = 0
                        proc_tds[i] = 0
                        proc_tds_opp[i] = 0
                    else: 
                        episodes += 1 
                        if r[i] > 0:
                            wins.append(1)
                            difficulty += dif_delta
                        elif r[i] < 0:
                            wins.append(0)
                            difficulty -= dif_delta
                        else:
                            wins.append(0.5)
                            difficulty -= dif_delta
                        if ppcg:
                            difficulty = min(1.0, max(0, difficulty))
                        else:
                            difficulty = 1
                        
                        episode_rewards.append(proc_rewards[i])
                        episode_tds.append(proc_tds[i])
                        episode_tds_opp.append(proc_tds_opp[i])
                        proc_rewards[i] = 0
                        proc_tds[i] = 0
                        proc_tds_opp[i] = 0

            # Update the observations returned by the environment
            spatial_obs, non_spatial_obs = update_obs(obs)

            # insert the step taken into memory
            memory.insert(step, spatial_obs, non_spatial_obs,
                          actions.data, values.data, shaped_reward, masks, action_masks)

        #print( academy.report_training() )
        
        next_value = ac_agent(Variable(memory.spatial_obs[-1], requires_grad=False), Variable(memory.non_spatial_obs[-1], requires_grad=False))[0].data

        # Compute returns
        memory.compute_returns(next_value, gamma)

        spatial = Variable(memory.spatial_obs[:-1])
        spatial = spatial.view(-1, *spatial_obs_space)
        non_spatial = Variable(memory.non_spatial_obs[:-1])
        non_spatial = non_spatial.view(-1, non_spatial.shape[-1])

        actions = Variable(torch.LongTensor(memory.actions.view(-1, 1)))
        actions_mask = Variable(memory.action_masks[:-1])

        # Evaluate the actions taken
        action_log_probs, values, dist_entropy = ac_agent.evaluate_actions(spatial, non_spatial, actions, actions_mask)

        values = values.view(steps_per_update, num_processes, 1)
        action_log_probs = action_log_probs.view(steps_per_update, num_processes, 1)

        advantages = Variable(memory.returns[:-1]) - values
        value_loss = advantages.pow(2).mean()
        #value_losses.append(value_loss)

        # Compute loss
        action_loss = -(Variable(advantages.data) * action_log_probs).mean()
        #policy_losses.append(action_loss)

        optimizer.zero_grad()

        total_loss = (value_loss * value_loss_coef + action_loss - dist_entropy * entropy_coef)
        total_loss.backward()

        nn.utils.clip_grad_norm_(ac_agent.parameters(), max_grad_norm)

        optimizer.step()

        memory.non_spatial_obs[0].copy_(memory.non_spatial_obs[-1])
        memory.spatial_obs[0].copy_(memory.spatial_obs[-1])

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

        # Self-play swap
        if selfplay and all_steps >= selfplay_next_swap:
            selfplay_next_swap = max(all_steps + 1, selfplay_next_swap+selfplay_swap_steps)
            lower = max(0, selfplay_models-1-(selfplay_window-1))
            i = random.randint(lower, selfplay_models-1)
            model_path = f"models/{model_name}_selfplay_{i}"
            print(f"Swapping opponent to {model_path}")
            envs.swap(A2CAgent(name=f"selfplay-{i}", env_name=env_name, filename=model_path))

        # Save
        if all_updates % save_interval == 0 and len(episode_rewards) >= num_processes:
            # Save to files
            with open(log_filename, "a") as myfile:
                myfile.write(log_to_file)

                  
        
        if all_updates % log_interval == 0: 
            gbg_log = academy.report_training()
            with open("logs/Gbg_log.txt", "a+") as f: 
                f.write( gbg_log )
            
            
            # Save model
            torch.save(ac_agent, "models/" + model_name)
            print("Gbg logged at step={}/{}".format(all_steps, num_steps )) 
            
            print(gbg_log + "\n Model saved!") 
            
        
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
            #value_losses.clear()
            #policy_losses.clear()

            
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


    torch.save(ac_agent, "models/" + model_name)
    envs.close()


def make_env(worker_id):
    print("Initializing worker", worker_id, "...")
    env = gym.make(env_name)
    return env


if __name__ == "__main__":
    main()
