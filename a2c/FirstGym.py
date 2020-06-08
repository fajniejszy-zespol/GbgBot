#!/usr/bin/env python3

#import gym
import numpy as np
import ffai
from pdb import set_trace
#from RewardShapping import RewardCalculation 
from scripted_bot_example import MyScriptedBot
from ffai.core.table import *

from time import sleep 

import GbG_curriculum as gc

setup_actions =set([ffai.ActionType.SETUP_FORMATION_WEDGE,
                            ffai.ActionType.SETUP_FORMATION_LINE,
                            ffai.ActionType.SETUP_FORMATION_SPREAD,
                            ffai.ActionType.SETUP_FORMATION_ZONE]) 

render_next_move = False 

def get_random_action(env): 
    debug = False 

    action_types = env.available_action_types()
    game_actions = env.game.get_available_actions() 

    if debug:
        print("Env actions", action_types)
        s = "" 
        for a in game_actions: 
            s = s + str(a.action_type) + " "
        print("Game actions", s)
    
    available_setup_actions = setup_actions.intersection(set([a.action_type for a in game_actions])) 
    if len(available_setup_actions)>0: 
        action_type = list(available_setup_actions)[0]
    else:
        while True:
            action_type = np.random.choice(action_types)
            # Ignore PLACE_PLAYER actions
            if action_type != ffai.ActionType.PLACE_PLAYER:
                break
    
    # Sample random position - if any
    available_positions = env.available_positions(action_type)
    position = np.random.choice(available_positions) if len(available_positions) > 0 else None

    # Create action dict    
    action = {
        'action-type': action_type,
        'x': position.x if position is not None else None,
        'y': position.y if position is not None else None
    }
    #action = ffai.Action(action_type=action_type, position=position, player=None)
    return action
    
def print_vars(o):
    dict_o = vars(o)
    for k in dict_o: 
        print(k, " - ", dict_o[k])

def print_dict(d): 
    for k in d: 
        print(k, " - ", d[k])
     

    
     
pitch_size = 11

# Load configurations, rules, arena and teams
#config = ffai.load_config("bot-bowl-ii")

config = ffai.load_config("ff-"+str(pitch_size))
config.fast_mode = False 
config.competition_mode = False
ruleset = ffai.load_rule_set(config.ruleset)
arena = ffai.load_arena(config.arena)

team1 = ffai.load_all_teams(ruleset, pitch_size)[0] 
team2 = ffai.load_all_teams(ruleset, pitch_size)[0] 


opponent_bot = MyScriptedBot('scripted')
    
#env =  ffai.FFAIEnv(config, team1, team2, opp_actor=opponent_bot)
env =  ffai.FFAIEnv(config, team1, team2)

seed = 1315
env.seed(seed)


#obs = env.step( get_random_action(env) )
#obs = env.step( get_random_action(env) )
#obs = env.step( get_random_action(env) )

def mstep(): 
    action = get_random_action(env)
    env.step( action )    
def mrend(): 
    env.render()
def msr(): 
    mstep()
    mrend()
    
def game_to_turn(env): 
    env.reset()
    game = env.game 
    
    while game.get_procedure().__class__.__name__ != "Turn": 
        game.step( env.game._forced_action() )
    game.step( env.game._forced_action() )
    
    board_x_max = len(game.state.pitch.board[0]) 
    board_y_max = len(game.state.pitch.board)
    
    #reset players to up and in the buttom wing
    y_pos = 1
    for players in [game.state.home_team.players, game.state.away_team.players]: 
        next_x_pos = 2
        for player in players: 
            if player.position is not None:
                # Set to ready
                player.state.reset()
                player.state.up = True
                
                position = ffai.core.model.Square(next_x_pos, y_pos)
                while game.state.pitch.board[position.y][position.x] is not None:
                    next_x_pos += 1 
                    if next_x_pos >= board_x_max: 
                        print("ERRORORORORO ")
                        exit()
                    position = ffai.core.model.Square(next_x_pos, y_pos)
                
                game.move(player, position) 
        y_pos = board_y_max -2 

    #self._reset_lecture(game)

score = gc.HandoffAndScore() 
score.level = 0

#reward = RewardCalculation(env.game, team1)
obs = env.reset(score)
#env.render()

def main():     
    pause_me = False 
    
    env.render()
    
    reset = False 
    
    while True: 
        try:
            action = get_random_action(env)
        except IndexError: 
            reset = True 
        
        if reset: 
            score.increase_diff()
            obs = env.reset(score)
            
            reset = False 
            env.render()
            if input()=="x": 
                set_trace()
            continue 
            
        
        
        obs = env.step( action )
        
        
if __name__ == "__main__":
    main() 
