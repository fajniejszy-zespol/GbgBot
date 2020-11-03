#!/usr/bin/env python3

#import gym
import numpy as np
import ffai
from pdb import set_trace
#from RewardShapping import RewardCalculation 
#from scripted_bot_example import MyScriptedBot
from ffai.core.table import *

from time import sleep 

import Curriculum as gc
import TrainAgent as gbgGym

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

lecture = gc.PreventScore(home_defence=True, debug=True) 

def main():     
    
    reset = True  
    
    while True: 
        
        if reset: 
            #lecture.increase_diff()
            lecture.level += 0.21
            print("{} lecture level: {}".format(lecture.name, lecture.get_level() )) 
            
            obs = env.reset(lecture)
            env.render()
            
            for a in env.game.get_available_actions(): 
                print(a.action_type)
            
            if input()=="x": 
               set_trace()
                
            reset=True 
            continue 
            
        try:
            action = get_random_action(env)
            print(f"choice={action['action-type']}")
            
            #if input()=="x": 
            #    set_trace()
            
            
            obs, reward, done, info = env.step( action )
            
            print(f"done = {done},  ball carried = {env.game.get_ball().is_carried}")
            
            
        
        except IndexError: 
            print("index_error")
            reset = True
            continue 
        
        if env.game.get_ball().is_carried: 
            env.render()
            x = input("ball carried!!  ")
            env.render()
            x = input("ball carried!!  ")
            
            
        if done: 
            reset = True 
            continue 
    
        
            
#        env.render()     
#        if input()=="x": 
#            set_trace()    
        # except AssertionError as err: 
            # env.render()
            # print("assertion error: {}".format(err))
            # set_trace()
            # exit() 
        
        
        #reward_shaped, prev_super_shaped = gbgGym.reward_function(env, info, shaped=True, obs=obs, prev_super_shaped = prev_super_shaped)
            
        #reset = True 
        
if __name__ == "__main__":
    main() 
