#!/usr/bin/env python3

#import gym
import numpy as np
import ffai
from pdb import set_trace
from ffai.core.table import *

from time import sleep 
import gym 

import Lectures as gc


# Lecture to render. 
lecture = gc.GameAgainstRandom() 

env =  gym.make('FFAI-11-v2')


def get_random_action(env): 
    debug = False 
    setup_actions =set([ffai.ActionType.SETUP_FORMATION_WEDGE,
                            ffai.ActionType.SETUP_FORMATION_LINE,
                            ffai.ActionType.SETUP_FORMATION_SPREAD,
                            ffai.ActionType.SETUP_FORMATION_ZONE]) 
    
    
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
    

def mstep(verbose=False): 
    action = get_random_action(env)
    if verbose: 
        if action['x'] is not None: 
            print(action['action-type'], f" ({action['x']}, {action['y']})")
        else: 
            print(action['action-type']) 
    
    obs, rew, done, info, lect_outcome = env.step( action )    
    return done 
    
def step_render(verbose=False): 
    mstep(verbose)
    env.render()


def main():     
    reset = True  
    while True: 
        
        if reset: 
            reset=False  
            env.reset(lecture) 
            env.render()
            
        c = input() 
        if c == "n": #next level 
            lecture.increase_level() 
            print(f"level: {lecture.get_level()}")
            reset = True 
        elif c == "r": #reset at same diff 
            reset = True 
        elif c == "f": #finish level with random steps, without rendering
            while not mstep():
                pass 
            reset = True 
        elif c == "s" or c == "": #take a random step 
            reset = step_render(verbose=True) 
        elif c == "t": # set trace 
            set_trace() 
        elif c == "a": #print actions 
            actions = env.game.get_available_actions()
            for a in actions: 
                print(a.action_type)
        elif c == "exit" or c == "e": 
            break 
if __name__ == "__main__":
    main() 
