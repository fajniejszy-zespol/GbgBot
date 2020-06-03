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
     

    
     
pitch_size = 7

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
obs = env.reset()

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

score = gc.Scoring() 
score.level = 2
score.reset_env(env) 
#reward = RewardCalculation(env.game, team1)
env.render()

def main():     
    pause_me = False 
    
    assert env.game.get_ball_carrier() is not None 
    env.render()
    prev_action = []
    print("----RESET-----")
    
    reset = False 
    
    success_outcomes = set() 
    failed_outcomes = set() 
    
    as_it_should_be = 0
    as_it_shouldnt  = 0
    
    
    while True: 
        try:
            action = get_random_action(env)
        except IndexError: 
            reset = True 
        
        if reset: 
            score.reset_env(env)
            
            reset = False 
            #print("----RESET-----")
            prev_action = []            
            env.render()
           
            if as_it_should_be + as_it_shouldnt > 0: 
                print("Failure rate = ", 100*  as_it_shouldnt / (as_it_should_be + as_it_shouldnt), "% " )  
            
            
            continue 
            
        #print("action choice: ", str(action))
        
        #if (action["action-type"] == ActionType.MOVE):
        
        proc = env.game.get_procedure()
        
        if hasattr(proc, "player"): 
            p = proc.player 
        
            ballHolder = env.game.get_ball_carrier() 
            
            if ballHolder is not None and ballHolder == p: 
                
                
                if (action["action-type"] == ActionType.MOVE):
                
                    
                    
                    s = "---Moving ball: " +  str(proc.player_action_type)
                    #print(proc.player_action_type)
                    
                    dodge = False 
                    if env.game.num_tackle_zones_in(p)>0: 
                        dodge = True 
                    #input() 
                    #env.render()
                    #set_trace()
                    
                    obs = env.step( action )
                    env.render()
                    ball = env.game.get_ball()
                    turn = env.game.state.home_team.state.turn
                    half = env.game.state.half 
                    
                    if ball is not None and ball.position is not None and env.game.get_ball_carrier() is None and p.state.up: 
                        
                        #print ("  BALL FUCKED!")
                        if dodge:
                            print ("  BALL FUCKED!")
                            print("  DODGE ")
                            input() 
                        #print(s)
                        #print ("  turn:", turn)
                        #print ("  previous actions (", len(prev_action),end="): ")
                        #for a in prev_action:
                        #    print(a, end =" ")
                        #print("")
                        
                        as_it_shouldnt  += 1 
                        
                        for x in env.game.state.reports: 
                            #if x.outcome_type in kickoff_weathers: 
                            failed_outcomes.add(x.outcome_type.name)
                        #if len(success_outcomes)>0 and len(failed_outcomes)>0: 
                         #   for x in failed_outcomes.difference(success_outcomes): 
                          #      print(x)    
                        
                        #input() 
                        
                    elif ball is not None and ball.position is not None and env.game.get_ball_carrier() is not None and half == 1: 
                        #print("success moved ball: ", end ="")
                        #print ("previous actions (", len(prev_action),end="): ")
                        #for a in prev_action:
                        #    print(a, end =" ")
                        #print("")
                        for x in env.game.state.reports: 
                            #if x.outcome_type in kickoff_weathers: 
                            success_outcomes.add(x.outcome_type.name)
                        
                        as_it_should_be += 1
                    
                    
                    reset = True     
                    continue
                else: 
                    prev_action.append( action["action-type"] )
        
        obs = env.step( action )
        
        #r = reward.get_reward(obs)
        #env.render(reward_array=r)
        #env.render() 
        
        
        
        
if __name__ == "__main__":
    main() 
