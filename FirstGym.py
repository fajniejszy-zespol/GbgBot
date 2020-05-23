#!/usr/bin/env python3

#import gym
import numpy as np
import ffai
from pdb import set_trace
from RewardShapping import RewardCalculation 
from scripted_bot_example import MyScriptedBot

from time import sleep 

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
	 

	
	 
pitch_size = 3

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

obs = env.step( get_random_action(env) )
obs = env.step( get_random_action(env) )
obs = env.step( get_random_action(env) )

reward = RewardCalculation(env.game, team1)
	
if __name__ == "__main__":
	obs = env.step( get_random_action(env) )
	obs = env.step( get_random_action(env) )
	obs = env.step( get_random_action(env) )
	env.render(reward_array=None)
	
	
	while True: 
		action = get_random_action(env)
		print("action choice: ", str(action))
		
		obs = env.step( action )
		
		r = reward.get_reward(obs)
		env.render(reward_array=r)
		
		input() 
			
