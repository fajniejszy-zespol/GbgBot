#!/usr/bin/env python3

#import gym
import numpy as np
import ffai
from pdb import set_trace
from RewardShapping import RewardCalculation 
from scripted_bot_example import MyScriptedBot

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
	
	if ffai.ActionType.SETUP_FORMATION_SPREAD in [a.action_type for a in game_actions]: 
		action_type = ffai.ActionType.SETUP_FORMATION_SPREAD
	else
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
	
	
	
pitch_size = 5

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

reward = RewardCalculation(env.game, team1)
	
if __name__ == "__main__":
	while True: 
		action = get_random_action(env)
		print("action choice: ", str(action))
		
		turns = (env.game.state.home_team.state.turn, env.game.state.away_team.state.turn) 
		obs = env.step( action )
		
		
		turns2 = (env.game.state.home_team.state.turn, env.game.state.away_team.state.turn) 
		
		if turns != turns2:
			print("turn update")
			print(turns2)
		
		
		r = reward.get_reward(None)
		#if "fetch ball" in r: 
		env.render(reward_array=r)
		input()
	
