#!/usr/bin/env python3

#import gym
import numpy as np
import ffai
import pdb
from RewardShapping import CalcReward 

def get_random_action(env, rnd): 
	action_types = env.available_action_types()
	
	
	while True:
		action_type = rnd.choice(action_types)
		# Ignore PLACE_PLAYER actions
		if action_type != ffai.ActionType.PLACE_PLAYER:
			break
	
	
	# Sample random position - if any
	available_positions = env.available_positions(action_type)
	pos = rnd.choice(available_positions) if len(available_positions) > 0 else None

	# Create action object
	action = {
		'action-type': action_type,
		'x': pos.x if pos is not None else None,
		'y': pos.y if pos is not None else None
	}
	return action
	
def print_vars(o):
	dict_o = vars(o)
	for k in dict_o: 
		print(k, " - ", dict_o[k])
	
	
	
pitch_size = 5

# Load configurations, rules, arena and teams
#config = ffai.load_config("bot-bowl-ii")

config = ffai.load_config("ff-"+str(pitch_size))
config.competition_mode = False
ruleset = ffai.load_rule_set(config.ruleset)
arena = ffai.load_arena(config.arena)

team1 = ffai.load_all_teams(ruleset, pitch_size)[0] 
team2 = ffai.load_all_teams(ruleset, pitch_size)[0] 

#home = ffai.load_team_by_filename("human-3", ruleset)
#away = ffai.load_team_by_filename("human-3", ruleset)
#config.competition_mode = False
#config.debug_mode = False
	


    # Create environment
    #env = gym.make("FFAI-v2")

    # Smaller variants
    # env = gym.make("FFAI-7-v2")
    # env = gym.make("FFAI-5-v2")
	


env =  ffai.FFAIEnv(config, team1, team2)

seed = 13125
env.seed(seed)
rnd = np.random.RandomState(seed)


obs = env.reset()
for i in range(5): 	
	action = get_random_action(env, rnd)
	obs = env.step( action )

	
if __name__ == "__main__":
	while True: 
		action = get_random_action(env, rnd)
		print(action)
		obs = env.step( action )
		
		
		r = CalcReward(None, env.game, team1 )
		if "fetch ball" in r: 
			env.render(reward_array=r)
			input()
	
		
	
	
	
	
	

