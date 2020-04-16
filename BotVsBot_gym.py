#!/usr/bin/env python3

#import gym
import numpy as np
import ffai

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
	

if __name__ == "__main__":

    # Create environment
    #env = gym.make("FFAI-v2")

    # Smaller variants
    # env = gym.make("FFAI-7-v2")
    # env = gym.make("FFAI-5-v2")


	print("start creating game") 
	# Load configurations, rules, arena and teams
	config = ffai.load_config("bot-bowl-ii")
	config.competition_mode = False
	ruleset = ffai.load_rule_set(config.ruleset)
	arena = ffai.load_arena(config.arena)
	home = ffai.load_team_by_filename("human", ruleset)
	away = ffai.load_team_by_filename("human", ruleset)
	config.competition_mode = False
	config.debug_mode = False




	env =  ffai.FFAIEnv(config, home, away)

	seed = 0
	env.seed(seed)
	rnd = np.random.RandomState(seed)

    
	o = env.reset()
	env.render()
	print("finished creating game")
	
	while True: 
		
		
		
		
		action = get_random_action(env, rnd)
		
		print(action)
		
		env.step( action )
		
		
		env.render()
		input()
	
		
	
	
	
	
	

