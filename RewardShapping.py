#!/usr/bin/env python3

def CalcReward(obs, game, team): 
	reward = {} 
	
	# Holding ball
	ball_holder = game.get_ball_carrier() 
	reward["Carry ball"] = 0 if ball_holder is None else ... 
	                       1 if ball_holder in team.players else ...
						   -1 # ball_holder in opp_team 
	
	# Opponent pickup ball
	if ball_holder is None: 
		ball = game.get_ball_position() 
		
		#Get Opp players that can reach ball. Save probability. Nultiply with probability to pickup 
		#Pick highest probability 
		
		
	
	
	return reward


