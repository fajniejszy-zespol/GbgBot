#!/usr/bin/env python3
import ffai.ai.pathfinding as pf
import pdb
import ffai

def get_move_and_pickup(game, p, ball): 
	
	
	if ball is None: 
		print("WTF!")
		exit() 
	
	if p is None: 
		print("WTF!")
		exit() 
	
	
	move_prob = pf.get_safest_path(game, p, ball)
	
	
	
	if move_prob is None: 
		return 0
	move_prob = move_prob.prob
	pickup_mod = game.get_pickup_modifiers(p,ball)
	
	
	pickup_prob = (p.role.ag  + pickup_mod)/6.0
	pickup_prob = min(pickup_prob, 5/6)
	pickup_prob = max(pickup_prob, 1/6)
	
	
	if p.has_skill(ffai.Skill.SURE_HANDS): 
		pickup_prob = 1-(1.0-pickup_prob)*(1-pickup_prob)
	
	print(move_prob, " * ", pickup_prob)
	
	
	return move_prob*pickup_prob
	
	
def CalcReward(obs, game, team): 
	reward = {} 
	
	# Holding ball
	ball_holder = game.get_ball_carrier() 
	
	if ball_holder is None: 
		reward["Carry ball"] = 0
	elif ball_holder in team.players: 
		reward["Carry ball"] = 1
	else: 
		reward["Carry ball"] = -1 
	
	
	# Opponent pickup ball
	ball = game.get_ball_position() 
		
	
	if ball_holder is None and ball is not None: 
		#pdb.set_trace()
		
		opp = game.state.home_team if game.state.home_team != team else game.state.away_team
		
		opp_players = [p for p in opp.players if p.state.up and p.position is not None]
		
		max_move_and_pickup_prob = max([get_move_and_pickup(game, p, ball) for p in opp_players])

		reward["fetch ball"] = max_move_and_pickup_prob
		print(reward["fetch ball"])
		#Get Opp players that can reach ball. Save probability. Nultiply with probability to pickup 
		#Pick highest probability 
	
		
	
	
	return reward


