#!/usr/bin/env python3
import ffai.ai.pathfinding as pf
import pdb
import ffai



	
	
class RewardCalculation: 
	def __init__(self, game, team): 
		self.game = game 
		self.team = team 
		self.opp_team = game.state.home_team if game.state.home_team != team else game.state.away_team
	
	def get_reward(self, obs): 
		reward = {} 
		
		
		
		ball = self.game.get_ball_position() 

		# Holding ball
		ball_holder = self.game.get_ball_carrier() 

		if ball_holder is None: 
			reward["Carry ball"] = 0
		elif ball_holder in self.team.players: 
			reward["Carry ball"] = 1
		else: 
			reward["Carry ball"] = -1 

		# Opponent pickup ball
		if ball_holder is None and ball is not None: 

			opp_players = [p for p in self.opp_team.players if p.state.up and p.position is not None]
			max_move_and_pickup_prob = max([self._get_move_and_pickup(p, ball) for p in opp_players])
			reward["fetch ball"] = max_move_and_pickup_prob
			
			print(reward["fetch ball"])
			
		
		
		return reward
		
		
	def _get_move_and_pickup(self, p, ball): 

		
		path = pf.get_safest_path(self.game, p, ball)
		
		if path is None: 
			return 0
		
		pickup_mod = self.game.get_pickup_modifiers(p,ball)
		
		pickup_prob = (p.role.ag  + pickup_mod)/6.0
		pickup_prob = min(pickup_prob, 5/6)
		pickup_prob = max(pickup_prob, 1/6)
		
		if p.has_skill(ffai.Skill.SURE_HANDS): 
			pickup_prob = 1-(1.0-pickup_prob)*(1-pickup_prob)
		
		return path.prob * pickup_prob