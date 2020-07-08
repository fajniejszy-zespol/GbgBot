#!/usr/bin/env python3
import ffai.ai.pathfinding as pf
import pdb
import ffai
	
class RewardCalculation: 
	def __init__(self, game, team): 
		self.game = game 
		
		if game.state.home_team != team:  
			self.opp_team = game.state.home_team 
			self.team = 	game.state.away_team
		else:
			self.team = 	game.state.home_team 
			self.opp_team = game.state.away_team
		
		self.old_turn = self.opp_team.state.turn
		
		self.old_own_score = self.team.state.score 
		self.old_opp_score = self.opp_team.state.score 
	
	
	def get_reward(self, obs): 
		""" Reward function is different depending on game state	
				Start of own turn. 
					Evaluate state and save for later 
					If Opponent had a previous turn that wasn't the first of a half: 
						handle what happened based on report. 
						should be added to reward of previous end of turn. 
				End of own turn.
					Evaluate state. Reward = state(end of turn) - state(start of turn) 
				Inside own turn 
					If last action had probability of turnover: 
						assume failure and stunned. 
						reward = max(prob * (state(end of turn) - state(start of turn) ))
						Argumentation = reward shall teach bot not to take stupid risks. 
		"""
		
		
		reward = {} 
		
		
		#Different scenarios for reward function 
		# 1 - Start of own turn
		start_own_turn = False
		
		# 2 - End of own turn 
		ended_own_turn = False
		turnover = False 
		
		# 3 - Inside own turn 
		in_own_turn = False

		# 4 - inside opp turn 
		in_opp_turn = False 
		
		
		
		
		current_turn = self.opp_team.state.turn 
		
		if current_turn != self.old_turn: 
			self.old_turn = current_turn
			new_turn_team = True
			print("    ")
			print("NEW TURN FOR TEAM")
			print("    ")
			
		if obs is not None:  
			procedures = obs[0]["procedures"]
			if procedures["Turn"] or procedures["PlayerAction"]: 
				return self.calc_state_based_reward(obs)
		else: 
			return {"no reward": 0.13 }
			
			
	def calc_state_based_reward(self, obs): 
		reward = {} 
	
		reward["score diff"] = 10*(self.team.state.score - self.opp_team.state.score)
		
		
		ball = self.game.get_ball_position() 
		ball_holder = self.game.get_ball_carrier() 

		# General ball rewards
		reward["Carry ball"] = self._carry_ball_reward() 
		if ball is not None: 
			reward["ball position"] = self._ball_position_reward()
		
		# We have the ball 
		if ball_holder is not None and ball_holder.team == self.team: 
			reward["ball tackezones"] = self.game.num_tackle_zones_in(ball_holder) * (-5) 
		
		
		# Ball on the floor  
		elif ball_holder is None and ball is not None: 
			reward["fetch ball threat"] = self._opp_pickup_ball_prob()
			print(reward["fetch ball threat"])
		
		# Opp has ball 
		elif ball_holder is not None and ball_holder.team == self.opp_team:
			reward["ball tackezones"] = self.game.num_tackle_zones_in(ball_holder) * (5)
		
		else: 
			# Should not reach this state 
			raise ImplementationsError("Should not reach this place!")
		
		#Players on the pitch 
		reward["player on pitch"] = self.get_player_on_pitch_score() 
		
		
		
		total_score = sum(reward.values()) 
		reward["total_score"] = total_score
		return reward 
	
	
	def _get_player_val(self, p): 
		if p.position is None:
			return 0 
		value = p.role.cost / 90000.0
		if not p.state.up: 
			if p.state.stunned: 
				value = value*0.5
			else: 
				value = value*0.75
		
		return value  
	
	def get_player_on_pitch_score(self): 
		
		
		team_value = sum([self._get_player_val(p) for p in self.team.players]) 
		opp_team_value = sum([self._get_player_val(p) for p in self.opp_team.players]) 
	
		return team_value - opp_team_value
	
	def _ball_position_reward(self): 
		ball = self.game.get_ball_position()
		width = self.game.arena.width
		
		if self.team == self.game.state.away_team: 
			return       (1.0 * ball.x / width)
		else: 
			return 1.0 - (1.0 * ball.x / width)
	
	def _opp_pickup_ball_prob(self): 
		ball = self.game.get_ball_position()
		opp_players = [p for p in self.opp_team.players if p.state.up and p.position is not None]
		return  max([self._get_move_and_pickup(p, ball) for p in opp_players])
		
	def _team_ball_carrier_blockable(self):
		ball_holder = self.game.get_ball_carrier()
		if not ball_holder in self.team:
			return 0
		
		position = ball_holder.position 
		"""
		available_blocks = []
        for yy in range(-1, 2, 1):
            for xx in range(-1, 2, 1):
                if yy == 0 and xx == 0:
                    continue
                p = Square(position.x+xx, position.y+yy)
                if not self.is_out_of_bounds(p)
                    player_at = self.get_player_at(p)
                    if player_at is not None:
                        if player_at.team == self.opp_team and player_at.state.up
                            available_blocks.append(player_at)
        
		
		if len(available_blocks)==0: 
			return 0
		
		block_probs = [self.game.get_block_probs(attacker, ball_holder) 
                         for attacker in available_blocks]
		if 
		
		
		self.game.get_block_probs(attacker, defender)
		"""
	
	def _carry_ball_reward(self): 
		ball_holder = self.game.get_ball_carrier() 
		if ball_holder is None: 
			return 0
		elif ball_holder in self.team.players: 
			return 1
		else: 
			return -1 
	
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