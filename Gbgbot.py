

""" 
CNN policy (same as njustesen) 

CNN wrapper encapsulates CNN policy to: 
* reduce number of observation boards
* remove padding in observations boards 
* setup GbgBot's config of the observation space and action space 
* handle actions and translating actions that are not available in neural network 
"""
 
class CNNPolicy(nn.Module):
    def __init__(self, spatial_shape, non_spatial_inputs, hidden_nodes, kernels, actions):
        super(CNNPolicy, self).__init__()

        # Spatial input stream
        self.conv1 = nn.Conv2d(spatial_shape[0], out_channels=kernels[0], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=kernels[0], out_channels=kernels[1], kernel_size=3, stride=1, padding=1)

        # Non-spatial input stream
        self.linear0 = nn.Linear(non_spatial_inputs, hidden_nodes)

        # Linear layers
        stream_size = kernels[1] * spatial_shape[1] * spatial_shape[2]
        stream_size += hidden_nodes
        self.linear1 = nn.Linear(stream_size, hidden_nodes)

        # The outputs
        self.critic = nn.Linear(hidden_nodes, 1)
        self.actor = nn.Linear(hidden_nodes, actions)

        self.train()
        self.reset_parameters()

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.linear0.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)
        self.actor.weight.data.mul_(relu_gain)
        self.critic.weight.data.mul_(relu_gain)

    def forward(self, spatial_input, non_spatial_input):
        """
        The forward functions defines how the data flows through the graph (layers)
        """
        # Spatial input through two convolutional layers
        x1 = self.conv1(spatial_input)
        x1 = F.relu(x1)
        x1 = self.conv2(x1)
        x1 = F.relu(x1)

        # Concatenate the input streams
        flatten_x1 = x1.flatten(start_dim=1)

        x2 = self.linear0(non_spatial_input)
        x2 = F.relu(x2)

        flatten_x2 = x2.flatten(start_dim=1)
        concatenated = torch.cat((flatten_x1, flatten_x2), dim=1)

        # Fully-connected layers
        x3 = self.linear1(concatenated)
        x3 = F.relu(x3)
        #x2 = self.linear2(x2)
        #x2 = F.relu(x2)

        # Output streams
        value = self.critic(x3)
        actor = self.actor(x3)

        # return value, policy
        return value, actor

    def act(self, spatial_inputs, non_spatial_input, action_mask):
        values, action_probs = self.get_action_probs(spatial_inputs, non_spatial_input, action_mask=action_mask)
        actions = action_probs.multinomial(1)
        return values, actions

    def evaluate_actions(self, spatial_inputs, non_spatial_input, actions, actions_mask):
        value, policy = self(spatial_inputs, non_spatial_input)
        actions_mask = actions_mask.view(-1, 1, actions_mask.shape[2]).squeeze().bool()
        policy[~actions_mask] = float('-inf')
        log_probs = F.log_softmax(policy, dim=1)
        probs = F.softmax(policy, dim=1)
        action_log_probs = log_probs.gather(1, actions)
        log_probs = torch.where(log_probs[None, :] == float('-inf'), torch.tensor(0.), log_probs)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return action_log_probs, value, dist_entropy

    def get_action_probs(self, spatial_input, non_spatial_input, action_mask):
        values, actions = self(spatial_input, non_spatial_input)
        # Masking step: Inspired by: http://juditacs.github.io/2018/12/27/masked-attention.html
        if action_mask is not None:
            actions[~action_mask] = float('-inf')
        action_probs = F.softmax(actions, dim=1)
        return values, action_probs


class GbgBotWrapper(object): 
	def __init__(self): 
	
		#Figure out structure of NN
		#setup CNNPolicy - load latest parameters if any compatible are available
		
		#
		pass 
	
	def act(self, game, obs): 
		
		proc = game.get_procedure()
        # print(type(proc))

		
		to_NN_directly = [FollowUp, Turn, PlayerAction]
		if any( [isinstance(proc, t) for t in to_NN_directly]):
			pass #TODO 
			raise("Not implemented!!")
		else: 
		
			# Call private function
			if isinstance(proc, CoinTossFlip):
				return return Action(ActionType.HEADS)
			if isinstance(proc, CoinTossKickReceive):
				return Action(ActionType.RECEIVE)
				# return Action(ActionType.KICK)
			if isinstance(proc, Setup): #TODO 
				raise("Not implemented!!")
				return self.setup(game)
				
			if isinstance(proc, Reroll):  #TODO 
				return Action(ActionType.DONT_USE_REROLL)
				
			if isinstance(proc, PlaceBall):
				left_center = Square(7, 8) #TODO: Fix for smaller boards 
				right_center = Square(20, 8) 
				if game.is_team_side(left_center, self.opp_team):
					return Action(ActionType.PLACE_BALL, position=left_center)
				return Action(ActionType.PLACE_BALL, position=right_center)
			if isinstance(proc, HighKick): #TODO 
				return Action(ActionType.SELECT_NONE)
				#return self.high_kick(game)
			if isinstance(proc, Touchback): #TODO 
				return self.touchback(game)
			if isinstance(proc, Block):
				#if proc.waiting_juggernaut:
				#    return self.use_juggernaut(game)
				#if proc.waiting_wrestle_attacker or proc.waiting_wrestle_defender:
				#    return self.use_wrestle(game)
				return self.block(game)
			if isinstance(proc, Push):
				#if proc.waiting_stand_firm:
				#    return self.use_stand_firm(game)
				return self.push(game)
			
			if isinstance(proc, Apothecary):
				return Action(ActionType.USE_APOTHECARY)
			
			if isinstance(proc, Interception):
				return self.interception(game)
	
	
	def follow_up(self, game): #TODO 
		# Use neural network but "move" actions. 
		raise("Not implemented!!")
		
	def touchback(self, game):
		"""
		Select player to give the ball to.
		"""
		p = None
		for player in game.get_players_on_pitch(self.my_team, up=True):
			if Skill.BLOCK in player.get_skills():
				return Action(ActionType.SELECT_PLAYER, player=player)
			p = player
		return Action(ActionType.SELECT_PLAYER, player=p)	
		
	def interception(self, game): #TODO 
		"""
		Select interceptor.
		"""
		for action in game.state.available_actions:
			if action.action_type == ActionType.SELECT_PLAYER:
				for player, agi_rolls in zip(action.players, action.agi_rolls):
					return Action(ActionType.SELECT_PLAYER, player=player)
		return Action(ActionType.SELECT_NONE)	
	
	def block(self, game): #stolen from scripted bot 
		"""
		Select block die or reroll.
		"""
		# Get attacker and defender
		attacker = game.get_procedure().attacker
		defender = game.get_procedure().defender
		is_blitz = game.get_procedure().blitz
		dice = game.num_block_dice(attacker, defender, blitz=is_blitz)

		# Loop through available dice results
		actions = set()
		for action_choice in game.state.available_actions:
			actions.add(action_choice.action_type)

		# 1. DEFENDER DOWN
		if ActionType.SELECT_DEFENDER_DOWN in actions:
			return Action(ActionType.SELECT_DEFENDER_DOWN)

		if ActionType.SELECT_DEFENDER_STUMBLES in actions and not (defender.has_skill(Skill.DODGE) and not attacker.has_skill(Skill.TACKLE)):
			return Action(ActionType.SELECT_DEFENDER_STUMBLES)

		if ActionType.SELECT_BOTH_DOWN in actions and not defender.has_skill(Skill.BLOCK) and attacker.has_skill(Skill.BLOCK):
			return Action(ActionType.SELECT_BOTH_DOWN)

		# 2. BOTH DOWN if opponent carries the ball and doesn't have block
		if ActionType.SELECT_BOTH_DOWN in actions and game.get_ball_carrier() == defender and not defender.has_skill(Skill.BLOCK):
			return Action(ActionType.SELECT_BOTH_DOWN)

		# 3. USE REROLL if defender carries the ball
		if ActionType.USE_REROLL in actions and game.get_ball_carrier() == defender:
			return Action(ActionType.USE_REROLL)

		# 4. PUSH
		if ActionType.SELECT_DEFENDER_STUMBLES in actions:
			return Action(ActionType.SELECT_DEFENDER_STUMBLES)

		if ActionType.SELECT_PUSH in actions:
			return Action(ActionType.SELECT_PUSH)

		# 5. BOTH DOWN
		if ActionType.SELECT_BOTH_DOWN in actions:
			return Action(ActionType.SELECT_BOTH_DOWN)

		# 6. USE REROLL to avoid attacker down unless a one-die block
		if ActionType.USE_REROLL in actions and dice > 1:
			return Action(ActionType.USE_REROLL)

		# 7. ATTACKER DOWN
		if ActionType.SELECT_ATTACKER_DOWN in actions:
			return Action(ActionType.SELECT_ATTACKER_DOWN)
