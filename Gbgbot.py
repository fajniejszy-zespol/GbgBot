

""" 
CNN policy (same as njustesen) 

CNN wrapper encapsulates CNN policy to: 
* reduce number of observation boards
* remove padding in observations boards 
* setup GbgBot's config of the observation space and action space 
* handle actions and translating actions that are not available in neural network 
"""

class Memory(object):
    def __init__(self, steps_per_update, num_processes, spatial_obs_shape, non_spatial_obs_shape, action_space):
        self.spatial_obs = torch.zeros(steps_per_update + 1, num_processes, *spatial_obs_shape)
        self.non_spatial_obs = torch.zeros(steps_per_update + 1, num_processes, *non_spatial_obs_shape)
        self.rewards = torch.zeros(steps_per_update, num_processes, 1)
        self.value_predictions = torch.zeros(steps_per_update + 1, num_processes, 1)
        self.returns = torch.zeros(steps_per_update + 1, num_processes, 1)
        action_shape = 1
        self.actions = torch.zeros(steps_per_update, num_processes, action_shape)
        self.actions = self.actions.long()
        self.masks = torch.ones(steps_per_update + 1, num_processes, 1)
        self.action_masks = torch.zeros(steps_per_update + 1, num_processes, action_space, dtype=torch.uint8)

    def cuda(self):
        self.spatial_obs = self.spatial_obs.cuda()
        self.non_spatial_obs = self.non_spatial_obs.cuda()
        self.rewards = self.rewards.cuda()
        self.value_predictions = self.value_predictions.cuda()
        self.returns = self.returns.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()
        self.action_masks = self.action_masks.cuda()

    def insert(self, step, spatial_obs, non_spatial_obs, action, value_pred, reward, mask, action_masks):
        self.spatial_obs[step + 1].copy_(spatial_obs)
        self.non_spatial_obs[step + 1].copy_(non_spatial_obs)
        self.actions[step].copy_(action)
        self.value_predictions[step].copy_(value_pred)
        self.rewards[step].copy_(reward)
        self.masks[step].copy_(mask)
        self.action_masks[step].copy_(action_masks)

    def compute_returns(self, next_value, gamma):
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.size(0))):
            self.returns[step] = self.returns[step + 1] * gamma * self.masks[step] + self.rewards[step]


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

    non_spatial_obs = [        "is move action",
                            "is block action",
                            "is blitz action",
                            "is pass action",
                            "is handoff action" ] 
    
    simple_action_types = [
        #ActionType.START_GAME,
        #ActionType.HEADS,
        #ActionType.TAILS,
        #ActionType.KICK,
        #ActionType.RECEIVE,
        #ActionType.END_SETUP,
        ActionType.END_PLAYER_TURN,
        #ActionType.USE_REROLL,
        #ActionType.DONT_USE_REROLL,
        #ActionType.USE_SKILL,
        #ActionType.DONT_USE_SKILL,
        #ActionType.SELECT_ATTACKER_DOWN,
        #ActionType.SELECT_BOTH_DOWN,
        #ActionType.SELECT_PUSH,
        #ActionType.SELECT_DEFENDER_STUMBLES,
        #ActionType.SELECT_DEFENDER_DOWN,
        #ActionType.SELECT_NONE
        ActionType.END_TURN,
        ActionType.STAND_UP
        
    ]
    
    positional_action_types = [
        #ActionType.PLACE_BALL,
        ActionType.PUSH,
        #ActionType.FOLLOW_UP,
        ActionType.MOVE,
        ActionType.BLOCK,
        ActionType.PASS,
        #ActionType.FOUL,
        ActionType.HANDOFF,
        #ActionType.LEAP,
        #ActionType.STAB,
        #ActionType.SELECT_PLAYER,
        ActionType.START_MOVE,
        ActionType.START_BLOCK,
        ActionType.START_BLITZ,
        ActionType.START_PASS,
        #ActionType.START_FOUL,
        ActionType.START_HANDOFF
    ]
    
    _layer_objects = [
        OccupiedLayer(),
        OwnPlayerLayer(),
        OppPlayerLayer(),
        OwnTackleZoneLayer(),
        OppTackleZoneLayer(),
        UpLayer(),
        #StunnedLayer(),
        UsedLayer(),
        RollProbabilityLayer(),
        BlockDiceLayer(),
        ActivePlayerLayer(),
        TargetPlayerLayer(),
        #MALayer(),
        #STLayer(),
        #AGLayer(),
        #AVLayer(),
        MovemenLeftLayer(),
        BallLayer(),
        #OwnHalfLayer(),
        #OwnTouchdownLayer(),
        #OppTouchdownLayer(),
        SkillLayer(Skill.BLOCK),
        SkillLayer(Skill.DODGE),
        #SkillLayer(Skill.SURE_HANDS),
        #SkillLayer(Skill.CATCH),
        SkillLayer(Skill.PASS)
    ]
    layers = [layer.name() for layer in GbgBotWrapper._layer_objects]
    
    
    
    def __init__(self, env, num_processes): #DONE
        
        self.step = 0 
        
        #Figure out structure of NN
        
        #OBSERVATION SPACE
        spatial_obs, not_spatial_obs = self._filter_observations(env.observation_space) #needs updating 
        
        #OBSERVATION SPACE - spatial     
        self.spatial_obs_space = spatial_obs.shape
        nbr_of_observation_layers = self.spatial_obs_space[0] # depends on GbgBot config 
        self.board_width  =  self.spatial_obs_space[1] #depends on board 
        self.board_height =  self.spatial_obs_space[2] # depends on board 
        self.board_squares = self.board_height * self.board_width
        
        spatial_obs = [nbr_of_observation_layers, self.board_width, self.board_height]
        
        #OBSERVATION SPACE - not spatial     
        num_non_spatial_obs = len(not_spatial_obs) 
        
        #ACTION SPACE 
        self.action_space = len(GbgBotWrapper.simple_action_types) + len(GbgBotWrapper.positional_action_types) * self.board_squares
        
        hidden_nodes=256  # Hyper Parameter
        kernels=[32, 64] # Hyper Parameter
        steps_per_update = 40  
        
        
        
        self.policy = CNNPolicy(spatial_obs, num_non_spatial_obs, hidden_nodes, kernels, self.action_space)
        #                        3D     , scalar,      scalar,      , 2D,     scalar  
        
        self.memory = Memory(steps_per_update, num_processes, spatial_obs, (1, num_non_spatial_obs), self.action_space)

    
    def _filter_observations(self, obs): #DONE
        
        #Spatial observations 
        obs_layers = [] 
        for layer_name in GbgBotWrapper.layers: 
             obs_layers.append( obs["board"][layer_name])
        
        spatial_obs = np.stack(obs_layers)[:, 1:-1, 1:-1]
        
        #Non spatial
        not_spatial_obs = [obs["state"][key] for key in GbgBotWrapper.non_spatial_obs ]
        
        return spatial_obs, not_spatial_obs
    
    def compute_action_masks(observations, envs):  #TODO - This is not in the observation. Use env.available_actions instead 
        masks = []
        m = False
        for ob in observations:
            mask = np.zeros(self.action_space)
            i = 0
            for action_type in GbgBotWrapper.simple_action_types:
                mask[i] = ob['available-action-types'][action_type.name]
                i += 1
            for action_type in spatial_action_types:
                if ob['available-action-types'][action_type.name] == 0:
                    mask[i:i+board_squares] = 0
                elif ob['available-action-types'][action_type.name] == 1:
                    position_mask = ob['board'][f"{action_type.name.replace('_', ' ').lower()} positions"]
                    position_mask_flatten = np.reshape(position_mask, (1, board_squares))
                    for j in range(board_squares):
                        mask[i + j] = position_mask_flatten[0][j]
                i += board_squares
            assert 1 in mask
            if m:
                print(mask)
            masks.append(mask)
        return masks

    def compute_action(self, action_idx): #DONE
        if action_idx < len(GbgBotWrapper.simple_action_types):
            return GbgBotWrapper.simple_action_types[action_idx], 0, 0
        spatial_idx             = action_idx - len(GbgBotWrapper.simple_action_types)
        spatial_pos_idx         = spatial_idx % self.board_squares
        spatial_y                 = int(spatial_pos_idx / self.board_width)
        spatial_x                 = int(spatial_pos_idx % self.board_width)
        spatial_action_type_idx = int(spatial_idx / self.board_squares)
        spatial_action_type     = GbgBotWrapper.positional_action_types[spatial_action_type_idx]
        return spatial_action_type, spatial_x, spatial_y
    
    def act(self, observations): #TODO Handle memory insert here 
        
        action_objects = []
        step = self.step 
        
        for i,obs in enumerate(obeservations): 
            
            proc = FFAIEnv.procedures[obs["procedure"].index(1.0)] 
            # print(type(proc))
            
            
            
            to_NN_directly = [FollowUp, Turn, PlayerAction]
            
            
            
            if any( [isinstance(proc, t) for t in to_NN_directly]):
                
                #Apply the filter 
                #TODO: Put stuff in memory
                
                #Todo handle follow-up here. 
                
                #Todo handle push into crowd here. 
                
                action_masks = self.compute_action_masks(obs)
                action_masks = torch.tensor(action_masks, dtype=torch.bool)

                values, actions = ac_agent.act(
                    Variable(self.memory.spatial_obs[step]),
                    Variable(self.memory.non_spatial_obs[step]),
                    Variable(action_masks))

                action_objects = []

                for action in actions:
                    action_type, x, y = compute_action(action.numpy()[0])
                    action_object = {
                        'action-type': action_type,
                        'x': x,
                        'y': y
                    }
                    action_objects.append(action_object)
                
                  
                
            else: 
                action_objects.append( act_scripted(proc, game) ) 
                #TODO: handle the memory so that this action is not used in updating the policy 
    
    def step_optimizer(self): #TODO  Check that everything work as expected and all variables are defined 
        next_value = self.policy(Variable(memory.spatial_obs[-1], requires_grad=False), Variable(memory.non_spatial_obs[-1], requires_grad=False))[0].data

        # Compute returns
        memory.compute_returns(next_value, gamma)

        spatial = Variable(memory.spatial_obs[:-1])
        spatial = spatial.view(-1, *(self.spatial_obs_space) )
        non_spatial = Variable(memory.non_spatial_obs[:-1])
        non_spatial = non_spatial.view(-1, non_spatial.shape[-1])

        actions = Variable(torch.LongTensor(memory.actions.view(-1, 1)))
        actions_mask = Variable(memory.action_masks[:-1])

        # Evaluate the actions taken
        action_log_probs, values, dist_entropy = ac_agent.evaluate_actions(spatial, non_spatial, actions, actions_mask)

        values = values.view(steps_per_update, num_processes, 1)
        action_log_probs = action_log_probs.view(steps_per_update, num_processes, 1)

        advantages = Variable(memory.returns[:-1]) - values
        value_loss = advantages.pow(2).mean()
        #value_losses.append(value_loss)

        # Compute loss
        action_loss = -(Variable(advantages.data) * action_log_probs).mean()
        #policy_losses.append(action_loss)

        optimizer.zero_grad()

        total_loss = (value_loss * value_loss_coef + action_loss - dist_entropy * entropy_coef)
        total_loss.backward()

        nn.utils.clip_grad_norm_(ac_agent.parameters(), max_grad_norm)

        optimizer.step()

        memory.non_spatial_obs[0].copy_(memory.non_spatial_obs[-1])
        memory.spatial_obs[0].copy_(memory.spatial_obs[-1])

        # Updates
        all_updates += 1
        # Episodes
        all_episodes += episodes
        episodes = 0
        # Steps
        all_steps += num_processes * steps_per_update

    
    def act_scripted(self, observation, game): #TODO: Setup 
        # Call private function
        if isinstance(proc, CoinTossFlip):
            return Action(ActionType.HEADS)
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
            #if proc.waiting_stand_firm:
        if isinstance(proc, Push):
            #    return self.use_stand_firm(game)
            return self.push(game)
        
        if isinstance(proc, Apothecary):
            return Action(ActionType.USE_APOTHECARY)
        
        if isinstance(proc, Interception):
            return self.interception(game)

        raise("Should not reach this place! act_scripted() ")
        
    def touchback(self, game):  #TODO
        """
        Select player to give the ball to.
        """
        p = None
        for player in game.get_players_on_pitch(self.my_team, up=True):
            if Skill.BLOCK in player.get_skills():
                return Action(ActionType.SELECT_PLAYER, player=player)
            p = player
        return Action(ActionType.SELECT_PLAYER, player=p)    
        
    def interception(self, game):
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
        
        #TODO - remove game
        
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
