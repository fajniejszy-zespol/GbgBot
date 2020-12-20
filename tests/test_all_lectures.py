
from ffai.core.load import *
from ffai.ai.bots.random_bot import *
from pytest import set_trace

import Lectures as gc
from Curriculum import Academy, LectureOutcome 
from VectorEnvironment import VecEnv, Memory, WorkerMemory, reward_function

import itertools as it 
import gym 
import torch 

class A2c_agent_tester(Agent): 
    def __init__(self, env): 
        super().__init__("a2c_random_tester")
        
        self.random_agent = RandomBot("Random_bot")
        self.cnn_used = False
        
        self.action_size = env.get_action_shape() 
        
    def act(self, game, env=None, obs=None):
        if game is None: 
            game = env.game 
        
        action = self.random_agent.act(game)
        x = action.position.x if action.position is not None else None 
        y = action.position.y if action.position is not None else None 
        
        if action.position is None and action.player is not None : 
            pos = action.player.position 
            x = pos.x 
            y = pos.y 
            assert x is not None and y is not None 
        
        action_object = {   'action-type': action.action_type,
                            'x': x,
                            'y': y } 
        
        self.cnn_used = self.cnn_used == False #flips the state
        
        assert obs is not None 
        spatial_obs, non_spatial_obs = self._update_obs(obs)
    
        
        if self.cnn_used: 
            actions = 1
            action_masks = np.zeros(self.action_size)
            action_masks = torch.tensor(action_masks, dtype=torch.bool)
            value = 0 
            return (action_object, actions, action_masks, value ,spatial_obs, non_spatial_obs)
        else: 
            return (action_object, None, None, None, spatial_obs, non_spatial_obs)
    
    def _update_obs(self, obs):
        """
        Takes the observation returned by the environment and transforms it to an numpy array that contains all of
        the feature layers and non-spatial info.
        """
        
        spatial_obs = np.stack(obs['board'].values())

        state = list(obs['state'].values())
        procedures = list(obs['procedures'].values())
        actions = list(obs['available-action-types'].values())

        non_spatial_obs = np.stack(state+procedures+actions)
        non_spatial_obs = np.expand_dims(non_spatial_obs, axis=0)
    
        return torch.from_numpy(np.stack(spatial_obs)).float(), torch.from_numpy(np.stack(non_spatial_obs)).float()
        
lectures_to_test = [gc.GameAgainstRandom(), gc.Scoring() ]

def test_all_lectures(): 
    configs = ["FFAI-11-v2", "FFAI-7-v2", "FFAI-5-v2"]
    for lect, config in it.product(lectures_to_test, configs):

        env = gym.make(config)

        while lect.get_level() < lect.max_level:
            env.reset(lecture=lect)

            try:
                assert env.game.home_agent.human
                assert not env.game.away_agent.human
                assert env.actor != env.opp_actor

                assert env.game.state.home_team != env.game.state.away_team

                assert env.own_team != env.opp_team
            except:
                set_trace()
            lect.increase_level() 
            
def test_academy(): 
    school = Academy(lectures_to_test)
    
    for lect in lectures_to_test: 
        for i in range(30): 
            outcome = LectureOutcome( lect, win=True )
            school.log_training(outcome)
    
    s = school.report() 

