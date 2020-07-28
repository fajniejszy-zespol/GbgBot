import ffai
from ffai.core.model import Square, Action, Agent 
from ffai.core.table import ActionType, Skill 
import ffai.core.procedure as FFAI_procs
import random 
from random import randint 
from pdb import set_trace
from copy import deepcopy 
from collections import Iterable
import numpy as np 
#import ffai.ai.pathfinding as pf 
import scripted_bot 
from ffai.ai.bots.random_bot import RandomBot

from scipy.special import softmax


def get_home_players(game): 
    players = [p for p in game.state.home_team.players if p.position is not None ]
    return random.sample(players, len(players))
    
def get_away_players(game): 
    players = [p for p in game.state.away_team.players if p.position is not None ]
    return random.sample(players, len(players))

def get_boundary_square(game, steps, from_position): 
    # return a position that is 'steps' away from 'from_position'
    # checks are done so it's square is available 
    board_x_max = len(game.state.pitch.board[0]) -2  
    board_y_max = len(game.state.pitch.board) -2
    
    steps = int(steps)
    assert steps > 0 

    avail_squares = steps*8 
    
    squares_per_side = 2*steps 
    
    i = 0 
    while True: 
        i +=1
        assert i<5000
        
        sq_index = randint(0, avail_squares -1)
        steps_along_side = sq_index % squares_per_side
   
        # up, including left corner 
        if sq_index // squares_per_side == 0: 
            dx = - steps + steps_along_side
            dy = - steps
        # right, including upper corner
        elif sq_index // squares_per_side == 1: 
            dx = + steps 
            dy = - steps + steps_along_side
        # down, including right corner
        elif sq_index // squares_per_side == 2: 
            dx = + steps - steps_along_side
            dy = + steps 
        # left, including lower corner    
        elif sq_index // squares_per_side == 3: 
            dx = - steps
            dy = + steps - steps_along_side
        else: 
            assert False 
    
        position = Square(from_position.x + dx, from_position.y + dy)
        x = position.x
        y = position.y 
        
        if x < 1 or x > board_x_max or  y < 1 or y > board_y_max: 
            continue 
    
        if game.state.pitch.board[y][x] is None: #it should y first, don't ask. 
            break  
            
    return position 
    
def scatter_ball(game, steps, from_position): 
    # scatters ball a certain amount of steps away for original position 
    # checks are done so it's not out of bounds or on a player 
    pos = get_boundary_square(game,steps,from_position)      
    game.get_ball().move_to(pos)

def set_player_state(player, p_used=None, p_down=None): 
    
    
    if p_down is not None: 
        player.state.up = not random.random() < p_down
    
    if p_used is not None and player.state.up: 
        player.state.used = random.random() < p_used
    
def move_player_within_square(game, player, x, y, give_ball=False, p_used=None, p_down=None): 
    # places the player at a random position within the given square. 
    
    assert isinstance(give_ball, bool)
    
    board_x_max = len(game.state.pitch.board[0]) -2  
    board_y_max = len(game.state.pitch.board) -2
    
     
    xx = sorted(x) if isinstance(x, Iterable) else (x,x)
    yy = sorted(y) if isinstance(y, Iterable) else (y,y)
    
    x_min = max(xx[0]  , 1 )
    x_max = min(xx[1]  , board_x_max ) 
    y_min = max(yy[0]  , 1 )
    y_max = min(yy[1]  , board_y_max ) 
    
    
    assert x_min <= x_max
    assert y_min <= y_max
    
    i = 0
    
    while True: 
        i += 1 
        assert i < 5000
        
        x = randint(x_min, x_max)
        y = randint(y_min, y_max)
        
        #if x < 1 or x > board_x_max or y < 1 or y > board_y_max:  
        #    continue 
        
        if game.state.pitch.board[y][x] is None: 
            break 
    
    game.move(player, Square(x,y)) 
    if give_ball == True: 
        game.get_ball().move_to( player.position ) 
        game.get_ball().is_carried = True 
    
    set_player_state(player, p_used=p_used, p_down=p_down)
        
    
def move_player_out_of_square(game, player, x, y, p_used=None, p_down=None):
    # places the player at a random position that is not in the given square. 
    
    xx = x if isinstance(x, Iterable) else (x,x)
    yy = y if isinstance(y, Iterable) else (y,y)
    
    x_min = xx[0]
    x_max = xx[1]
    y_min = yy[0]
    y_max = yy[1]
    
    board_x_max = len(game.state.pitch.board[0]) -2  
    board_y_max = len(game.state.pitch.board) -2
    
    i = 0
    
    while True: 
        x = randint(1, board_x_max)
        y = randint(1, board_y_max)
        
        if x_min <= x <= x_max and y_min <= y <= y_max: 
            i += 1 
            assert i<5000
            continue 
        
        
        if game.state.pitch.board[y][x] is None: 
            break 
    
    game.move(player, Square(x,y)) 
    set_player_state(player, p_used=p_used, p_down=p_down)
    
def move_players_out_of_square(game, players,x,y, p_used=None, p_down=None): 
    for p in players: 
        move_player_out_of_square(game,p,x,y,p_used=p_used, p_down=p_down)
        
    
class Lecture: 

    def __init__(self, name, max_level, delta_level = 0.1): 
        self.name           = name 
        self.level          = 0 
        self.max_level      = max_level
        self.delta_level    = delta_level
        self.exceptions_thrown = 0
        
    
    def increase_diff(self): 
        self.level = min( self.max_level, self.level +self.delta_level )
    def decrease_diff(self):
        self.level = max(              0, self.level -self.delta_level )        
    def get_diff(self): 
        return min(self.level, self.max_level) / self.max_level 
    def get_level(self): 
        return min( int(self.level), self.max_level) 
    
    def get_opp_actor(self): 
        return None 
    
    def lec_reset_required(self): 
        return True 
    
    def reset_game(self, game): 
        
        while True:  
            proc = game.get_procedure()
            if (proc.__class__.__name__ == "Turn" and 
                            proc.team == game.state.home_team and 
                            len( get_home_players(game) )>0 ) and not game.is_blitz() and not game.is_quick_snap() :
                break
            try: 
                while True: 
                    a = game._forced_action() 
                    if a.action_type.name != ActionType.PLACE_PLAYER: 
                        break  
                #a = env.game._forced_action() 
                game.step( a )
            except AssertionError as e: 
                pass
        
        board_x_max = len(game.state.pitch.board[0]) 
        board_y_max = len(game.state.pitch.board)
        
        #reset players to up and in the buttom wing
        y_pos = 0 #used to be 1 
        next_x_pos = 0
        players = game.state.home_team.players + game.state.away_team.players
        
        for player in players: 
            if player.position is None:
                continue 
            assert next_x_pos < board_x_max 
                
            # Set to ready
            player.state.reset()
            player.state.up = True
            
            position = ffai.core.model.Square(next_x_pos, y_pos)
            game.move(player, position) 
            
            next_x_pos += 1 

        game.set_available_actions()
        
        
        self._reset_lecture(game)
        
        game.set_available_actions()
    
    def _reset_lecture(self, game): raise "not implemented"
    def training_done(self, game): raise "not implemented"        
    def allowed_fail_rate(self): return 0 
    def is_full_game_lect(self): return False  
    
     
class Academy: 
    
    def __init__(self, lectures, nbr_of_processes, match_processes=0): 
        self.match_processes = match_processes
        self.nbr_of_processes = nbr_of_processes 
        
        assert  match_processes <= nbr_of_processes
        assert nbr_of_processes > 0 
        
        self.lectures       = lectures  
        self.match_lectures = [] 
        
        self.lect_names = [l.name for l in lectures]
        
        for l in self.lectures: 
            assert self.lect_names.count(l.name) == 1 
            if l.is_full_game_lect(): 
                self.match_lectures.append(l)
                
        self.match_lec_index = [self.lectures.index(l) for l in self.match_lectures]    
            
        self.history_size = 500    
            
        
        self.len_lects = len(self.lectures) 
        
        #History variables 
        self.latest_hundred = np.zeros( (self.len_lects, self.history_size) )
        self.rewards        = np.zeros( (self.len_lects, self.history_size) )
        self.latest_diff    = np.zeros( (self.len_lects, self.history_size) )
        self.indices        = np.zeros( (self.len_lects, ), dtype=int )
        self.episodes       = np.zeros( (self.len_lects, ), dtype=int  )
        self.max_acheived   = np.zeros( (self.len_lects, ) )
        
        self.max_name_len = max( [len(l.name) for l in lectures] )
        
        self.only_matches = (nbr_of_processes == match_processes)
        
        if match_processes <= 0:
            self.nbr_match_processes = 0 
        else: 
            self.nbr_match_processes = int(self.len_lects *(nbr_of_processes/match_processes -1)) 
        
        self.lec_prob = -2* np.ones( (self.len_lects + self.nbr_match_processes,) )  
        self.lecture_pool = self.lectures + [None] * self.nbr_match_processes 
        
        self._update_probs() 
    
    def _update_probs(self): 
     
        if self.only_matches: 
            return 
        
        diff_term       =  5*(self.latest_diff.max(axis=1) - self.latest_diff.min(axis=1))
        finished_term   =  -3*self.latest_diff.mean() #*self.latest_hundred.mean(axis=1)  
        
        self.lec_prob[ :self.len_lects ] = diff_term + finished_term 
        self.lec_prob[ self.match_lec_index ] = float("-inf")
        
        self.lec_prob_soft = softmax( self.lec_prob) 
        
        
    def get_next_lectures(self, nn):
        if self.only_matches: 
            return [None]*nn 
        
        lectures = np.random.choice( self.lecture_pool, nn-len(self.match_lectures), p = self.lec_prob_soft) 
        
        return self.match_lectures + list(lectures) 
        
    def log_training(self, data, reward): 
        
        name    = data[0]
        level   = data[1]
        outcome = data[2]
        
        lec_index = self.lect_names.index( name )
        
        # increase difficulty 
        if outcome == True and self.lectures[ lec_index ].get_level() <= level: 
            self.lectures[ lec_index ].increase_diff() 
        
        # decrease difficulty
        elif self.lectures[ lec_index ]. allowed_fail_rate() < random.random():
            self.lectures[ lec_index ].decrease_diff()
        
        # else: unchanged difficulty  
        
        
        #Logg result 
        self.rewards[lec_index, self.indices[lec_index] ]           = reward 
        self.latest_hundred[lec_index, self.indices[lec_index] ]    = outcome 
        self.latest_diff[lec_index, self.indices[lec_index] ]       = self.lectures[ lec_index ].get_diff() 
        self.indices[lec_index]                                     = (self.indices[lec_index]+1) % self.history_size
        self.episodes[lec_index] += 1 
        
        self.max_acheived[lec_index] = max( self.max_acheived[lec_index], self.lectures[lec_index].get_level() * outcome )
        
        self._update_probs() 
        
    def report_training(self, filename=None): 
        # render plots 
        
        
        s=""
        for l in self.lectures: 
            lec_index = self.lect_names.index( l.name )
            
            extra_spaces = self.max_name_len - len(l.name)
   #         s_temp = "{} - {:.4f} ({}/{})\n ".format(l.name, l.get_diff(), str(l.get_level()), str(l.max_level)  ) 
            
            name        = l.name + " "*extra_spaces
            episodes    = self.episodes[lec_index]
            #diff        =  l.get_diff()
            #max_diff    = self.max[lec_index] 
            lvl         = str( l.get_level() )
            max_acheived= self.max_acheived[lec_index]
            max_lvl     = l.max_level
            avg         = self.latest_hundred[lec_index,:].mean() 
            prob        = self.lec_prob_soft[lec_index]
            reward      = self.rewards[lec_index,:].mean() 
            #exceptions  = l.exceptions_thrown
            
            i=self.indices[lec_index]
            reward_flattened = np.concatenate( (self.rewards[lec_index, i:], self.rewards[lec_index, :i])) 
            reward_delta = reward_flattened[ self.history_size//2: ].mean() - reward_flattened[ :self.history_size//2 ].mean() 
            
            
            s_log = "{}, ep={:.0f}, lvl= {} ({:.0f})/{:.0f}, avg={:.0f}, p={:.0f}, rewrd= {:.3f}, rewrd/dt= {:.4f}".format(name, episodes, lvl, max_acheived, max_lvl, 100*avg, 100*prob,reward,  reward_delta)
            s += s_log + "\n"
        return s
            

# ### Lectures ### 
class Scoring(Lecture): 
    def __init__(self): 
        self.dst_mod = 9
        self.obstacle_mod = 4
        super().__init__("Score", self.dst_mod*self.obstacle_mod  -1) 
        
    def _reset_lecture(self, game): 

        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        dst_to_td = (level % self.dst_mod) +2 
        obstacle_level = (level // self.dst_mod) % self.obstacle_mod
        
        home_players = get_home_players(game)
        away_players = get_away_players(game)
        
        #setup ball carrier
        p_carrier = home_players.pop() 
        move_player_within_square(game, p_carrier, dst_to_td, [1, board_y_max], give_ball=True)
        
        extra_ma = (p_carrier.position.x - 1) - p_carrier.get_ma() -1 
        p_carrier.extra_ma = max(min(extra_ma, 2), 0) 
        
        
        if obstacle_level > 0: 
            #Place it so a dodge is needed when dst_to_td is low. 
            p_obstacle = away_players.pop() 
            x =  p_carrier.position.x
            y =  p_carrier.position.y
            
            if x < 4: 
                #Force dodge or blitz 
                dx = obstacle_level - 2 
                move_player_within_square(game, p_obstacle, [x-dx, x+1], [y-1, y+1])
            else:     
                #Avoid tacklezones 
                dy = 3 - obstacle_level 
                
                move_player_within_square(game, p_obstacle, [1, x-2], [y-dy, y+dy])
                
        
        #place rest of players at random places out of the way 
        x_left   =   0
        x_right  =   p_carrier.position.x+1
        y_top    =   p_carrier.position.y-1
        y_bottom =   p_carrier.position.y+1
        
        move_players_out_of_square(game, home_players+away_players, [x_left, x_right], [y_top, y_bottom] )
        
        self.turn = deepcopy(game.state.home_team.state.turn)  
    
    def training_done(self, game): 
        training_complete = self.turn  !=  game.state.home_team.state.turn
        training_outcome = game.state.home_team.state.score > 0 
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        return 0 

class PickupAndScore(Lecture): 
    def __init__(self): 
        self.dst_mod = 5
        self.ball_mod = 4
        self.noise_mod = 3
        self.marked_ball_mod = 3
        
        super().__init__("Pickup", self.dst_mod * self.ball_mod * self.noise_mod * self.marked_ball_mod -1) 
        
    def _reset_lecture(self, game): 

        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        dst_to_td =     (level % self.dst_mod) +2
        ball_start =    (level // self.dst_mod) % self.ball_mod 
        noise =         (level // self.dst_mod // self.ball_mod ) % self.noise_mod
        marked_ball =   (level // self.dst_mod // self.ball_mod  // self.noise_mod) % self.marked_ball_mod 
        
        home_players = get_home_players(game)
        away_players = get_away_players(game)
        
        #setup ball carrier
        p_carrier = home_players.pop() 
        game.move(p_carrier, Square(dst_to_td, randint(2, board_y_max -2 ) ) )
        
        if ball_start == 0: 
            scatter_ball(game, ball_start +1 , p_carrier.position)
            a = Action(action_type=ActionType.START_MOVE, position = p_carrier.position, player = p_carrier )
            game.step(a)
        else: 
            scatter_ball(game, ball_start , p_carrier.position)
        ball_pos = game.get_ball().position 
        p_pos = p_carrier.position 
        
        #Mark the ball
        x=ball_pos.x 
        y=ball_pos.y 
        p_down = 1 - noise / (self.noise_mod -1)
        move_player_within_square(game, away_players.pop(), x=[ x-1, x+1], y=[y-1, y+1], p_down = p_down)   
        
        #place rest of players at random places out of the way 
        x_left   =   0
        x_right  =   max( p_pos.x, ball_pos.x)+1
        y_top    =   min( p_pos.y, ball_pos.y)-1
        y_bottom =   max( p_pos.y, ball_pos.y)+1
        
        p_used = 1 - noise / (self.noise_mod -1)
        
        move_players_out_of_square(game, home_players, [x_left, x_right], [y_top, y_bottom],p_used=p_used)
        move_players_out_of_square(game, away_players, [x_left, x_right], [y_top, y_bottom] )
        
        self.turn = deepcopy(game.state.home_team.state.turn)  
    
    def training_done(self, game): 
        training_complete = self.turn  !=  game.state.home_team.state.turn
        training_outcome = game.state.home_team.state.score > 0 
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        return 2/6
        
class PassAndScore(Lecture): 
    def __init__(self, handoff = True,delta_level=0.1): 
        self.pass_dist_mod = 6
        self.score_dist_mod = 7
        self.noise_mod = 3
        
        self.handoff = handoff
        
        max_level = self.pass_dist_mod * self.score_dist_mod * self.noise_mod -1
        
        if handoff: 
            super().__init__("Handoff to Score", max_level,delta_level=delta_level) 
        else:
            super().__init__("Pass to Score", max_level, delta_level= delta_level)    
            
    def _reset_lecture(self, game): 
        
        assert game.is_pass_available()
        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        extra_pass_dist = 1 if self.handoff else 4
        
        level = self.get_level()        
        noise      = (level %  self.noise_mod)
        dist_pass  = (level // self.noise_mod) % self.pass_dist_mod + extra_pass_dist 
        dist_to_td = (level // self.noise_mod // self.pass_dist_mod) % self.score_dist_mod +1 #1 = start in td zone
        
        
        #get players 
        home_players = get_home_players(game)
        away_players = get_away_players(game)
        
        #setup scorer 
        p_score = home_players.pop() 
        p_score_x = dist_to_td
        p_score_y = randint(2, board_y_max -1 )
        p_score.extra_skills = [] if random.random() < 0.5 else [Skill.CATCH]
        game.move(p_score, Square( p_score_x, p_score_y) )
        
        #setup passer
        p_pass  = home_players.pop() 
        #p_pass.extra_skills = [] if random.random() < 0.5 else [Skill.PASS] #Pass skill not implemeted
        
        p_pass_x = p_score_x + dist_pass 
        dx = abs(p_pass_x - p_score_x) 
        move_player_within_square(game, p_pass, x=p_pass_x, y=[p_score_y-dx, p_score_y+dx], give_ball=True )
        
        #setup passer movement left 
        p_pass.state.moves  = p_pass.get_ma() + 2 #add two to remove GFIs
        
        if self.handoff: 
            if dx > 1: 
                #make sure passer can't score but can reach scorer 
                to_not_score= p_pass_x -1 #double gfi to score is ok. 
                to_handoff = dx #double gfi to score is ok.
                assert to_handoff <= to_not_score 
                
                p_pass.state.moves -= randint(to_handoff, to_not_score) 
            
        else: #PassAction  
            if noise > 0: 
                #make sure passer can't reach scorer: 
                to_not_handoff = dx-2 
                p_pass.state.moves  = p_pass.get_ma() + 2 #add two to remove GFIs 
                p_pass.state.moves -= randint(0, to_not_handoff) 
                
        assert 0 <= p_pass.state.moves 
        
        
        if noise == 0: 
            # Start the pass/handoff action 
            
            action_type = ActionType.START_HANDOFF if self.handoff else ActionType.START_PASS
            a = Action(action_type=action_type, position = p_pass.position, player = p_pass )
            game.step(a)
        
        if True: 
            x_min = 0
            x_max = max(p_score.position.x, p_pass.position.x)+1
            
            y_min = min(p_score.position.y, p_pass.position.y)-1
            y_max = max(p_score.position.y, p_pass.position.y)+1
            
            if noise <= 2: 
                p_used = 1-noise/2
                p_down = 1-noise/2
            else:
                p_used = 0
                p_down = 0
            
            move_players_out_of_square(game, away_players, [x_min, x_max], [y_min , y_max], p_down=p_down )
            move_players_out_of_square(game, home_players, [x_min, x_max], [y_min , y_max], p_used=p_used, p_down=p_down )
            
            self.turn = deepcopy(game.state.home_team.state.turn)  
            
        if False: 
            print("pass moves: {}/{}".format(p_pass.state.moves, p_pass.get_ma() ))
        
    def training_done(self, game): 
        training_complete = self.turn  !=  game.state.home_team.state.turn
        training_outcome = game.state.home_team.state.score > 0 
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        return 2/6 
        
class BlockBallCarrier(Lecture): 
    def __init__(self): 
        self.challenge_level = 5
        #self.obstacle_mod = 4
        super().__init__("BlockBall", self.challenge_level  -1, delta_level=0.05) 
        
    def _reset_lecture(self, game): 

        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        challenge = (level % self.challenge_level) 
        
        home_players = get_home_players(game)
        away_players = get_away_players(game)
        
        #setup ball carrier
        p_carrier = away_players.pop() 
        move_player_within_square(game, p_carrier, [2, board_x_max-1], [2, board_y_max-1], give_ball=True)
        
        x = p_carrier.position.x
        y = p_carrier.position.y 
        
        if challenge < 5:
            if   challenge == 0: p_on_ball = 5
            elif challenge == 1: p_on_ball = 2
            elif challenge == 2: p_on_ball = 2
            elif challenge == 3: p_on_ball = 3
            elif challenge == 4: p_on_ball = 2
            
            for _ in range(p_on_ball):
                if len(home_players) == 0: break 
                
                p = home_players.pop()
                if challenge <  3: move_player_within_square(game, p, [x-1,x+1], [y-1, y+1])
                if challenge >= 3: game.move(p, get_boundary_square(game, challenge-1, p_carrier.position))  
                
                if random.random() < 0.4: p.state.up = False 
 
            
        #place rest of players out of the way 
        
        move_players_out_of_square(game, home_players+away_players, [x-3, x+3], [y-3, y+3] )
        
        self.turn = deepcopy(game.state.home_team.state.turn)  
    
    def training_done(self, game): 
        
        carrier = game.get_ball_carrier()
        
        
        training_outcome = game.state.home_team.state.score > 0 or carrier not in get_away_players(game)
        
        training_complete = self.turn  !=  game.state.home_team.state.turn or training_outcome
        
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        return 0 

class CrowdSurf(Lecture): 
    def __init__(self): 
        self.challenge_level = 5
        #self.obstacle_mod = 4
        super().__init__("Surf", self.challenge_level  -1, delta_level=0.05) 
        
    def _reset_lecture(self, game): 

        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        challenge = (level % self.challenge_level) 
        
        home_players = get_home_players(game)
        away_players = get_away_players(game)
        
        #setup victim
        p_victim = away_players.pop() 
        y = random.choice([1, board_y_max]) 
        move_player_within_square(game, p_victim, [2, board_x_max-1], y)
        x = p_victim.position.x
        y = p_victim.position.y 
        
        dy = 1 if y == 1 else -1 
        dx = random.choice([-1, 1])
        
        #challenge 0: blitz or block with assist. Other players used 
        #challenge 1: block diagnonally with road block in place 
        
        #challenge 2: blitz needed 1 step
        #challenge 3: blitz needed 2-4 steps 
        #challenge 4: other players not used 
        #challenge 5: 
        
        #ball on other victim, away team or home team. 
        
        assists = 2 
        assists_p_used = 0 
        #Setup the blocker 
        p_blocker = home_players.pop() 
        if challenge == 0: 
            game.move(p_blocker, Square(x, y+dy)) 
            assists = 1
        
        if challenge == 1: 
            game.move(p_blocker, Square(x+dx, y+dy)) 
            p2 = home_players.pop() 
            game.move(p2, Square(x-dx, y)) 
            assists = 0
        
        elif challenge == 2: 
            move_player_within_square(game, p_blocker, [x-1, x+1], [y+1*dy, y+2*dy] )
            assists = 1
            assists_p_used = 0.4
        
        elif challenge == 3 or challenge == 4 : 
            move_player_within_square(game, p_blocker, [x-3, x+3], [y+2*dy, y+4*dy ] )
            assists = 1
            assists_p_used = 0.4
        
        #setup assists 
        for _ in range(assists):
            if len(home_players) > 0:
                p = home_players.pop()
                move_player_within_square(game, p, [x-1, x+1], [y, y+dy])
                p.state.used =  random.random() < assists_p_used
        
            
            
        
        #game.get_ball().move_to( p_victim.position ) 
        #game.get_ball().is_carried = True 
        #assert game.get_ball_carrier() in get_away_players(game) 
        
        if challenge < 3: 
            p_used = 1
        else: 
            p_used = 0
        
        move_players_out_of_square(game, home_players, [x-3, x+3], [y-3, y+3], p_used=p_used )
        move_players_out_of_square(game, away_players, [x-3, x+3], [y-3, y+3])
        
        if challenge == 0: 
            a = Action(action_type=ActionType.START_BLITZ, position = p_blocker.position, player = p_blocker)
            game.step(a)
        
        self.turn = deepcopy(game.state.home_team.state.turn)  
        self.len_away_team = len( get_away_players(game) ) 
    
        
    def training_done(self, game): 
        
        carrier = game.get_ball_carrier()
        
        
        training_outcome = self.len_away_team > len( get_away_players(game) ) 
        
        training_complete = self.turn  !=  game.state.home_team.state.turn or training_outcome
        
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        return 0 

class PreventScore(Lecture): 
    def __init__(self): 
        
        self.extra_skills = [ [], 
                              [Skill.BLOCK], 
                              [Skill.DODGE], 
                              [Skill.DODGE, Skill.BLOCK]] 
        
        self.noise_mod = 4  # nbr of players available for movement. 
        self.moves_mod = 5 #scatter from perfect screen position 
        self.opp_skills = len(self.extra_skills) #no, block, dodge, blodge, catch, catch+dodge, catch+dodge+pass 
        self.turns_left_mod =  3 
        
        max_level = self.noise_mod * self.moves_mod * self.opp_skills * self.turns_left_mod -1
        
        super().__init__("Prevent Score", max_level, delta_level= 0.2)
        
    def _reset_lecture(self, game): 
        
        
        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        noise      =  level %  self.noise_mod 
        moves      = (level // self.noise_mod) % self.moves_mod
        opp_skills = (level // self.noise_mod // self.moves_mod) % self.opp_skills
        turns_left = (level // self.noise_mod // self.moves_mod // self.opp_skills) % self.turns_left_mod 
        
        #get players 
        home_players = get_home_players(game)
        away_players = get_away_players(game)
        
        #setup ball carrier  
        p_score = away_players.pop() 
        move_player_within_square(game, p_score, [board_x_max-6, board_x_max-3 ], [1, board_y_max], give_ball=True)
        p_score_x = p_score.position.x
        p_score_y = p_score.position.y
        p_score.extra_skills = self.extra_skills[opp_skills]
        
        self.opp_scorer = p_score  
        
        #Setup adjescent 
        dr = 1
        p_down = 1-noise/ (self.noise_mod -1) 
        move_player_within_square(game, home_players.pop(), x=[p_score_x-dr, p_score_x+dr], y=[p_score_y-dr,p_score_y+dr], p_down=p_down )  
        
        #setup screeners 
        if   board_y_max == 15: # 11 pitch 
            screen_ys = [2, 5, 8, 11, 14]
        elif board_y_max == 11: # 7 pitch 
            screen_ys = [2, 5, 8, 11]
        elif board_y_max == 9:  # 5 pitch   
            screen_ys = [2, 5, 8]
        else: assert False 
        screen_ys = random.sample(screen_ys, len(screen_ys) )
        
        for i, y in enumerate(screen_ys): 
            p_used = ( not i < noise)  
            p_down = (p_used==0) * 0.3
            
            dr = (p_used==0)*moves
            move_player_within_square(game, home_players.pop(), x=[p_score_x+2, p_score_x+4], y=[y-dr, y+dr], p_down=p_down, p_used=p_used) 
        
        #Setup rest
        p_used = 1-level/ self.max_level
        move_players_out_of_square(game, home_players, x=[p_score_x-2, 30], y=[0,20], p_used= p_used) 
        
        move_players_out_of_square(game, away_players, x=[p_score_x-2, 30], y=[0,20], p_down= 0.25) 
        
        #Log turn 
        self.turn = deepcopy(game.state.home_team.state.turn)  
        self.opp_turn = deepcopy(game.state.away_team.state.turn)  
        
    def training_done(self, game): 
        #Play until end of drive. 
        
        # outcome is opp score didn't change. 
        
        training_complete = self.turn       != game.state.home_team.state.turn and \
                            self.opp_turn   != game.state.away_team.state.turn
        
        training_outcome = game.state.away_team.state.score == 0 
        
        
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        return 0
        
class ChooseBlockDie(Lecture): 
    action_types = [    ActionType.SELECT_ATTACKER_DOWN,
                        ActionType.SELECT_BOTH_DOWN,
                        ActionType.SELECT_PUSH,
                        ActionType.SELECT_DEFENDER_STUMBLES,
                        ActionType.SELECT_DEFENDER_DOWN]
    
    def __init__(self): 
        self.opp_skills = [ [], [Skill.BLOCK], [Skill.DODGE]]  
        self.own_skills = [ [Skill.BLOCK], [] ]  
        
        self.dice_mod = 3
        self.opp_skills_mod = len(self.opp_skills)
        self.own_skills_mod = len(self.own_skills)
                              
        #self.obstacle_mod = 4
        super().__init__("Choose Die", self.dice_mod * self.opp_skills_mod * self.own_skills_mod  -1, delta_level=0.05) 
    
    def _reset_lecture(self, game): 
        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        self.dices    =  3-(level % self.dice_mod) 
        blocker_skill =    (level // self.dice_mod) % self.own_skills_mod 
        victim_skill  =    (level // self.dice_mod // self.own_skills_mod)  % self.opp_skills_mod 
        
        blocker_team    = get_home_players(game)
        victim_team     = get_away_players(game) 
        
        victim = victim_team.pop() 
        move_player_within_square(game, victim, x = [2,board_x_max-1], y = [2, board_y_max-1], give_ball=random.random() < 0.5)
        x = victim.position.x
        y = victim.position.y
        
        blocker = blocker_team.pop() 
        move_player_within_square(game, blocker, x = [x-1,x+1], y = [y-1, y+1])
        
        
        #Setup skills
        if random.random() < 0.8: 
            blocker.extra_skills = self.own_skills[blocker_skill]
        if random.random() < 0.8: 
            victim.extra_skills = self.opp_skills[victim_skill]
        
        #setup assists if needed for two die 
        target_str = victim.get_st()  + 1 + victim.get_st()*(self.dices==3)
        blocker_assists  = target_str - blocker.get_st() 
        for _ in range(blocker_assists): 
            move_player_within_square(game, blocker_team.pop(), x = [x-1,x+1], y = [y-1, y+1], p_used=1)
        
        #Setup rest of players: 
        move_players_out_of_square(game, blocker_team, [x-4, x+4], [y-4, y+4], p_used=1)
        move_players_out_of_square(game, victim_team, [x-4, x+4], [y-4, y+4])
        
        #Randomly place ball 
        ball_pos = Square( randint(1,board_x_max), randint(1,board_y_max)) 
        game.get_ball().move_to( ball_pos ) 
        game.get_ball().is_carried = game.get_player_at(ball_pos) is not None  
                     
        game.set_available_actions()
        a = Action(action_type=ActionType.START_BLOCK, position = blocker.position, player = blocker )
        game.step(a)
        a = Action(action_type=ActionType.BLOCK, position = victim.position, player = victim)
        game.step(a)
            
        self.actions = [a.action_type for a in game.get_available_actions() ] 
        
        assert True in [a in self.actions for a in ChooseBlockDie.action_types]
           
        self.victim = victim
        self.blocker = blocker 
        
        assert game.state.active_player ==  self.blocker
        
#    def get_opp_actor(self): 
#        return ChooseBlockDie.BlockBot()
    
    def training_done(self, game): 
        
        training_complete = game.state.active_player != self.blocker
        
        training_outcome = (not self.victim.state.up) and self.blocker.state.up 
        
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        
        # if ActionType.SELECT_DEFENDER_DOWN in self.actions: 
            # return 0 
        
        # elif ActionType.SELECT_DEFENDER_STUMBLES in self.actions: 
            # return 0
            
        # elif ActionType.SELECT_BOTH_DOWN and (not self.victim.has_skill(Skill.BLOCK) ) and self.blocker.has_skill(Skill.BLOCK): 
            # return 0 
            
        # else: 
            # return 1 
        level = self.get_level()        
        self.dices    =  3-(level % self.dice_mod) 
        
        if self.dices == 1: 
            return 0.5
        elif self.dices == 2: 
            return 0.25
        else: #self.dices == 3: 
            return 0
        
class PlayAgentLecture(Lecture): 
    def __init__(self,name, agent): 
        super().__init__(name, 10)
        self.agent = agent 
    def get_opp_actor(self): 
        return self.agent 
    def lec_reset_required(self): 
        return False 
    def training_done(self, game): 
        return  game.state.game_over, (game.state.away_team.state.score <= game.state.home_team.state.score) 
    def is_full_game_lect(self): return True 
    
    
    
class PlayScriptedBot(PlayAgentLecture): 
    def __init__(self): 
        super().__init__("Scripted bot", ffai.make_bot('scripted'))

class PlayRandomBot(PlayAgentLecture): 
    def __init__(self): 
        super().__init__("Random bot", RandomBot("Random"))

        
        
    
        
    
    
if False:        
    # class Caging(Lecture): 
        # def __init__(self): 
            # self.distance_to_cage = 6
            # self.cage_setup_level = 6
            # super().__init__("Caging", self.distance_to_cage * self.cage_setup_level -1) 
        
        # def _reset_lecture(self, game): 
            
     #       ### CONFIG ### # 
            # board_x_max = len(game.state.pitch.board[0]) -2  
            # board_y_max = len(game.state.pitch.board) -2
        
    #        Level configuration 
            # level = self.get_level()        
            # distance_to_cage = (level % self.distance_to_cage) +1 
            # cage_setup_level = (level // self.distance_to_cage) % self.cage_setup_level 
            
    #        get players 
            # home_players = get_home_players(game)
            
    #        setup cage 
            # cage_x = randint(4, board_x_max -1 - distance_to_cage) 
            # cage_y = randint(2, board_y_max -1) 
            
            # ps = [home_players.pop() for _ in range(4) ]
            # game.move(ps[0], Square( cage_x+1, cage_y+1) )
            # game.move(ps[1], Square( cage_x+1, cage_y-1) )
            # game.move(ps[2], Square( cage_x-1, cage_y+1) )
            # game.move(ps[3], Square( cage_x-1, cage_y-1) )
            
            # for i in range(4):
                # if i >= cage_setup_level: 
                    # ps[i].state.used = True 
            
            # if cage_setup_level >= 5:
                # move_player_within_square(game, ps[2], [cage_x-4, cage_y+1], [cage_x-1, cage_y+4])
                # ps[2].state.up = False
                
            # if cage_setup_level >= 6:
                # move_player_within_square(game, ps[3], [cage_x-4, cage_y-4], [cage_x-1, cage_y-1])
                # ps[3].state.up = False
                
            # p_carrier = home_players.pop() 
            # move_player_within_square(game, p_carrier, [cage_x, cage_y-distance_to_cage], [cage_x+distance_to_cage, cage_y+distance_to_cage])
            # game.get_ball().move_to( p_carrier.position ) 
            # game.get_ball().is_carried = True 
                    
            
            # xs = [p.position.x for p in ps].append(p_carrier.position.x)
            # ys = [p.position.y for p in ps].append(p_carrier.position.y)
            
            # x_min = min(xs)
            # x_max = max(xs)
            # y_min = min(ys)
            # y_max = max(ys)
            
            
            # for p in home_players: 
                # move_player_out_of_square(game, p, [x_min, y_min], [x_max, y_max])
            
    #        place away team on other side of pitch, of the wings
            # for p in get_away_players(game): 
                # move_player_out_of_square(game, p, [x_min, y_min], [x_max, y_max])        
            
            # self.turn = deepcopy(game.state.home_team.state.turn)  
            
            
        # def training_done(self, game): 
            # training_complete = self.turn  !=  game.state.home_team.state.turn
            
            # num_tackle_zones_at(self, player, position)
            
            # training_outcome = game.state.home_team.state.score > 0 
            # return training_complete, training_outcome
        
        # def allowed_fail_rate(self): 
            # return 0 
    pass
