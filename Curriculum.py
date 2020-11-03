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

game_turn_memoized = {} 
def get_empty_game_turn(config, turn, clear_board=True, hometeam="human", awayteam="human", away_agent=None ):
    
    pitch_size = config.pitch_size  
    
    key = f"{hometeam} {awayteam} {pitch_size} {turn)""
    if key in game_turn_memoized:
        game = deepcopy(game_turn_memoized[key])
        
        if away_agent is not None: 
            game.replace_away_agent(away_agent)
        return game 
    
    D3.FixedRolls = []
    D6.FixedRolls = [3,3,3,3,3,3,3,3,3] #No crazy kickoff or broken armors 
    D8.FixedRolls = []
    BBDie.FixedRolls = []
    
    ruleset = load_rule_set(config.ruleset)
    
    size_suffix = f"{pitch_size}" if pitch_size != 11 else "" 
    home = load_team_by_filename(hometeam+size_suffix, ruleset, board_size=pitch_size)
    away = load_team_by_filename(awayteam+size_suffix, ruleset, board_size=pitch_size)
    game = Game(seed, home, away, Agent("human1", human=True), Agent("human2", human=True) , config)
    game.init()
    
    if turn > 0: 
        game.step(Action(ActionType.START_GAME))
        game.step(Action(ActionType.HEADS))
        game.step(Action(ActionType.KICK))
        game.step(Action(ActionType.SETUP_FORMATION_ZONE))
        game.step(Action(ActionType.END_SETUP))
        game.step(Action(ActionType.SETUP_FORMATION_WEDGE))
        game.step(Action(ActionType.END_SETUP))
        random_agent = RandomBot("home")
        while type(game.get_procedure()) is not Turn or game.is_quick_snap() or game.is_blitz():
            action = random_agent.act(game)
            game.step(action)
        
        while game.state.home_team.state.turn != turn: 
            game.step(Action(ActionType.END_TURN))
        
        if clear_board: 
            game.clear_board()
        
    if away_agent is not None: 
        game.replace_away_agent(away_agent)
    else: 
        game.replace_away_agent(random_agent)
    D6.FixedRolls = []
    game_turn_memoized[key] = deepcopy(game)
    
    return game

 
    
def get_home_players(game): 
    num = min(game.config.pitch_max, len(game.state.home_team.players) ) 
    return random.sample(game.state.home_team.players, num)
    
def get_away_players(game): 
    num = min(game.config.pitch_max, len(game.state.away_team.players) ) 
    return random.sample(game.state.away_team.players, num)

def get_boundary_square(game, steps, from_position): 
    steps = int(steps)
    
    if steps == 0: 
        if game.state.pitch.board[from_position.y][from_position.x] is None: 
            return from_position
        else: 
            steps += 1 
    
    # return a position that is 'steps' away from 'from_position'
    # checks are done so it's square is available 
    board_x_max = len(game.state.pitch.board[0]) -2  
    board_y_max = len(game.state.pitch.board) -2
    
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
    if steps > 0:  
        pos = get_boundary_square(game,steps,from_position)      
    else: 
        pos = from_position
    
    game.get_ball().move_to(pos)
    game.get_ball().is_carried = False 

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

def swap_game(game): 

    moved_players = []
    board_x_max = len(game.state.pitch.board[0]) -2      
    
    player_to_move  = get_home_players(game) + get_away_players(game) 
    for p in player_to_move: 
        
        if p in moved_players: 
            continue 
        
        old_x = p.position.x 
        new_x = 27  - old_x 
        
        potential_swap_p =  game.state.pitch.board[p.position.y][new_x]
        if potential_swap_p is not None: 
            game.move(potential_swap_p , Square(0,0) ) 
        
        game.move(p, Square(new_x, p.position.y) ) 
        
        if potential_swap_p is not None: 
            game.move(potential_swap_p , Square(old_x, p.position.y) ) 
            moved_players.append(potential_swap_p) 
            
    # ball_pos = game.get_ball().position 
    
    # ball_new_x  = 27-ball_pos.x
    # ball_y      = ball_pos.y
    
    # game.get_ball().move_to( Square(ball_new_x, ball_y) )
    
    # assert game.get_ball().position.x ==  ball_new_x
    
class Lecture: 
    def __init__(self, name, max_level, delta_level = 0.1): 
        self.name           = name 
        self.level          = 0 
        self.max_level      = max_level
        self.delta_level    = delta_level
        self.exceptions_thrown = 0
        
        assert delta_level > 0 

    def increase_diff(self): 
        self.level = min( self.max_level, self.level +self.delta_level )
    def decrease_diff(self):
        self.level = max(              0, self.level -self.delta_level )        
    def get_diff(self): 
        return min(self.level, self.max_level) / self.max_level 
    def get_level(self): 
        return min( int(self.level), self.max_level) 
        
    def reset_game(self, config): 
        """ 
        :paran config: integer of pitch size, (currently 3,5,7,11)
        :return: return a fully initialized game object, with opp_agent initialized
        """ 
        raise NotImplementedError("Must be overridden by subclass")
    
    def evaluate(self, game, drive_over): 
        """
        :param game: game object to be judged
        :param drive_over: is set to True last time this function is called. 
        :return: a LectureOutcome object  
        """
        raise NotImplementedError("Must be overridden by subclass")
    def allowed_fail_rate(self): 
        """
        Not sure how to use this. TODO TBD 
        """
        return 0 
    
class LectureOutcome: 
    def __init__(self, lecture, win, draw=None): 
        
        if win:  
            self.outcome = 1 
        elif draw is not None and draw: 
            self.outcome = 0 
        else: 
            self.outcome = -1  
            
        self.lect_type = type(lecture)

        
    
class Academy: 
    
    def __init__(self, lectures, nbr_of_processes, ordinary_matches=0): 

        self.nbr_of_processes = nbr_of_processes 
        self.ordinary_matches = ordinary_matches 
        
        assert  ordinary_matches <= nbr_of_processes
        assert  nbr_of_processes > 0 
        
        self.lectures       = lectures  
        
        self.lect_names = [l.name for l in lectures]
        
        for l in self.lectures: 
            assert self.lect_names.count(l.name) == 1 
            
        self.history_size = 300    
            
        
        self.len_lects = len(self.lectures) 
        
        #History variables 
        self.latest_hundred = np.zeros( (self.len_lects, self.history_size) )
        self.rewards        = np.zeros( (self.len_lects, self.history_size) )
        self.latest_level   = np.zeros( (self.len_lects, self.history_size) )
        self.indices        = np.zeros( (self.len_lects, ), dtype=int )
        self.episodes       = np.zeros( (self.len_lects, ), dtype=int  )
        self.max_acheived   = np.zeros( (self.len_lects, ) )
        self.max_name_len = max( [len(l.name) for l in lectures] )
        
        self.history_filled = np.zeros( (self.len_lects, ), dtype=bool  )
        
        self.static_max_level = np.array( [l.max_level for l in self.lectures]  )
        
        
        self.lec_prob = np.zeros( (self.len_lects ,) )  
        self.lecture_pool = self.lectures
        
        self._update_probs() 
    
    def _update_probs(self): 
     
        
        levels = np.array( [l.get_level() for l in self.lectures]  )
        
        #diff_term       =   0.03*(self.latest_level.max(axis=1) - self.latest_level.min(axis=1))
        
        low_progress    =  3 *  np.array([1-l.get_diff() for l in self.lectures])
        forgetting_term  =  3*(self.max_acheived - levels) / self.static_max_level 
        history_term =  4*np.ones( (self.len_lects, ) ) * (self.history_filled == False) 
        
        self.lec_prob =  forgetting_term + history_term + low_progress #+ diff_term
        
        self.lec_prob_soft = softmax( self.lec_prob) 
        
        # self.bonus_matches = 2*int(round(self.lec_prob.mean() -0.49)  ) 
        
        # bonus_min = -self.ordinary_matches +1 
        # bonus_max = self.nbr_of_processes - self.ordinary_matches - 1
        
        self.bonus_matches = 0 #min(max(self.bonus_matches,bonus_min), bonus_max )    
            
    def get_next_lectures(self):        
    
        lecture_picks = self.nbr_of_processes - self.ordinary_matches - self.bonus_matches 
    
        lectures = np.random.choice( self.lecture_pool, lecture_picks , p = self.lec_prob_soft) 
        
        return list(lectures) + [None] * (self.ordinary_matches+self.bonus_matches)
        
    def add_lecture(self, lect): 
        assert False 
    
    def log_training(self, data, reward): 
        
        name    = data[0]
        level   = data[1]
        outcome = data[2]
        
        lec_index = self.lect_names.index( name )
        lect = self.lectures[ lec_index ]
        
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
        self.latest_level[lec_index, self.indices[lec_index] ]      = lect.level / lect.delta_level
        self.indices[lec_index]                                     = (self.indices[lec_index]+1) % self.history_size
        self.episodes[lec_index] += 1 
        
        self.max_acheived[lec_index] = max( self.max_acheived[lec_index], self.lectures[lec_index].get_level() * outcome )
        
        self.history_filled[lec_index] = self.history_filled[lec_index] or self.indices[lec_index]+10 >= self.history_size
        
        
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
            
            reward_success = self.rewards[lec_index, self.latest_hundred[lec_index] > 0.9 ].mean() 
            reward_fail = self.rewards[lec_index, self.latest_hundred[lec_index] < 0.9 ].mean() 

            
            #exceptions  = l.exceptions_thrown
            
            i=self.indices[lec_index]
            reward_flattened = np.concatenate( (self.rewards[lec_index, i:], self.rewards[lec_index, :i])) 
            reward_delta = reward_flattened[ self.history_size//2: ].mean() - reward_flattened[ :self.history_size//2 ].mean() 
            
            
            s_log = "{}, ep={:.0f}, lvl= {} ({:.0f})/{:.0f}, avg={:.0f}, p={:.0f}, rewrd= {:.2f} ({:.2f}), rewrd/dt= {:.2f}".format(name, episodes, lvl, max_acheived, max_lvl, 100*avg, 100*prob, reward_fail,  reward_success, reward_delta)
            s += s_log + "\n"
        return s
            

