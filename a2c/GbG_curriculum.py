import ffai
from ffai.core.model import Square, Action
from ffai.core.table import ActionType 
import random 
from random import randint 
from pdb import set_trace
from copy import deepcopy 

def get_home_players(game): 
    players = [p for p in game.state.home_team.players if p.position is not None ]
    return random.sample(players, len(players))
    
def get_away_players(game): 
    players = [p for p in game.state.away_team.players if p.position is not None ]
    return random.sample(players, len(players))

    
    
def scatter_ball(game, steps, from_position): 
    # scatters ball a certain amount of steps away for original position 
    # checks are done so it's not out of bounds or on a player 
    board_x_max = len(game.state.pitch.board[0]) -2  
    board_y_max = len(game.state.pitch.board) -2
    
    steps = int(steps)
    assert steps > 0 

    avail_squares = steps*8 
    
    squares_per_side = 2*steps 
    
    
    while True: 
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
            
    game.get_ball().move_to(position)

def move_player_within_square(game, player, upper_left, lower_right): 
    # places the player at a random position within the given square. 
    
    x_min = upper_left[0]
    x_max = lower_right[0]
    y_min = upper_left[1]
    y_max = lower_right[1]
    
    while True: 
        x = randint(x_min, x_max)
        y = randint(y_min, y_max)
        if game.state.pitch.board[y][x] is None: 
            break 
    
    game.move(player, Square(x,y)) 
    
def move_player_out_of_square(game, player, upper_left, lower_right):
    # places the player at a random position that is not in the given square. 
    
    x_min = upper_left[0]
    x_max = lower_right[0]
    y_min = upper_left[1]
    y_max = lower_right[1]
    
    board_x_max = len(game.state.pitch.board[0]) -2  
    board_y_max = len(game.state.pitch.board) -2
    
    while True: 
        x = randint(1, board_x_max)
        y = randint(1, board_y_max)
        
        if x_min <= x <= x_max and y_min <= y <= y_max: 
            continue 
        
        if game.state.pitch.board[y][x] is None: 
            break 
    
    game.move(player, Square(x,y)) 
    
class Lecture: 

    def __init__(self, name, max_level, delta_level = 0.1): 
        self.name           = name 
        self.level          = 0 
        self.max_level      = max_level
        self.delta_level    = delta_level
    
    
    def increase_diff(self): 
        self.level = min( self.max_level, self.level +self.delta_level )
    def decrease_diff(self):
        self.level = max(              0, self.level -self.delta_level )        
    def get_diff(self): 
        return min(self.level, self.max_level) / self.max_level 
    def get_level(self): 
        return min( int(self.level), self.max_level) 
    
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
                #print("Assertion error!")
                #print(proc)
                #print(a)
                #print(e)
            #except AttributeError: 
            #    print 
        
        #game.step( env.game._forced_action() )
        
        board_x_max = len(game.state.pitch.board[0]) 
        board_y_max = len(game.state.pitch.board)
        
        #reset players to up and in the buttom wing
        y_pos = 0 #used to be 1 
        for players in [game.state.home_team.players, game.state.away_team.players]: 
            next_x_pos = 2
            for player in players: 
                if player.position is not None:
                    # Set to ready
                    player.state.reset()
                    player.state.up = True
                    
                    while True:  
                        next_x_pos += 1 
                        try: 
                            assert next_x_pos < board_x_max 
                        except: 
                            print(next_x_pos, "<", board_x_max)
                            exit() 
                        
                        position = ffai.core.model.Square(next_x_pos, y_pos)
                        
                        if game.state.pitch.board[position.y][position.x] is None:
                            break 
                    
                    game.move(player, position) 
            y_pos = board_y_max -1 #used to be 2
        
        game.set_available_actions()
        
        self._reset_lecture(game)
    
    def _reset_lecture(self, game): raise "not implemented"
    def training_done(self, game): raise "not implemented"        
    def allowed_fail_rate(self): raise "not implemented" 
    
    
    
class Academy: 
    
    def __init__(self, lectures): 
        self.lectures       = {} 
        for l in lectures: 
            self.lectures[l.name] = l 

        
        self.len_lects = len(self.lectures) 
            
        # difficulty history for plots 
        
    def get_next_lecture(self):
        #TODO: modify distribution according to progress 
        
        
        return random.choice( list(self.lectures.values()) ) 
        
    def log_training(self, name, outcome): 
        
        # increase difficulty 
        if outcome == True: 
            self.lectures[ name ].increase_diff() 
        
        # decrease difficulty
        elif self.lectures[ name ]. allowed_fail_rate() < random.random():
            self.lectures[ name ].decrease_diff()
        
        # unchanged difficulty  
        else: 
            pass 
            
        
    def report_training(self, filename=None): 
        # render plots 
        
        
        s="reporting from Gbg Trainer: "
        for l in self.lectures.values(): 
            s += l.name + " - " + "{:.4f}".format(l.get_diff()) +" (" + str(l.get_level()) + "/" +str(l.max_level) + ")" 
        return s
            
class Scoring(Lecture): 
    def __init__(self): 
        self.dst_mod = 7
        self.ball_mod = 3
        super().__init__("Scoring", self.dst_mod * self.ball_mod -1) 
        
    def _reset_lecture(self, game): 

        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        dst_to_td = (level % self.dst_mod) +2 
        ball_start = (level // self.dst_mod) % self.ball_mod 

        home_players = get_home_players(game)
        
        #setup ball carrier
        p = home_players.pop() 
        game.move(p, Square(dst_to_td, randint(2, board_y_max -2 ) ) )
        
        if ball_start == 0: 
            #Give ball to player 
            game.get_ball().move_to( p.position ) 
            game.get_ball().is_carried = True 
        else: 
            scatter_ball(game, ball_start, p.position)
            
            
        #place rest of team at random places a bit a way 
        for p in home_players: 
            move_player_within_square(game, p, [2*board_x_max//3, 1], [board_x_max, board_y_max] )
        
        #place away team on other side of pitch, of the wings 
        for p in get_away_players(game): 
            move_player_within_square(game, p, [dst_to_td+2+ball_start, 1], [board_x_max, board_y_max] )
        
        self.turn = deepcopy(game.state.home_team.state.turn)  
    
    def training_done(self, game): 
        training_complete = self.turn  !=  game.state.home_team.state.turn
        training_outcome = game.state.home_team.state.score > 0 
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        return 0 
    
class HandoffAndScore(Lecture):    
    def __init__(self): 
        self.dst_mod = 5
        self.ball_mod = 4
        super().__init__("HandoffAndScore", self.dst_mod * self.ball_mod -1) 
    def _reset_lecture(self, game): 
        
        assert game.is_handoff_available()
        # ### CONFIG ### # 
        board_x_max = len(game.state.pitch.board[0]) -2  
        board_y_max = len(game.state.pitch.board) -2
    
        #Level configuration 
        level = self.get_level()        
        dst_to_td = (level % self.dst_mod) +2 
        ball_start = (level // self.dst_mod) % self.ball_mod 
        
        #get players 
        home_players = get_home_players(game)
        
        #setup scorer 
        p_score = home_players.pop() 
        p_score_x = dst_to_td
        p_score_y = randint(2, board_y_max -1 )
        game.move(p_score, Square( p_score_x, p_score_y) )
        
        #setup passer
        p_pass  = home_players.pop() 
        p_pass_intended_moves = max(ball_start - 1, 0)
        p_pass.state.moves = p_pass.get_ma()  -  p_pass_intended_moves
        p_pass_x = p_score_x + 1 + p_pass_intended_moves
        dy_max =  1 + p_pass_intended_moves
        p_pass_y = p_score_y + randint(-dy_max, dy_max)
        p_pass_y = min(p_pass_y, board_y_max)
        p_pass_y = max(p_pass_y, 1)
        game.move(p_pass, Square( p_pass_x, p_pass_y) )
        game.get_ball().move_to( p_pass.position ) 
        game.get_ball().is_carried = True 
        
        
        
        if ball_start == 0: 
            
            
            a = Action(action_type=ActionType.START_HANDOFF, position = p_pass.position, player = p_pass )
            game.step(a)
            
            # Start the hand_off action 
        
        #place rest of team at random places a bit a way 
        for p in home_players: 
            move_player_out_of_square(game, p, [0, 0], [p_pass_x +2 , board_y_max] )
        
        #place away team on other side of pitch, of the wings
        
        x_min = 0
        x_max = p_pass_x+1
        y_min = min(p_score_y, p_pass_y)-1
        y_max = max(p_score_y, p_pass_y)+1
        
        for p in get_away_players(game): 
            move_player_out_of_square(game, p, [x_min, y_min], [x_max , y_max] )
        
        
        self.turn = deepcopy(game.state.home_team.state.turn)  
        
        
    def training_done(self, game): 
        training_complete = self.turn  !=  game.state.home_team.state.turn
        training_outcome = game.state.home_team.state.score > 0 
        return training_complete, training_outcome
    
    def allowed_fail_rate(self): 
        return 0 
        
#class PassAndScore(Lecture):    

