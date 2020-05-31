

import random

def get_home_players(game): 
    return [p for p in game.state.home_team.players if p.position is not None ]

class Lecture: 

    def __init__(self, name, nbr_of_levels): 
        self.name           = name 
        self.level          = 1 
        self.nbr_of_levels  = nbr_of_levels
    
    def increase_diff(self): 
        self.level = min( self.nbr_of_levels, self.level + 1)
    def decrease_diff(self):
        self.level = max( 1, self.level -1 )        
    def get_diff(self): 
        return (self.level -1) / (self.nbr_of_levels - 1)
    def get_level(self): 
        return self.level
    
    def reset_env(self, env): 
        env.reset()
        game = env.game 
        
        while game.get_procedure().__class__.__name__ != "Turn": 
            game.step( env.game._forced_action() )
        game.step( env.game._forced_action() )
        
        board_x_max = len(game.state.pitch.board[0]) 
        board_y_max = len(game.state.pitch.board)
        
        #reset players to up and in the buttom wing
        y_pos = 1
        for players in [game.state.home_team.players, game.state.away_team.players]: 
            next_x_pos = 2
            for player in players: 
                if player.position is not None:
                    # Set to ready
                    player.state.reset()
                    player.state.up = True
                    
                    position = ffai.core.model.Square(next_x_pos, y_pos)
                    while game.state.pitch.board[position.y][position.x] is not None:
                        next_x_pos += 1 
                        if next_x_pos >= board_x_max: 
                            print("ERRORORORORO ")
                            exit()
                        position = ffai.core.model.Square(next_x_pos, y_pos)
                    
                    game.move(player, position) 
            y_pos = board_y_max -2 

        self._reset_lecture(game)
    
    
    def _reset_lecture(self, game): raise "not implemented"
    def training_done(self, game): raise "not implemented"        
    def allowed_fail_rate(self): raise "not implemented" 
    
    
    
class GbgTrainer: 
    
    def __init__(self, lectures): 
        self.lectures       = {} 
        for l in lectures: 
            self.lectures[l.name] = l 

        
        self.len_lects = len(self.lectures) 
            
        # difficulty history for plots 
        
    def get_next_lecture(self):
        #TODO: modify distribution according to progress 
        return random.choice(self.lectures) 
        
    def log_training(self, result): 
        outcome = result["outcome"]
        name = result["name"]
        
        # increase difficulty 
        if outcome == True: 
            self.lectures[ name ].increase_diff() 
        
        # decrease difficulty
        elif self.lectures[ name ]. allowed_fail_rate() < random.random():
            self.lectures[ name ].decrease_diff()
        
        # unchanged difficulty  
        else: 
            pass 
            
        
    def report_training(self, filename): 
        # render plots 
        print("reporting from Gbg Trainer")
        for l in self.lectures.values(): 
            print(l.name, " - ", l.get_diff() ) 
            
            
class Scoring(Lecture): 
    def __init__(self): 
        super().__init__("Scoring", 5) 
        
    def _reset_lecture(self, game): 
        
        home_players = random.sample( get_home_players(game) )
        
        #away_players = random.sample(game.state.home_team.players)
    
        #setup ball carrarier 
        p = home_players.pop() 
        
        #place rest of team at random places a bit a way 
        pass 
        
        #place away team on other side of pitch, of the wings 
        pass 
        
    
    def training_done(self, game): raise "not implemented"        
    def allowed_fail_rate(self): raise "not implemented" 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    