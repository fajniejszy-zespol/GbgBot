

import random

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
    
    def reset_game(self, game): raise "not implemented"
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
        
    def reset_game(self, game): raise "not implemented"
    def training_done(self, game): raise "not implemented"        
    def allowed_fail_rate(self): raise "not implemented" 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    