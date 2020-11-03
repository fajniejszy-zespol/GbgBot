
from ffai.core.load import *
import Lectures as gc
from Curriculum import Academy, LectureOutcome 
import itertools as it 


lectures_to_test = [gc.GameAgainstRandom()]  

def test_all_lectures(): 
    
    
    configs = ["ff-11", "ff-7", "ff-5", "ff-3", "ff-1"]
    for lect, config in it.product(lectures_to_test, configs): 
        
        while lect.get_level() < lect.max_level:   
            g = lect.reset_game(config) 
            assert g.home_agent.human 
            assert not g.away_agent.human 
            
            lect.increase_level() 
        
        
def test_academy(): 
    school = Academy(lectures_to_test)
    
    
    for lect in lectures_to_test: 
        for i in range(30): 
            outcome = LectureOutcome( lect, win=True )
            school.log_training(outcome)
    
    s = school.report() 
        


    

    