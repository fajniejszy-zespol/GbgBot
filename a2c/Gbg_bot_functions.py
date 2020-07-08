import ffai 
from ffai.core.table import ActionType, Skill
from ffai.core.model import Square, Action


def action_type_available(action_type, game): 
    return action_type in [a.action_type for a in game.get_available_actions()]
            
def if_kick_receive(game): 
    return action_type_available(ActionType.RECEIVE, game)
def choose_receive(game): 
    #print("in action function")
    return Action(ActionType.KICK)

def if_place_ball(game): 
    return action_type_available( ActionType.PLACE_BALL, game )
def choose_place_ball_middle(game): 
    board_x_max = len(game.state.pitch.board[0]) 
    board_y_max = len(game.state.pitch.board) 
    
    y = int(board_y_max/2)+1
    x = int(board_x_max/4)
    
    left_center = Square(x, y) 
    action = Action(ActionType.PLACE_BALL, position=left_center)
    
    if game._is_action_allowed(action):
        #print("correct place ball guess")
        return action 
    else: 
        print("wrong place ball guess")
        right_center = Square( int(board_x_max*3/4), y)
        return Action(ActionType.PLACE_BALL, position=right_center)

def is_block_dice(game): 
    actions = [a.action_type for a in game.get_available_actions()]
    block_dices = [ ActionType.SELECT_PUSH,
                    ActionType.SELECT_ATTACKER_DOWN,
                    ActionType.SELECT_BOTH_DOWN,
                    ActionType.SELECT_DEFENDER_STUMBLES,
                    ActionType.SELECT_DEFENDER_DOWN]
    return any( [ (bd in actions) for  bd in block_dices]) 

def block(game): #stolen from scripted bot 
    """
    Select block die or reroll.
    """
    
    #TODO - remove game
    proc = game.get_procedure()
    
    if isinstance(proc, ffai.Reroll): 
        proc = proc.context 
    
    # Get attacker and defender
    attacker = proc.attacker
    defender = proc.defender
    is_blitz = proc.blitz
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
