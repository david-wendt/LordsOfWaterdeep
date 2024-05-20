from game.quests import * 
from game.resources import * 
from game.buildings import *

SCORE_PER_VP = 0.5 
# 'Score' here means 'one turn's worth of value'
# and at the end of the game you can get 2VP/turn.
# However, we need to make sure that completing a quest
# increases total score.
# Seems like the intended value is 0.25 (based on analysis here https://docs.google.com/spreadsheets/d/1rGbUNVHCKTy-D7s4yezfK91vA1HQFlCmWWY78Ize7xY/edit#gid=0
# to make the numbers work out for various quest categories), so 
# I think I'm happy upping it to 0.5 to incentivize quest completion

# Define the Lord cards (i.e. secret identities)
LORD_CARDS = []
for i,type1 in enumerate(QUEST_TYPES):
    for type2 in QUEST_TYPES[i+1:]:
        LORD_CARDS.append((type1, type2))
# TODO (later version): uncomment building lord card
# LORD_CARDS.append("Buildings")


# Define number of agents per player as a function of number of players
def agentsPerPlayer(numPlayers: int):
    '''
    Return the number of agents per player
    as a function of the number of players in 
    the game.
    
    Args:
        numPlayers: the total number of players in the game.
        
    Returns:
        the number of agents per player.
    '''
    if numPlayers == 2: return 4
    if numPlayers == 3: return 3
    if numPlayers == 4: return 2
    if numPlayers == 5: return 2 
    else: 
        raise ValueError("Number of players must be an integer between 2 and 5, inclusive.")
    
INTRIGUES = [
    'Choice of any resource', # 4 gold or 2 F/R or 1 W/C
] * 20
# TODO (later): figure out what to do for intrigue cards

REASSIGNED = '__reassigned__' # 'occupier' of a waterdeep harbor slot
    # once the agent there has been reassigned 

def main():
    pass


if __name__ == '__main__':
    main()