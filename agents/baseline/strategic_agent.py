import numpy as np 

from agents.agent import Agent
from game.buildings import Building, CustomBuilding
from game.quests import Quest
from game.game_info import *
from game.player import Player
from game.game import GameState
from game import utils 
from agents.baseline import strategy_utils

class AbstractStrategicAgent(Agent):
    def placeAgent(self, game: GameState, player: Player, 
                   actions: list[Building | CustomBuilding]):
        '''Choose a building in which to place an agent'''
        raise NotImplementedError("Override in subclasses")
    
    def chooseQuest(self, game: GameState, player: Player, 
                   actions: list[Quest]):
        '''Choose which quest from Cliffwatch to take'''
        raise NotImplementedError("Override in subclasses")
    
    def completeQuest(self, game: GameState, player: Player, 
                     actions: list[Quest]):
        '''Choose which active quest to complete, or to not complete any'''
        raise NotImplementedError("Override in subclasses")
    
    def purchaseBuilding(self, game: GameState, player: Player, 
                        actions: list[CustomBuilding]):
        '''Choose which building from Builder's Hall to purchase
            (out of the affordable subset)'''
        raise NotImplementedError("Override in subclasses")
    
    def chooseReward(self, game: GameState, player: Player, 
                    actions: list[Resources | FixedResources]):
        '''Choose which of two owner rewards to take as building owner,
        OR Choose which standard resource bundle to get as a reward 
            for 'Call in a Favor' intrigue card'''
        raise NotImplementedError("Override in subclasses")
    
    def playIntrigue(self, game: GameState, player: Player, 
                    actions: list[str]):
        '''Choose which intrigue card to play'''
        raise NotImplementedError("Override in subclasses")
    
    def giveResource(self, game: GameState, player: Player, 
                    actions: list[Player]):
        '''Choose which opponent to give a resource to'''
        raise NotImplementedError("Override in subclasses")

    def act(self, gameState, playerState, actions, score):
        '''
        Strategically choose an action to play. For each possible
        type of action, call a function that handles that case
        specifically. This wrapper is called for all `StrategicAgent`s,
        which should each override the above functions which are used here.
        
        (The casework here is the same as in featurize.getActionMask,
        with structure of assertions slightly changed but functionally
        the same.)
        '''

        if isinstance(actions[0], Building) or (
            isinstance(actions[0], CustomBuilding) 
                and actions[0].owner is not None):
            # Action type: Choose a building in which to place an agent
            for building in actions:
                if isinstance(building, CustomBuilding):
                    assert building.owner is not None 
                else:
                    assert isinstance(building, Building)

            self.placeAgent(gameState, playerState, actions)
        
        elif isinstance(actions[0], Quest):
            # Action type: Choose which quest from Cliffwatch to take
            assert len(actions) == NUM_CLIFFWATCH_QUESTS
            for i,quest in enumerate(actions):
                assert isinstance(quest, Quest)
            
            self.chooseQuest(gameState, playerState, actions)

        elif actions[0] == DO_NOT_COMPLETE_QUEST:
            # Action type: Choose which active quest to complete, or to not complete any
            for i,action in enumerate(actions):
                if i != 0: assert isinstance(action, Quest)
            
            self.completeQuest(gameState, playerState, actions)

        elif isinstance(actions[0], CustomBuilding) and actions[0].owner is None:
            # Action type: Choose which building from Builder's Hall to purchase
            # (out of the affordable subset)
            assert len(actions) <= NUM_BUILDERS_HALL
            for building in actions:
                assert isinstance(building, CustomBuilding) and building.owner is None
            self.purchaseBuilding(gameState, playerState, actions)

        elif isinstance(actions[0], FixedResources):
            # Action type: Choose which of two owner rewards to take as building owner
            assert len(actions) == 2 and isinstance(actions[1], FixedResources)
            self.chooseReward(gameState, playerState, actions)

        elif isinstance(actions[0], str):
            # Action type: Choose which intrigue card to play
            assert set(actions).issubset(INTRIGUES)
            self.playIntrigue(gameState, playerState, actions)

        elif isinstance(actions[0], Player):
            # Action type: Choose which opponent to give a resource to
            opponents = utils.getOpponents(gameState.players, playerState)
            assert actions == opponents and len(actions) == gameState.numPlayers - 1
            self.giveResource(gameState, playerState, actions)

        elif isinstance(actions[0], Resources):
            # Action type: Choose which standard resource bundle to get as a reward for 'Call in a Favor' intrigue card
            assert actions == STANDARD_RESOURCE_BUNDLES
            self.chooseReward(gameState, playerState, actions)

        else:
            raise ValueError("Unknown action type:", actions)
        
class BasicStrategicAgent(AbstractStrategicAgent):
    def placeAgent(self, game: GameState, player: Player, 
                   actions: list[Building | CustomBuilding]):
        '''Choose a building in which to place an agent'''
        waterdeepHarbors = utils.getWaterdeepHarbors(actions)
        if len(waterdeepHarbors) > 0 and player.numIntrigues() and np.random.rand() < 0.5:
            return actions.index(waterdeepHarbors[0])
        if BUILDERS_HALL in actions and np.random.rand() < 0.5:
            return actions.index(BUILDERS_HALL)
        resourcesNeeded = strategy_utils.resourcesNeeded(player)
        return np.argmax([
            resourcesNeeded.dot(building.rewards)
            for building in actions
        ])
    
    def chooseQuest(self, game: GameState, player: Player, 
                   actions: list[Quest]):
        '''Choose which quest from Cliffwatch to take.
        Choose the best lord-aligned quest if one is available,
        otherwise the best quest.'''
        quests = strategy_utils.rankQuests(actions, player.lordCard)
        lordQuests = strategy_utils.lordQuests(quests, player.lordCard)
        plotQuests = strategy_utils.plotQuests(quests)
        plotLordQuests = set(lordQuests).intersection(plotQuests)
        if len(plotLordQuests):
            return min(plotLordQuests)
        elif len(lordQuests):
            return lordQuests[0]
        elif len(plotQuests):
            return plotQuests[0]
        return quests[0]

    def completeQuest(self, game: GameState, player: Player, 
                     actions: list[Quest]):
        '''Choose which active quest to complete, or to not complete any.'''
        return self.chooseQuest(game, player, actions)
    
    def purchaseBuilding(self, game: GameState, player: Player, 
                        actions: list[CustomBuilding]):
        '''Choose which building from Builder's Hall to purchase
            (out of the affordable subset). 
            
            For now, pick the one which gives owners the rewards
            that the player currently needs'''
        resourcesNeeded = strategy_utils.resourcesNeeded(player)
        return np.argmax([
            building.ownerRewards.dot(resourcesNeeded)
            for building in actions
        ])
    
    def chooseReward(self, game: GameState, player: Player, 
                    actions: list[Resources | FixedResources]):
        '''Choose which of two owner rewards to take as building owner,
        OR Choose which standard resource bundle to get as a reward 
            for 'Call in a Favor' intrigue card'''
        
        resourcesNeeded = strategy_utils.resourcesNeeded(player)
        return np.argmax([
            resourcesNeeded.dot(resources) 
            for resources in actions
        ])
    
    def playIntrigue(self, game: GameState, player: Player, 
                    actions: list[str]):
        '''Choose which intrigue card to play'''
        opponents = utils.getOpponents(game.players, player)
        resourcesNeeded = strategy_utils.resourcesNeeded(player)
        return np.argmax([
            strategy_utils.scoreIntrigue(action, resourcesNeeded, opponents)
            for action in actions
        ])
    
    def giveResource(self, game: GameState, player: Player, 
                    actions: list[Player]):
        '''Choose which opponent to give a resource to.
        Pick the opponent with the lowest `public score`'''
        oppScores = [strategy_utils.playerPublicScore(player) for player in actions]
        return np.argmin(oppScores)
    
class StrategicAgent(BasicStrategicAgent):
    ''' TODO (Joey):
    Take a look at the BasicStrategicAgent class above that I wrote.
    Feel free to change anything if you think any of it doesn't make
    sense, but my idea was that it was about as simple as a somewhat
    'strategic' agent could get. 

    Override some (or all) of the functions defined there with 
    smarter implementations here. Your call on how smart to make them.
    We can also do an ExpertStrategicAgent or something to have 3 instead
    of 2 levels of agent quality. I don't think we necessarily have to 
    go overboard with making great strategic agents, but I think we 
    should at least have something reasonable, a little better than
    what I've implemented above.

    Ideas: 
    - better strategy for when to play waterdeep harbor. Maybe based on number 
        of agents in the game (to take scarcity into account), and/or 
        how useful your intrigue cards are (based on dot with resourcesNeeded)?
    - better strategy for when to play bulider's hall. Maybe based on number 
        of agents in the game (to take scarcity into account), and/or 
        how useful the owner rewards are (based on dot with resourcesNeeded)?
    - maybe avoid placing agents in custom buildings when they are owned by the rival
        (highest-public-score opponent)
    - maybe sometimes choose to get the castle if you are toward the end of turn order
    - for scoring a player (see strategy_utils.playerPublicScore), 
        maybe add points per owned building?
        I don't really want to include this in player.score
        since I don't want to directly incentivize the delayed reward of building costs
        but it probably should be taken into account here (though this player scoring
        should have a very minor impact overall)
    - take another look at FixedResources.dot(Resources) method. The VPs/Qs/Is are
        mildly busted, but I tried to make it reasonable with the proxy 'needed'
        values in strategy_utils.resourcesNeeded(). Maybe just improve this proxy?
        it's probably sufficient though, all things considered
    - in strategy_utils.scoreIntrigue, consider subtracting some score based on giving 
        resources, and the resources needed by other players. This seems convoluted
        and fairly unnecessary though, since we give resources to a non-rival opponent
    '''

def main():
    quests = QUESTS[:4]
    player = Player('david', Agent(), 4, ('Arcana', 'Skullduggery'))
    agent = BasicStrategicAgent()
    agent.chooseQuest(None, player, quests)