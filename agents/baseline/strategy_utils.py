from game.quests import Quest
from game.player import Player
from game.game_info import *

PLOT_QUEST_VALUE = 4 # Based on breakeven value in spreadsheet https://docs.google.com/spreadsheets/d/1rGbUNVHCKTy-D7s4yezfK91vA1HQFlCmWWY78Ize7xY/edit#gid=0

def evaluateQuest(quest: Quest, lordCard: list[str]):
    return (
        quest.rewards.clerics
        + quest.rewards.wizards
        + quest.rewards.fighters / 2.
        + quest.rewards.rogues / 2.
        + quest.rewards.gold / 4.
        + quest.rewards.VPs * SCORE_PER_VP
        + quest.rewards.intrigues / 2.
        + quest.rewards.quests / 2.
        - quest.requirements.clerics
        - quest.requirements.wizards
        - quest.requirements.fighters / 2.
        - quest.requirements.rogues / 2.
        - quest.requirements.gold / 4.
        + LORD_BONUS_VP * (quest.type in lordCard) * SCORE_PER_VP
        + quest.plot * PLOT_QUEST_VALUE
    )

def rankQuests(quests: list[Quest], lordCard: list[str]):
    return sorted(quests, key=lambda quest: evaluateQuest(quest, lordCard), reverse=True)

def lordQuests(quests: list[Quest], lordCard: list[str]):
    questTypes = np.array([quest.type for quest in quests])
    lordType1,lordType2 = lordCard
    return np.where(np.logical_or(
        questTypes == lordType1,
        questTypes == lordType2
    ))[0]

def plotQuests(quests: list[Quest]):
    return np.where([quest.plot for quest in quests])[0]

def np_topk(arr, k):
    # Get top k elements from array `arr` (NOT sorted)
    return np.partition(arr,k)[:k]

def scoreCompletedQuests(quests: list[Quest]):
    n_plot_quests = 0
    quest_type_counts = [0] * len(QUEST_TYPES)
    for quest in quests:
        if quest.plot:
            n_plot_quests += 1
        quest_type_counts[QUEST_TYPES.index(quest.type)] += 1

    # Treat the two most frequent quest types as the guessed lord card types
    guessed_lord_card_counts = np_topk(quest_type_counts,2)
    n_lord_quests_completed = guessed_lord_card_counts.sum()
    return LORD_BONUS_VP * n_lord_quests_completed * SCORE_PER_VP + PLOT_QUEST_VALUE * n_plot_quests

def playerPublicScore(player: Player):
    '''Compute an RL agent's score, but only with contributions
    from public information.'''

    # 'Score' here represents number of turns' worth of resources acquired
    score = (
        player.resources.clerics
        + player.resources.wizards
        + player.resources.fighters / 2.
        + player.resources.rogues / 2.
        + player.resources.gold / 4.
        + player.resources.VPs * SCORE_PER_VP
    )

    # Intrigues 
    score += player.numIntrigues() / 2.

    # Castle Waterdeep
    score += player.hasCastle / 2.

    # Active quests
    score += len(player.activeQuests) / 2.

    # Completed quests (lord + plot)
    score += scoreCompletedQuests(player.completedQuests)

    # TODO: add bulidings owned?

    return score 

def resourcesNeeded(player: Player):
    resources = Resources()
    for quest in player.activeQuests:
        reqs,nQ,nI = quest.requirements.toResources()
        assert nQ == nI == 0
        resources += reqs

    ownedResources = player.resources
    ownedResources.VPs = 0
    resources -= ownedResources

    # Proxy for value of a VP/Q/I relative to value of resources needed
    scaled_mean = resources.scaled_mean()
    resources.VPs += scaled_mean / SCORE_PER_VP 
    resources = resources.toFixedResources()

    # 10 is max number of active quests, and proxy for desired number of intrigues 
    resources.quests += 10 - len(player.activeQuests)
    resources.intrigues += 10 - player.numIntrigues()

    return resources

def scoreRemoval(resourcesTaken: Resources, 
                 resourcesReceived: Resources, 
                 resourcesNeeded: Resources,
                 opponents: list[Player], 
                 rival: Player):
    nResourcesReceived = 0
    for opp in opponents:
        if resourcesTaken.dot(opp.resources) == 0:
            nResourcesReceived += 1
    return (
        resourcesNeeded.dot(nResourcesReceived * resourcesReceived)
        + resourcesTaken.dot(rival.resources)
    )
    

def scoreIntrigue(intrigue: str, resourcesNeeded: FixedResources, opponents: list[Player]):
    assert intrigue in INTRIGUES
    rival = max(opponents, key=playerPublicScore)

    if intrigue == 'Call in a Favor': #"Choose to take 4 gold or 2 fighters or 2 rogues or 1 wizard or 1 cleric",
        return max([resourcesNeeded.dot(resources) for resources in STANDARD_RESOURCE_BUNDLES])
    elif intrigue == 'Lack of Faith': #"Each opponent removes 1 cleric. For each that cannot do so, score 2 VP",
        return scoreRemoval(Resources(clerics=1), Resources(VPs=2), resourcesNeeded, opponents, rival)
    elif intrigue == 'Ambush': #"Each opponent removes 1 fighter. For each that cannot do so, receive 1 fighter",
        return scoreRemoval(Resources(fighters=1), Resources(fighters=1), resourcesNeeded, opponents, rival)
    elif intrigue == 'Assassination': #"Each opponent removes 1 rogue. For each that cannot do so, receive 2 gold",
        return scoreRemoval(Resources(rogues=1), Resources(gold=2), resourcesNeeded, opponents, rival)
    elif intrigue == 'Arcane Mishap': #"Each opponent removes 1 wizard. For each that cannot do so, receive 1 intrigue card",
        return scoreRemoval(Resources(wizards=1), FixedResources(intrigues=1), resourcesNeeded, opponents, rival)
    elif intrigue == 'Spread the Wealth': #"Take 4 gold, choose 1 opponent to get 2 gold",
        return resourcesNeeded.dot(Resources(gold=4))
    elif intrigue == 'Graduation Day': #"Take 2 wizards, choose 1 opponent to get 1 wizard",
        return resourcesNeeded.dot(Resources(wizards=2))
    elif intrigue == 'Conscription': #"Take 2 fighters, choose 1 opponent to get 1 fighter",
        return resourcesNeeded.dot(Resources(fighters=2))
    elif intrigue == 'Good Faith': #"Take 2 clerics, choose 1 opponent to get 1 cleric",
        return resourcesNeeded.dot(Resources(clerics=2))
    elif intrigue == 'Crime Wave': #"Take 2 rogues, choose 1 opponent to get 1 rogue",
        return resourcesNeeded.dot(Resources(rogues=2))