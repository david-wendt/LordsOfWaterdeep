import numpy as np
import pandas as pd 
from dataclasses import dataclass

from game.resources import FixedResources

# Define all quest types
# (setting as constants for autocomplete + limiting magic constants)
ARCANA = "Arcana"
PIETY = "Piety"
SKULLDUGGERY = "Skullduggery"
WARFARE = "Warfare"
COMMERCE = "Commerce"

QUEST_TYPES = [ARCANA, PIETY, SKULLDUGGERY, WARFARE, COMMERCE]

LETTER_TO_QUEST_TYPE = {qtype[0]: qtype for qtype in QUEST_TYPES}

DO_NOT_COMPLETE_QUEST = "Do not complete a quest"

@dataclass(frozen=True)
class Quest:
    ''' Class representing a quest '''
    name: str 
    type: str 
    requirements: FixedResources 
    rewards: FixedResources 
    plot: bool

    def __repr__(self) -> str:
        return f"{self.name} ({self.type}):\n\t\tRequires {self.requirements}\n\t\tRewards {self.rewards}"\
            + f"\n\t\tPlot Quest (+2VP/future quest of type): {self.plot}"

# Two quests that I originally edited:
#     Quest('Convert a Noble to Lathander EDITED', PIETY,
#         requirements=FixedResources(clerics=2, wizards=1),
#         rewards=FixedResources(quests=1, VPs=10)), # Changed from 8 to 10 to put on equal footing with the next two
#     Quest('Thin the City Watch EDITED', COMMERCE,
#         requirements=FixedResources(clerics=1, fighters=1, rogues=1, gold=4),
#         rewards=FixedResources(rogues=2, VPs=8)), # Seems to be OP otherwise (see spreadsheet)

# TODO (later): add plot quests 

def parseQuests() -> list[Quest]:
    fname = 'data/quests.csv'

    df = pd.read_csv(
        fname, comment='#'
    ).drop([
        'Net',
        'Net.1',
        'Profit',
        'Notes'
    ], axis=1)

    # Filter out plot quests
    plotQuests = df['Special benefits'].isna()
    df = df.loc[plotQuests,:].drop('Special benefits', axis=1)
    df = df.fillna(0.0)

    for col in df.columns:
        if col != df[col].dtype == np.float64:
            df[col] = df[col].astype(np.int64)

    quests = []
    for i,row in df.iterrows():
        quests.append(Quest(
            name=row['Name'],
            type=LETTER_TO_QUEST_TYPE[row['Type']],
            requirements=FixedResources(
                wizards=row['W'],
                clerics=row['C'],
                fighters=row['F'],
                rogues=row['R'],
                gold=row['G']
            ),
            rewards=FixedResources(
                wizards=row['W.1'],
                clerics=row['C.1'],
                fighters=row['F.1'],
                rogues=row['R.1'],
                gold=row['G.1'],
                intrigues=row['I'],
                quests=row['Quest'],
                VPs=row['VP']
            ),
            plot=bool(row['Plot'])
        ))

    return quests

QUESTS = parseQuests()

def main():
    # data = {qtype: [] for qtype in QUEST_TYPES}
    for quest in QUESTS:
        # data[quest.type].append(quest.rewards.VPs)
        print(quest)

    # for qtype,vps in data.items():
    #     print(qtype, len(vps), np.mean(vps))
    # print(len(QUESTS))

    # df = pd.read_csv(
    #     'data/quests.csv', comment='#'
    # ).drop([
    #     'Net',
    #     'Net.1',
    #     'Profit',
    #     'Notes'
    # ], axis=1)

    # print(df.groupby('Type').count())
    # print(df[['Type', 'Name', 'Special benefits']].loc[df.Type == 'C',])