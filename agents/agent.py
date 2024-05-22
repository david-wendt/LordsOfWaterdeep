class Agent():
    def __init__(self) -> None:
        pass 

    def act(self, state, playerState, actions, score) -> int:
        ''' Override this in subclasses'''
        raise NotImplementedError