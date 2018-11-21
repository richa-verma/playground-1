import numpy as np
from pommerman import agents
from keras.models import load_model

class RL_learned_agent(agents.BaseAgent):
    def __init__(self):
        super().__init__()
        self.actor_model = load_model('conv.h5')
        
    def act(self, obs, action_space):
        
        def featurize(obs):
           
            board = obs['board']
        
            # convert board items into bitmaps
            maps = [board == i for i in range(10)]
            maps.append(obs['bomb_blast_strength'])
            maps.append(obs['bomb_life'])
        
            # duplicate ammo, blast_strength and can_kick over entire map
            maps.append(np.full(board.shape, obs['ammo']))
            maps.append(np.full(board.shape, obs['blast_strength']))
            maps.append(np.full(board.shape, obs['can_kick']))
        
            # add my position as bitmap
            position = np.zeros(board.shape)
            position[obs['position']] = 1
            maps.append(position)
        
            # add teammate
            if obs['teammate'] is not None:
                maps.append(board == obs['teammate'].value)
            else:
                maps.append(np.zeros(board.shape))
        
            # add enemies
            enemies = [board == e.value for e in obs['enemies']]
            maps.append(np.any(enemies, axis=0))
            return np.stack(maps, axis=2)
        
        feat = featurize(obs)
        probs = self.actor_model.predict(feat[np.newaxis])
        
        action = np.argmax(probs)
        return action.item()
