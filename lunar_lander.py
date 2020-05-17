import gym, random, tempfile
import numpy as np
from gym import wrappers
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

class LearningAgent:
    def __init__(self, indim, odim):
        self.indim = indim
        self.odim = odim
        self.epsilon = 1
        self.minEpsilon = 0.001
        self.decayEpsilon = 0.99
        self.learningRate = 0.0001
        self.epochs = 1
        self.minibatchSize = 64
        self.memory = deque(maxlen=15000000)
        self.verbose = 0
        self.model = self.getModel()

    def getModel(self):
        model = Sequential()

        model.add(Dense(150, input_dim=self.indim, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.odim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learningRate))
        return model

    def appendToMemory(self, currentState, act, r, nextState, done):
        self.memory.append((currentState, act, r, nextState, done))

    def selectAction(self, currentState):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.odim)
        q = self.model.predict(currentState)
        return np.argmax(q[0])
        
    def replay(self):
#        minibatch = random.sample(self.memory, self.minibatchSize)
#        states, actions, rewards, next_states, dones = self.getMiniBatchItems(minibatch)
#
#        states = np.squeeze(states)
#        next_states = np.squeeze(next_states)
#
#        targets = rewards + 0.8*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)
#        targets_full = self.model.predict_on_batch(states)
#        ind = np.array([i for i in range(self.minibatchSize)])
#        targets_full[[ind], [actions]] = targets
#
#        self.model.fit(states, targets_full, epochs=1, verbose=0)
#
        minibatch = random.sample(self.memory, self.minibatchSize)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)
        
#        minibatch = random.sample(self.memory, self.minibatchSize)
#        minibatch = np.array(minibatch)

        y = rewards + 0.99*(np.amax(self.model.predict_on_batch(next_states), axis=1))*(1-dones)


        yTarget = self.model.predict_on_batch(states)
        ind = np.array([i for i in range(self.minibatchSize)])
        yTarget[[ind], [actions]] = y
        self.model.fit(states,
                       yTarget,
                       epochs=self.epochs,
                       verbose=self.verbose)

if __name__ == "__main__":

    env = gym.make('LunarLander-v2')

    indim = env.observation_space.shape[0]
    odim = env.action_space.n

    agent = LearningAgent(indim, odim)
    
    episodes = 1650

    rewardS = deque(maxlen=50)
    resultFrom550To600Episodes = []
    for e in range(episodes):
        currentReward = 0
        currentState = env.reset()
        currentState = np.reshape(currentState, (1, indim))

        for step in range(1000):
#            if e > 550:
#                env.render()
                
            act = agent.selectAction(currentState)
            nextState, reward, done, info = env.step(act)
            currentReward += reward
            nextState = np.reshape(nextState, (1, indim))
            agent.appendToMemory(currentState, act, reward, nextState, done)
            currentState = nextState
            if len(agent.memory) > agent.minibatchSize and e < 1551:
                agent.replay()
            
            if done:
                break
        if e > 1550:
            resultFrom550To600Episodes.append(currentReward)
#            if currentReward > 200:
#                resultUpper200 += 1
#            elif currentReward > 100:
#                resultFrom100to200 += 1
#            elif resultUpper0 > 0:
#                resultFrom0To100 += 1
#            else:
#                resultLess0 += 1
                
        if agent.epsilon > agent.minEpsilon:
            agent.epsilon = max(agent.minEpsilon, agent.epsilon*agent.decayEpsilon)
            
        rewardS.append(currentReward)
        print ('Number of episode: ', e, ' SCORE: ', '%.2f' % currentReward, ' MEAN: ', '%.2f' % np.average(rewardS), ' STEP COUNT: ', step, ' epsilon: ', '%.2f' % agent.epsilon)
    
#    print("200+ = ",resultUpper200)
#    print("From 100 to 200 = ",resultFrom100to200)
#    print("From 0 to 100 = ",resultFrom0To100)
#    print("Less than 0 = ",resultLess0)
    num_bins = 5
    n, bins, patches = plt.hist(resultFrom550To600Episodes, num_bins, facecolor='blue', alpha=0.5)
    plt.show()
    
    env.close()
