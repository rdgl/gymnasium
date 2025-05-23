import gymnasium as gym
import numpy as np
import pickle

def run(episodes, isTraining = True, render = False):
    
    env = gym.make('MountainCar-v0', render_mode = 'human' if render else None)
    
    pos_space = np.linspace(env.observation_space.low[0], env.observation_space.high[0], 20)
    vel_space = np.linspace(env.observation_space.low[1], env.observation_space.high[1], 20)
    
    if(isTraining):
        q = np.zeros((len(pos_space), len(vel_space), env.action_space.n))
    else:
        f = open('mountain_car.pkl', 'rb')
        q = pickle.load(f)
        f.close()
    
    alpha = 0.1
    gamma = 0.99
    
    epsilon = 1
    ep_decay_rate = 0.0005
    random_gen = np.random.default_rng()
    
    for i in range(episodes):
        state = env.reset()[0]
        state_p = min(np.digitize(state[0], pos_space), len(pos_space)- 1)
        state_v = min(np.digitize(state[1], vel_space), len(vel_space) - 1)
        
        rewards = 0
        terminate = False
        
        while(not terminate):
            
            if isTraining and random_gen.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, :])
                
            new_state, reward , terminate, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            
            if isTraining:
                q[state_p, state_v, action] += alpha*(reward + gamma * np.max(q[new_state_p, new_state_v, :]) - q[state_p, state_v, action])

            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            
            rewards += reward
            
        epsilon = max(0, epsilon - ep_decay_rate)
        
        if (i) % 100 == 0:
            print(f"Episode: {i} | Rewards: {rewards} | Epsilon: {epsilon}")
        
    env.close()
    
    if isTraining:
        f = open('mountain_car.pkl', 'wb')
        pickle.dump(q, f)
        f.close()
        
if __name__ == '__main__':
     run(2500, isTraining=True, render=False) # while training
    
    # run(1, isTraining=False, render=True) # for testing and evaluation

