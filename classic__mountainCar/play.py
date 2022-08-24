import gym
import numpy as np
import h5py

######## Pseudocode #################
### 
### 1 | Load Mountain Car environment
### 2 | Load q table from storage
### 3 | Some parameters 
### 4 | Let's load the Q-Table for the agent, and let it play using that
### 5 | Close the environment
###
#####################################

def main():
    def get_discrete_state(state):
        discrete_state = (state - env.observation_space.low) / discrete_os_win_size
        return tuple(discrete_state.astype(np.uint))

    ### 1 | Load Mountain Car environment

    env = gym.make("MountainCar-v0")
    env.reset()

    ### 2 | Load q table from storage

    f = h5py.File("qtable.hdf5", "r")
    q_table = f['qtable']


    ### 3 | Some parameters 
    TOTAL_EPISODES = 100

    DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE


    ### 4 | Let's load the Q-Table for the agent, and let it play using that
    for episode in range(TOTAL_EPISODES):
        done = False
        discrete_state = get_discrete_state(env.reset())
        while not done:
            action = np.argmax(q_table[discrete_state])
            new_state, reward, done, _ = env.step(action)
        
            new_discrete_state = get_discrete_state(new_state)

            env.render()
            discrete_state = new_discrete_state

    ### 5 | Close the environment
    env.close()


if __name__ == "__main__":
    main()

