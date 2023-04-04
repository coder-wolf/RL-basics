import gymnasium as gym
import numpy as np
import h5py # For saving the weights 

### We'll use this function to save trained model for future use
def saveWeights(q_table):
    file = h5py.File("qtable.hdf5", "w")
    dataset = file.create_dataset("qtable", (q_table.shape), dtype="f", data=q_table)

def main():
    def get_discrete_state(state):
        discrete_state = (state - env.observation_space.low) / discrete_os_win_size
        return tuple(discrete_state.astype(np.uint))

    ########### "Q-Learning Algorithm" -- Pseudo-code ############
    
    ### 1 | Load the Mountain Car environment
    ### 2 | Set the Learning parameters
    ### 3 | Initialize the "Q-Table"
    ### 4 | Train the Agent, and saving the weights in q_table
    ###   "---.
    ###    4.1 | Chose an action based on E-greedy method. 
    ###    4.2 | Perform the action, choosen in step-4.1
    ###    4.3 | Convert the state, to a discrete state, for the Q-table. (Only necessary in image/video inputs)
    ###        "---.
    ###      4.4.1 | Calculating the "Max Future Reward, for "Bellman Equation"
    ###      4.4.2 | Calculating the "Q Value" using "bellman equation"
    ###      4.4.3 | Updating the "Q-Table" with new Q Value
    ###   |--------
    ### 5 | Save weights for future usage. | This is important

    ##############################################################

### 1| Load the Mountain Car environment
    env = gym.make("MountainCar-v0")
    env.reset()


    ### 2 | Set the Learning parameters

    # Quick NOTE: LEARNING_RATE vs epsilon difference?
    #
    # Ans :- learning rate is associated with how big you take a leap and 
    #        epsilon is associated witg how random you take an action
    EPISODES = 25000
    DISCOUNT = 0.95

    LEARNING_RATE = 0.1 # See note above ↑

    epsilon = 0.9 # See note above ↑
    START_EPSILON_DECAYING = 1
    END_EPSILON_DECAYING = EPISODES // 2
    epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)



    # Quick Note: EPISODES ? 
    #
    # Ans :- How long you want to train the agent.
    #        Same as epoch in Deep Learning
    SHOW_EVERY = 2000
    DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
    discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

    
    ### 3 | Initialize the "Q-Table"
    
    q_table = np.random.uniform(low=-2, high=0, size = (DISCRETE_OS_SIZE + [env.action_space.n]))


    ### 4 | Training the Agent, and saving the weights in q_table
    for episode in range(EPISODES):
        ### Just rendering the process.
        ##  Every 5000 Steps.
        ##  For seeing what's going on, with the agent.
        if episode % SHOW_EVERY == 0:
            print(episode)
            render = True
        else:
            render = False


        discrete_state = get_discrete_state(env.reset()) # This is the "Current State"
        done = False
        
        while not done:
            ### 4.1 | Chosing an action based on E-greedy method. 
            if np.random.random() > epsilon:
                action = np.argmax(q_table[discrete_state])
                # This action is being taken from the learnt "Q-Table" 
            else:
                action = np.random.randint(0, env.action_space.n)
                # This action is totally random.


            ### 4.2 | Now, we're taking the action we've just,
            #         and applying that on the environment
            new_state, reward, done, _ = env.step(action)

            ### 4.3 | Now, formatting(in a way) the state into
            #         a format, that we can save in our
            #         q table
            new_discrete_state = get_discrete_state(new_state)
            if render:
                env.render()

            if not done:
                ### 4.4.1 | Calculating the "Max Future Reward, for "Bellman Equation"
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action, )]


                ### 4.4.2 | Calculating the "Q Value" using bellman equation
                new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

                ### 4.4.3 | Updating the "Q-Table" with new Q Value
                q_table[discrete_state + (action, )] = new_q

            elif new_state[0] >= env.goal_position:
                q_table[discrete_state + (action, )] = 0

            discrete_state = new_discrete_state

        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value
            # Why are we doing this?
            # Ans :- Because we wanna reduce how many random actions we take,
            #        gradually, over time.
            #        Else, our model will never converge,
            #        and the training will go on forever.

    env.close()
    ## Training's complete.


    ### 5 | Save weights for future usage. | This is important
    saveWeights(q_table)

if __name__ == "__main__":
    main()
