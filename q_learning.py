import random
import gym
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import time


def update_q_table(Q, state, action, reward, next_state, alpha, Gamma):
    """
    Update the Q table for a given action-state pair following the Q-learning algorithm.
    Takes as input the Q function, action-state pair, the reward, the next state,
    learning rate, and discount factor. Returns the updated Q table.
    """
    max_q_next_state = np.max(Q[next_state, :])
    Q[state, action] = Q[state, action] + alpha * (reward + Gamma * max_q_next_state - Q[state, action])
    return Q


def epsilon_greedy(Q, state, Epsilon):
    """
    Implements the epsilon-greedy algorithm.
    Takes as input the Q function for all states, a state, and the exploration rate.
    Returns the action to take following the epsilon-greedy strategy.
    """
    if random.uniform(0, 1) < Epsilon:
        # Choose a random action
        action = env.action_space.sample()
    else:
        # Choose the action with the highest Q value
        action = np.argmax(Q[state, :])
    return action


if __name__ == "__main__":
    env = gym.make("Taxi-v3")
    env.reset()
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    # Learning parameters
    alpha = 0.05
    Gamma = 0.95
    Epsilon = 0.1

    # Training parameters
    n_epochs = 2000  # Number of episodes
    max_iterations_per_epoch = 150  # Max iterations per episode
    total_rewards = []

    for epoch in range(n_epochs):
        total_reward = 0
        state, _ = env.reset()

        for _ in range(max_iterations_per_epoch):
            action = epsilon_greedy(Q=Q, state=state, Epsilon=Epsilon)
            next_state, reward, done, _, info = env.step(action)

            total_reward += reward

            Q = update_q_table(
                Q=Q, state=state, action=action, reward=reward, next_state=next_state,
                alpha=alpha, Gamma=Gamma
            )

            state = next_state  # Update state

            if done:
                break  # Episode finished

        print("Episode #", epoch, " : Total Reward = ", total_reward)
        total_rewards.append(total_reward)

    print("Average Reward = ", np.mean(total_rewards))

    # Plot the rewards as a function of epochs
    plt.plot(total_rewards)
    plt.xlabel('Epoch')
    plt.ylabel('Total Reward')
    plt.title('Total Rewards over Epochs')
    plt.show()

    print("Training finished.\n")

    """
    Evaluate the Q-learning algorithm
    """

    def evaluate_policy(Q, env, n_episodes=100):
        total_rewards = []
        for episode in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = np.argmax(Q[state, :])  # Choose best action
                next_state, reward, done, _, info = env.step(action)
                episode_reward += reward
                state = next_state
            total_rewards.append(episode_reward)
        avg_reward = np.mean(total_rewards)
        print(f"Average reward over {n_episodes} episodes: {avg_reward}")
        return total_rewards

    eval_rewards = evaluate_policy(Q, env, n_episodes=100)
    env.close()
