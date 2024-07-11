import gym
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000

SHOW_EVERY = 2000

env = gym.make("MountainCar-v0")

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (
    env.observation_space.high - env.observation_space.low
) / DISCRETE_OS_SIZE

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(
    low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n])
)

ep_rewards = []
aggr_ep_reward = {"ep": [], "avg": [], "min": [], "max": []}


if not os.path.exists("qtables"):
    os.makedirs("qtables")


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(discrete_state.astype(int))


for episode in range(EPISODES):
    episode_reward = 0
    render = False
    if episode % SHOW_EVERY == 0:
        render = True
        print(f"Episode: {episode}")

    env = gym.make("MountainCar-v0", render_mode="human" if render else None)
    observation, info = env.reset()
    discrete_state = get_discrete_state(observation)

    done = False
    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, termination, truncation, _ = env.step(action)
        episode_reward += reward
        new_discrete_state = get_discrete_state(new_state)

        if termination or truncation:
            done = True
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (
                reward + DISCOUNT * max_future_q
            )
            q_table[discrete_state + (action,)] = new_q

        elif new_state[0] >= env.goal_position:
            print(f"Reach the goal on episode {episode}")
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)

    if not episode % 10 == 0:
        np.save(f"qtables/{episode}-qtable.npy", q_table)

    if not episode % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_reward["ep"].append(episode)
        aggr_ep_reward["avg"].append(average_reward)
        aggr_ep_reward["min"].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_reward["max"].append(max(ep_rewards[-SHOW_EVERY:]))

        print(
            f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])}, max: {max(ep_rewards[-SHOW_EVERY:])}"
        )

env.close()

plt.plot(aggr_ep_reward["ep"], aggr_ep_reward["avg"], label="avg")
plt.plot(aggr_ep_reward["ep"], aggr_ep_reward["min"], label="min")
plt.plot(aggr_ep_reward["ep"], aggr_ep_reward["max"], label="max")
plt.legend(loc=4)
plt.show()
