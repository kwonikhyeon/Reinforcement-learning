import gym
from gym.wrappers import Monitor

env = gym.make('CartPole-v0')
env.reset()

for _ in range(10):
    env.render()
    observation, reward, done, info = env.step(0)
    print(observation, done)
    if done:
        break
env.close()