import gym

env = gym.make('CartPole-v0')
observation = env.reset()

for i in range(100):
    env.render()
    if observation[2] > 0:
        action = 1
    else:
        action = 0

    observation, reward, done, info = env.step(action)
    print(observation, done)
    if done:
        print(i + 1)
        break
env.close()