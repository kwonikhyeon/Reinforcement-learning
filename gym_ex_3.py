import gym

env = gym.make('CartPole-v0')
print(env.action_space) # Discrete : 고정된 범위의 음이아닌 숫자를 허용하는 공간
print(env.observation_space) # Box : n차원의 박스

print(env.observation_space.high)
print(env.observation_space.low)