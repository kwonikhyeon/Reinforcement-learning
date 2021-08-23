import gym

env = gym.make('CartPole-v0')
observation = env.reset()
action = env.action_space.sample()
step = env.step(action)

print('First observation:', observation) # [카트의 위치, 카트의 속도, 막대기의 각도, 막대기의 회전율]
print('Action:', action) # 0 or 1
print('Step:', step) # observation, reward, done, info