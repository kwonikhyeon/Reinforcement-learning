import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

# 4개의 입력, 2개의 출력, 2개의 히든 레이어(각각 24개 뉴런)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(24, input_dim=4, activation=tf.nn.relu),
    tf.keras.layers.Dense(24, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation='linear')
])

# 모델 훈련방식 = adam, 손실함수 = mean_squared_error, 학습률 = 0.001

model.compile(optimizer='adam',
            loss='mean_squared_error')

score = []
memory = deque(maxlen=2000)
env = gym.make('CartPole-v0')

for i in range(1000):
    state = env.reset()
    state = np.reshape(state, [1, 4])
    eps = 1/(i/50+10) # 엡실론 값이 0.1에서 0.03까지 에피소드가 반복될수록 감소
    
    for t in range(200):
        env.render()
        if np.random.rand() < eps:
            action = np.random.randint(0, 2)
        else:
            predict = model.predict(state)
            action = np.argmax(predict)

        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])

        memory.append((state, action, reward, next_state, done))
        state = next_state

        if done or t == 199:
            print('Episode', i, 'Score', t+1)
            score.append(t+1)
            break

    if i > 10:
        minibatch = random.sample(memory, 16)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + 0.9 * np.amax(model.predict(next_state)[0])
            target_outputs = model.predict(state)
            target_outputs[0][action] = target
            model.fit(state, target_outputs, epochs=1, verbose=0)

env.close()
print(score)
