import gym
import tensorflow as tf
import numpy as np

env = gym.make('CartPole-v0')

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, input_shape=(4,), activation=tf.nn.relu),
    tf.keras.layers.Dense(2)
])

observation = env.reset()

predict = model.predict(observation.reshape(1,4))
action = np.argmax(predict)

print(observation)
print(predict)
print(action)
