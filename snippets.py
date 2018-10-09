#/bin/python
"""This is where sub parts of code will be tested"""


import pandas
import gym


env = gym.make("Pong-v0")
obs = env.reset()
pan = pandas.Panel(obs)
df = pan.swapaxes(0, 2).to_frame()
rm = df.rolling(12, 1, center=True).mean()
rst = df.rolling(12, 1, center=True).std()
print(rm)
print(rst)

