import pandas
import gym


def RollingMean(df):
    return df.rolling(12, 1, center=True).mean()

def RollingStd(df):
    return df.rolling(12, 1, center=True).std()


if __name__ == "__main__":
    env = gym.make("Pong-v0")
    obs = env.reset()
    pan = pandas.Panel(obs)
    df = pan.swapaxes(0, 2).to_frame()
    rm = RollingMean(df)
    rstd = RollingStd(df)