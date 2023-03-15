import numpy as np
from gym import spaces
from gym.core import Env

class Dummy(Env):
    def __init__(self):
        self.action_space = spaces.Box(low=-0.4, high=0.4, shape=(1,))
        self.observation_space = spaces.Box(low=-200, high=200, shape=(2,))
        self.M = 1 # car mass [kg]
        self.b = 0.001    # friction coef [N/m/s]
        self.Dt = 0.10 # timestep [s]

        self.A = np.array([[0, self.Dt],
                          [0, self.b*self.Dt/self.M]])

        self.B = np.array([0,self.Dt/self.M]).reshape((2,1))

        self.initial_state = np.array([-10.0, 2.0]).reshape((2,1))

    def step(self, action):
        self.state += self.A @ self.state + self.B * action
                      #0.1 * np.random.normal(scale=[[1e-3], [1e-3], [1e-3], [0.001]], size=(4,1))

        reward = -(self.state[0]**2 + self.state[1]**2*0.01)
        return np.reshape(self.state[:], (2,)), reward, False, None

    def reset(self):
        self.state = self.initial_state + 0.3 * np.random.normal(size=(2,1))
        return np.reshape(self.state[:], (2,))
