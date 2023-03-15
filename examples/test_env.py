import numpy as np
import gym
from pilco.models import PILCO
from pilco.controllers import RbfController, LinearController
from pilco.rewards import ExponentialReward
from pilco.rewards import SquareReward
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from gpflow import set_trainable
# from tensorflow import logging
np.random.seed(0)

from dummy_env import Dummy

from utils import rollout, policy

def plotIt(X,Y):
    plt.clf()
    plt.subplot(1,2,1)
    plt.plot(X[:,0:2])
    plt.plot(Y[:,0:2])
    plt.subplot(1,2,2)
    plt.plot(X[:,0],Y[:,0])
    plt.draw() 
    plt.pause(0.001)
    plt.show()

env = Dummy()
# Initial random rollouts to generate a dataset
X,Y, _, _ = rollout(env=env, pilco=None, random=True, timesteps=100, render=False)
for i in range(1,5):
    X_, Y_, _, _ = rollout(env=env, pilco=None, random=True,  timesteps=100, render=False)
    X = np.vstack((X, X_))
    Y = np.vstack((Y, Y_))


state_dim = Y.shape[1]
control_dim = X.shape[1] - state_dim
#controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=10)
controller = LinearController(state_dim=state_dim, control_dim=control_dim)
print(state_dim)
print(control_dim)
#pilco = PILCO((X, Y), controller=controller, horizon=100)
# Example of user provided reward function, setting a custom target state
#R = ExponentialReward(state_dim=state_dim, t=np.array([0,0]),W=np.diag([1,0]))

# Kanske hittar lokalt minima då låg hastighet rewardas
R = SquareReward(state_dim=state_dim, W=np.diag([1,0.01]))
pilco = PILCO((X, Y), controller=controller, horizon=100, reward=R)
plt.ion()
plt.show(block=False)
print(tf.executing_eagerly())
for rollouts in range(10):
    print("New Iteration")
    pilco.optimize_models(restarts=1)
    pilco.optimize_policy()
    import pdb; #pdb.set_trace()
    X_new, Y_new, _, _ = rollout(env=env, pilco=pilco, timesteps=300, render=False, verbose=False)
    # Update dataset
    plotIt(X_new,Y_new)
    X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
    pilco.mgpr.set_data((X , Y))
    
plt.pause(0.001)
input("Test")

