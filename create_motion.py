#!/usr/bin/env python3
"""
Shows how to toss a capsule to a container.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt
import pickle


timestep=0.005
task_length=5 # in seconds
number_of_task_samples=int(np.round(task_length/timestep))

q0=np.zeros(number_of_task_samples)
q1=np.zeros(number_of_task_samples)

for ii in range(number_of_task_samples):
	q0[ii]=np.sin(2*(2*np.pi*ii/number_of_task_samples))
	q1[ii]=np.cos(2*(2*np.pi*ii/number_of_task_samples))
#import pdb; pdb.set_trace()
task_kinematics=np.transpose(np.concatenate(([[q0], [np.gradient(q0)], [q1], [np.gradient(q1)]]),axis=0))


# running the model
print("running the model")
mlp=pickle.load(open("mlp_model.sav", 'rb')) # loading the model
est_task_actuations=mlp.predict(task_kinematics)

# plotting the results
plt.figure()
plt.subplot(311)
plt.plot(range(task_kinematics.shape[0]), est_task_actuations[:,0])

plt.subplot(312)
plt.plot(range(task_kinematics.shape[0]), est_task_actuations[:,1])

plt.subplot(313)
plt.plot(range(task_kinematics.shape[0]), est_task_actuations[:,2])
plt.show(block=True)
#import pdb; pdb.set_trace()
np.save("task_kinematics",task_kinematics)
np.save("est_task_actuations",est_task_actuations)