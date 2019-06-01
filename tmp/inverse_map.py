#!/usr/bin/env python3
"""
Shows how to toss a capsule to a container.
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt
import pickle

#loading data
print("loading data")
kinematics=np.load("babbling_kinematics.npy")
actuations=np.load("babbling_actuations.npy")
number_of_samples=actuations.shape[0]

train_ratio=1 # from 0 to 1, 0 being all test and 1 being all train
kinematics_train=kinematics[:int(np.round(train_ratio*number_of_samples)),:]
kinematics_test=kinematics[int(np.round(train_ratio*number_of_samples))+1:,:]
actuations_train=actuations[:int(np.round(train_ratio*number_of_samples)),:]
actuations_test=actuations[int(np.round(train_ratio*number_of_samples))+1:,:]
number_of_samples_test=actuations_test.shape[0]

#training the model
print("training the model")
mlp = MLPRegressor(hidden_layer_sizes=(15), activation = "logistic")
mlp.fit(kinematics_train, actuations_train)
pickle.dump(mlp,open("mlp_model.sav", 'wb'))

# running the model
print("running the model")
#mlp=pickle.load(open("mlp_model.sav", 'rb')) # loading the model
est_actuations=mlp.predict(kinematics)

# plotting the results
plt.figure()
plt.subplot(311)
plt.plot(range(0, actuations.shape[0]), actuations[:,0], range(0, actuations.shape[0]), est_actuations[:,0])

plt.subplot(312)
plt.plot(range(0, actuations.shape[0]), actuations[:,1], range(0, actuations.shape[0]), est_actuations[:,1])

plt.subplot(313)
plt.plot(range(0, actuations.shape[0]), actuations[:,2], range(0, actuations.shape[0]), est_actuations[:,2])
plt.show(block=True)
#import pdb; pdb.set_trace()
