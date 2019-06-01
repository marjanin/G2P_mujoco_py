#!/usr/bin/env python3
"""
Shows how to toss a capsule to a container.
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import os


#loading data
print("loading data")
task_kinematics=np.load("task_kinematics.npy")
est_task_actuations=np.load("est_task_actuations.npy")

model = load_model_from_path("C:/Users/Ali/Google Drive/Current/USC/Github/mujoco-py/xmls/nmi_leg.xml")
sim = MjSim(model)

viewer = MjViewer(sim)
sim_state = sim.get_state()

control_vector_length=sim.data.ctrl.__len__()
print("control_vector_length "+str(control_vector_length))

simulation_time=5*60.0
timestep=0.005
number_of_task_samples=task_kinematics.shape[0]

real_task_kinematics=np.zeros((number_of_task_samples,4))
real_task_actuations=np.zeros((number_of_task_samples,3))

#while True:
sim.set_state(sim_state)
for ii in range(number_of_task_samples):
    sim.data.ctrl[:]=est_task_actuations[ii,:]
    sim.step()
    current_kinematics_array=np.array([sim.data.qpos[0], sim.data.qvel[0], sim.data.qpos[1], sim.data.qvel[1]])
    real_task_kinematics[ii,:]=current_kinematics_array
    real_task_actuations[ii,:]=sim.data.ctrl
    viewer.render()

np.save("real_task_kinematics",real_task_kinematics)
np.save("real_task_actuations",real_task_actuations)
 #   if os.getenv('TESTING') is not None:
 #       break
