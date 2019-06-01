#!/usr/bin/env python3
"""
Shows how to toss a capsule to a container.
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
import os

np.random.seed(0)

model = load_model_from_path("C:/Users/Ali/Google Drive/Current/USC/Github/mujoco-py/xmls/nmi_leg.xml")
sim = MjSim(model)

viewer = MjViewer(sim)
sim_state = sim.get_state()

control_vector_length=sim.data.ctrl.__len__()
print("control_vector_length "+str(control_vector_length))

simulation_time=5*60.0
timestep=0.005
babble_phase_time=3

run_samples=int(np.round(simulation_time/timestep))
babble_phase_samples=int(np.round(babble_phase_time/timestep))
kinematics=np.zeros((run_samples,4))
actuations=np.zeros((run_samples,3))

#while True:
sim.set_state(sim_state)
for ii in range(run_samples):
    current_kinematics_array=np.array([sim.data.qpos[0], sim.data.qvel[0], sim.data.qpos[1], sim.data.qvel[1]])
    if (ii%babble_phase_samples)==0:
        sim.data.ctrl[:] = np.random.uniform(0,1,control_vector_length)
    sim.step()
    kinematics[ii,:]=current_kinematics_array
    actuations[ii,:]=sim.data.ctrl
    #viewer.render()

np.save("babbling_kinematics",kinematics)
np.save("babbling_actuations",actuations)
 #   if os.getenv('TESTING') is not None:
 #       break
