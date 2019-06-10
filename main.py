
# next is to add accel and see the difference
# add stiffness too
import numpy as np
from matplotlib import pyplot as plt
from all_functions import *
import pickle

# [babbling_kinematics, babbling_activations] = babbling_fcn(simulation_minutes=25)
# model = inverse_mapping_fcn(kinematics=babbling_kinematics, activations=babbling_activations)

# [model, errors, cum_kinematics, cum_activations] =\
# in_air_adaptation_fcn(
# 	model=model, babbling_kinematics=babbling_kinematics, babbling_activations=babbling_activations, number_of_refinements=10)

# pickle.dump([model,cum_kinematics, cum_activations],open("results/mlp_model.sav", 'wb'))
[model,cum_kinematics, cum_activations] = pickle.load(open("results/mlp_model.sav", 'rb')) # loading the model

[best_reward, all_rewards] = \
learn_to_move_fcn(
	model=model, cum_kinematics= cum_kinematics, cum_activations=cum_activations, refinement=True, Mj_render=True)

input("End of the simulation; press anykey to exit")
#import pdb; pdb.set_trace()
