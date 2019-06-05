
# next is to add accel and see the difference
# add stiffness too
import numpy as np
from matplotlib import pyplot as plt
from all_functions import *




[babbling_kinematics, babbling_activations] = babbling_fcn(simulation_minutes=5)
model = inverse_mapping_fcn(kinematics=babbling_kinematics, activations=babbling_activations)


[model, errors] = in_air_adaptation_fcn(model=model, babbling_kinematics=babbling_kinematics, babbling_activations=babbling_activations, number_of_refinements=10)
# mini test code
new_features = gen_features_fcn(100, prev_features=np.ones(10,))
print(new_features)
[q0_filtered, q1_filtered] = feat_to_positions_fcn(new_features)
task_kinematics = positions_to_kinematics(q0_filtered, q1_filtered, timestep = 0.005)
est_task_activations = estimate_activations_fcn(model=model, desired_kinematics=task_kinematics)
[real_task_kinematics, real_task_activations] = run_task_fcn(task_kinematics, est_task_activations, Mj_render=True)

input("End of the simulation; press anykey to exit")
#import pdb; pdb.set_trace()
