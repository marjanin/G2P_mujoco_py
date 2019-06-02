
# next is to add accel and see the difference
# add stiffness too
import numpy as np
from all_functions import babbling_fcn, inverse_mapping_fcn, create_task_kinematics_fcn, estimate_activations_fcn, run_task_fcn

[babbling_kinematics, babbling_activations] = babbling_fcn(simulation_minutes = 5)
model = inverse_mapping_fcn(kinematics= babbling_kinematics, activations= babbling_activations)
task_kinematics = create_task_kinematics_fcn(task_length = 10, number_of_cycles = 7)
est_task_activations = estimate_activations_fcn(model = model, desired_kinematics = task_kinematics)
[real_task_kinematics, real_task_activations] = run_task_fcn(task_kinematics, est_task_activations)

cum_kinematics = babbling_kinematics
cum_activations = babbling_activations
for ii in range(10):
	cum_kinematics=np.concatenate([cum_kinematics, real_task_kinematics])
	cum_activations=np.concatenate([cum_activations, real_task_activations])
	model = inverse_mapping_fcn(kinematics= cum_kinematics, activations= cum_activations, prior_model=model)
	est_task_activations = estimate_activations_fcn(model = model, desired_kinematics = task_kinematics)
	[real_task_kinematics, real_task_activations] = run_task_fcn(task_kinematics, est_task_activations)

#input("press anykey to exit")
#import pdb; pdb.set_trace()
