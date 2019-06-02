
# next is to add accel and see the difference
# add stiffness too
import numpy as np
from all_functions import babbling_fcn, inverse_mapping_fcn, create_task_kinematics_fcn, run_task_fcn

[babbling_kinematics, babbling_actuations] = babbling_fcn(simulation_minutes = 5)
mlp = inverse_mapping_fcn(kinematics= babbling_kinematics, actuations= babbling_actuations)
[task_kinematics, est_task_actuations] = create_task_kinematics_fcn(mlp, task_length = 10, number_of_cycles = 7)
[real_task_kinematics, real_task_actuations] = run_task_fcn(task_kinematics, est_task_actuations)
input("first run completed, press enter to continue")

cum_kinematics=np.concatenate([babbling_kinematics, real_task_kinematics])
cum_actuations=np.concatenate([babbling_actuations, real_task_actuations])
mlp = inverse_mapping_fcn(kinematics= cum_kinematics, actuations= cum_actuations, prior_model=mlp)


input("press anykey to exit")
#import pdb; pdb.set_trace()
