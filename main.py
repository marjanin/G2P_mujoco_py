
# next is to add accel and see the difference
# add stiffness too

from all_functions import babbling_fcn, inverse_mapping_fcn, create_task_kinematics_fcn, run_task_fcn

[babbling_kinematics, babbling_actuations] = babbling_fcn(5)
mlp = inverse_mapping_fcn(babbling_kinematics, babbling_actuations)
[task_kinematics, est_task_actuations] = create_task_kinematics_fcn(10, 7, mlp)
[real_task_kinematics, real_task_actuations] = run_task_fcn(task_kinematics, est_task_actuations)

input("press anykey to exit")