
# next is to add accel and see the difference
# add stiffness too
import numpy as np
from matplotlib import pyplot as plt
from all_functions import *
import pickle



# [babbling_kinematics, babbling_activations] = babbling_fcn(simulation_minutes=25)
# model = inverse_mapping_fcn(kinematics=babbling_kinematics, activations=babbling_activations)

# [model, errors] = in_air_adaptation_fcn(model=model, babbling_kinematics=babbling_kinematics, babbling_activations=babbling_activations, number_of_refinements=10)
# pickle.dump(model,open("results/mlp_model.sav", 'wb'))

# mini test code
model = pickle.load(open("results/mlp_model.sav", 'rb')) # loading the model



# new_features = gen_features_fcn(prev_reward = 1000, prev_features=np.ones(10,))
# [q0_filtered, q1_filtered] = feat_to_positions_fcn(new_features)
# task_kinematics = positions_to_kinematics_fcn(q0_filtered, q1_filtered, timestep = 0.005)
# kinematics_show(task_kinematics)
# run_kinematics = attempt_to_run_kinematics(attempt_kinematics = task_kinematics)
# est_task_activations = estimate_activations_fcn(model=model, desired_kinematics=run_kinematics)
# [real_task_kinematics, real_task_activations, chassis_pos] = run_task_fcn(est_task_activations, chassis_fix=False, Mj_render=True)

prev_reward = np.array([0])
all_rewards = prev_reward
new_features = gen_features_fcn(prev_reward=prev_reward, feat_vec_length=10)

while prev_reward < 8:
	new_features = gen_features_fcn(prev_reward=prev_reward, prev_features=new_features)# .9*np.ones([9,])#
	print(new_features)
	[q0_filtered, q1_filtered] = feat_to_positions_fcn(new_features, show=True)
	task_kinematics = positions_to_kinematics_fcn(q0_filtered, q1_filtered, timestep = 0.005)
	#kinematics_show(task_kinematics)
	run_kinematics = attempt_to_run_kinematics(attempt_kinematics=task_kinematics)
	est_task_activations = estimate_activations_fcn(model=model, desired_kinematics=run_kinematics)
	[real_task_kinematics, real_task_activations, chassis_pos]=run_task_fcn(est_task_activations, chassis_fix=False, Mj_render=False)
	
	# plt.figure()
	# plt.plot(range(chassis_pos.shape[0]),chassis_pos)
	# plt.show(block=True)
	#import pdb; pdb.set_trace()
	prev_reward = chassis_pos[-1]
	print("reward: ", prev_reward)
	print("Max reward so far: ", np.max(all_rewards))

	all_rewards = np.append(all_rewards, chassis_pos[-1])
[real_task_kinematics, real_task_activations, chassis_pos]=run_task_fcn(est_task_activations, chassis_fix=False, Mj_render=True)
print("reward: ", all_rewards)

input("End of the simulation; press anykey to exit")
#import pdb; pdb.set_trace()
