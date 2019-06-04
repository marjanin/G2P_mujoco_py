
# next is to add accel and see the difference
# add stiffness too
import numpy as np
from matplotlib import pyplot as plt
from all_functions import *


def gen_features_fcn(prev_reward, **kwargs):
	#import pdb; pdb.set_trace()
	reward_thresh = 10
	feat_min = 0.1
	feat_max = 0.9
	sigma=.00001 # should be inversly proportional to reward
	if ("prev_features" in kwargs):
		prev_features = kwargs["prev_features"]
	elif ("feat_vec_length" in kwargs):
		prev_features = np.random.uniform(feat_min,feat_max,kwargs["feat_vec_length"])
	else:
		raise NameError('Either prev_features or feat_vec_length needs to be provided')
	if prev_reward<reward_thresh:
		new_features = np.random.uniform(feat_min,feat_max,kwargs["feat_vec_length"])	
	else:
		new_features = np.zeros(prev_features.shape[0],)
		for ii in range(prev_features.shape[0]):
			new_features[ii] = np.random.normal(prev_features[ii],sigma)
		new_features = np.maximum(new_features, feat_min*np.ones(prev_features.shape[0],))
		new_features = np.minimum(new_features, feat_max*np.ones(prev_features.shape[0],))
	return new_features


def feat_to_kinematics_fcn(features):
	number_of_features = features.shape[0]
	feat_angles = np.linspace(0, 2*np.pi*(number_of_features/(number_of_features+1)), number_of_features)
	q0_raw = features*np.sin(feat_angles)
	q1_raw = features*np.cos(feat_angles)
	q0_scaled = (q0_raw*np.pi/3)
	q1_scaled = (q1_raw*np.pi/4)+(np.pi/4)

	plt.figure()
	plt.plot(q0_scaled, q1_scaled)
	plt.xlabel = "q0"
	plt.ylabel = "q1"
	plt.show(block=True)
	return q0_scaled, q1_scaled

#[babbling_kinematics, babbling_activations] = babbling_fcn(simulation_minutes=5)
#model = inverse_mapping_fcn(kinematics=babbling_kinematics, activations=babbling_activations)


#[model, errors] = in_air_adaptation_fcn(model=model, babbling_kinematics=babbling_kinematics, babbling_activations=babbling_activations, number_of_refinements=10)
#gen_features_fcn(prev_reward = 10, feat_vec_length = 15)

# mini test code
new_features = gen_features_fcn(100, prev_features=np.ones(10,))
print(new_features)
[q0_scaled, q1_scaled] = feat_to_kinematics_fcn(new_features)

#import pdb; pdb.set_trace()
input("press anykey to exit")
#import pdb; pdb.set_trace()
