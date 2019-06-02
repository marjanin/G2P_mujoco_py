from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt
#import pickle
import os


def babbling_fcn(simulation_minutes = 5 ):
	"""
	this function babbles in the mujoco environment and then
	returns input outputs (actuation values and kinematics)
	"""
	np.random.seed(0) # to get consistent results for debugging purposes

	model = load_model_from_path("C:/Users/Ali/Google Drive/Current/USC/Github/mujoco-py/xmls/nmi_leg.xml")
	sim = MjSim(model)

	#viewer = MjViewer(sim)
	sim_state = sim.get_state()

	control_vector_length=sim.data.ctrl.__len__()
	print("control_vector_length: "+str(control_vector_length))
	simulation_minutes
	simulation_time=simulation_minutes*60.0
	timestep=0.005
	babble_phase_time=3

	run_samples=int(np.round(simulation_time/timestep))
	babble_phase_samples=int(np.round(babble_phase_time/timestep))
	babbling_kinematics=np.zeros((run_samples,6))
	babbling_activations=np.zeros((run_samples,3))

	#while True:
	sim.set_state(sim_state)
	for ii in range(run_samples):
	    current_kinematics_array=np.array(
	    	[sim.data.qpos[0],
	    	sim.data.qvel[0],
	    	0, sim.data.qpos[1],
	    	sim.data.qvel[1],
	    	0])
	    if (ii%babble_phase_samples)==0:
	        sim.data.ctrl[:] = np.random.uniform(0,1,control_vector_length)
	    sim.step()
	    babbling_kinematics[ii,:]=current_kinematics_array
	    babbling_activations[ii,:]=sim.data.ctrl
	    #viewer.render()
    # adding acceleration
	babbling_kinematics = np.transpose(
		np.concatenate(
			(
				[babbling_kinematics[:,0]],
				[babbling_kinematics[:,1]],
				[np.gradient(babbling_kinematics[:,1])/timestep],
				[babbling_kinematics[:,3]],
				[babbling_kinematics[:,4]],
				[np.gradient(babbling_kinematics[:,4])/timestep]),
			axis=0)
		)
	print("min and max joint 0, min and max joint 1:")
	print(
		np.min(babbling_kinematics[:,0]),
		np.max(babbling_kinematics[:,0]),
		np.min(babbling_kinematics[:,3]),
		np.max(babbling_kinematics[:,3]))
	return babbling_kinematics, babbling_activations
	#np.save("babbling_kinematics",babbling_kinematics)
	#np.save("babbling_activations",babbling_activations)

def inverse_mapping_fcn(kinematics, activations, **kwargs):
	"""
	this function used the babbling data to create an inverse mapping using a
	MLP NN
	"""
	number_of_samples=activations.shape[0]
	train_ratio=1 # from 0 to 1, 0 being all test and 1 being all train
	kinematics_train=kinematics[:int(np.round(train_ratio*number_of_samples)),:]
	kinematics_test=kinematics[int(np.round(train_ratio*number_of_samples))+1:,:]
	activations_train=activations[:int(np.round(train_ratio*number_of_samples)),:]
	activations_test=activations[int(np.round(train_ratio*number_of_samples))+1:,:]
	number_of_samples_test=activations_test.shape[0]

	#training the model
	print("training the model")
	if kwargs=={}:
		model = MLPRegressor(hidden_layer_sizes=(15), activation = "logistic",  verbose = True, warm_start = True)
	else:
		model=kwargs["prior_model"]

	model.fit(kinematics_train, activations_train)
	#pickle.dump(model,open("mlp_model.sav", 'wb'))

	# running the model
	print("running the model")
	#model=pickle.load(open("mlp_model.sav", 'rb')) # loading the model
	est_activations=model.predict(kinematics)

	# plotting the results
	plt.figure()
	plt.subplot(311)
	plt.plot(range(activations.shape[0]), activations[:,0], range(activations.shape[0]), est_activations[:,0])

	plt.subplot(312)
	plt.plot(range(activations.shape[0]), activations[:,1], range(activations.shape[0]), est_activations[:,1])

	plt.subplot(313)
	plt.plot(range(activations.shape[0]), activations[:,2], range(activations.shape[0]), est_activations[:,2])
	plt.show(block=False)
	return model
	#import pdb; pdb.set_trace()



def create_task_kinematics_fcn(task_length = 10 , number_of_cycles = 7):
	"""
	this function creates desired task kinematics and their corresponding 
	actuation values predicted using the inverse mapping
	"""
	timestep=0.005
	#task_length=5 # in seconds
	number_of_task_samples=int(np.round(task_length/timestep))

	q0=np.zeros(number_of_task_samples)
	q1=np.zeros(number_of_task_samples)

	for ii in range(number_of_task_samples):
		q0[ii]=(np.pi/3)*np.sin(number_of_cycles*(2*np.pi*ii/number_of_task_samples))
		q1[ii]=-1*(np.pi/2)*((-1*np.cos(number_of_cycles*(2*np.pi*ii/number_of_task_samples))+1)/2)
	#import pdb; pdb.set_trace()
	task_kinematics=np.transpose(
		np.concatenate(
			(
				[[q0],
				[np.gradient(q0)/timestep],
				[np.gradient(np.gradient(q0)/timestep)/timestep],
				[q1],
				[np.gradient(q1)/timestep],
				[np.gradient(np.gradient(q1)/timestep)/timestep]]),
			axis=0)
		)
	#np.save("task_kinematics",task_kinematics)
	#np.save("est_task_activations",est_task_activations)
	#import pdb; pdb.set_trace()
	return task_kinematics

def estimate_activations_fcn(model, desired_kinematics):
# running the model
	print("running the model")
	#model=pickle.load(open("mlp_model.sav", 'rb')) # loading the model
	est_activations=model.predict(desired_kinematics)
	# plotting the results
	plt.figure()
	plt.subplot(311)
	plt.plot(range(desired_kinematics.shape[0]), est_activations[:,0])

	plt.subplot(312)
	plt.plot(range(desired_kinematics.shape[0]), est_activations[:,1])

	plt.subplot(313)
	plt.plot(range(desired_kinematics.shape[0]), est_activations[:,2])
	plt.show(block=False)
	return est_activations

def run_task_fcn(task_kinematics, est_task_activations):
	"""
	this function runs the predicted activations generatred from running
	the inverse map on the desired task kinematics
	"""

	#loading data
	#print("loading data")
	#task_kinematics=np.load("task_kinematics.npy")
	#est_task_activations=np.load("est_task_activations.npy")


	model = load_model_from_path("C:/Users/Ali/Google Drive/Current/USC/Github/mujoco-py/xmls/nmi_leg.xml")
	sim = MjSim(model)

	#viewer = MjViewer(sim)
	sim_state = sim.get_state()

	control_vector_length=sim.data.ctrl.__len__()
	print("control_vector_length "+str(control_vector_length))

	timestep=0.005
	number_of_task_samples=task_kinematics.shape[0]

	real_task_kinematics=np.zeros((number_of_task_samples,6))
	real_task_activations=np.zeros((number_of_task_samples,3))

	#while True:
	sim.set_state(sim_state)
	for ii in range(number_of_task_samples):
	    sim.data.ctrl[:]=est_task_activations[ii,:]
	    sim.step()
	    current_kinematics_array=np.array(
	    	[sim.data.qpos[0],
	    	sim.data.qvel[0],
	    	sim.data.qacc[0],
	    	sim.data.qpos[1], sim.data.qvel[1],
	    	sim.data.qacc[1]])
	    real_task_kinematics[ii,:]=current_kinematics_array
	    real_task_activations[ii,:]=sim.data.ctrl
	    #viewer.render()

	#np.save("real_task_kinematics",real_task_kinematics)
	#np.save("real_task_activations",real_task_activations)
	plt.figure()
	plt.subplot(611)
	plt.plot(range(number_of_task_samples), task_kinematics[:,0], range(number_of_task_samples), real_task_kinematics[:,0])
	plt.subplot(612)
	plt.plot(range(number_of_task_samples), task_kinematics[:,1], range(number_of_task_samples), real_task_kinematics[:,1])
	plt.subplot(613)
	plt.plot(range(number_of_task_samples), task_kinematics[:,2], range(number_of_task_samples), real_task_kinematics[:,2])
	plt.subplot(614)
	plt.plot(range(number_of_task_samples), task_kinematics[:,3], range(number_of_task_samples), real_task_kinematics[:,3])
	plt.subplot(615)
	plt.plot(range(number_of_task_samples), task_kinematics[:,4], range(number_of_task_samples), real_task_kinematics[:,4])
	plt.subplot(616)
	plt.plot(range(number_of_task_samples), task_kinematics[:,5], range(number_of_task_samples), real_task_kinematics[:,5])
	plt.show(block=True)
	return real_task_kinematics, real_task_activations
	 #   if os.getenv('TESTING') is not None:
 #       break
#import pdb; pdb.set_trace()
	