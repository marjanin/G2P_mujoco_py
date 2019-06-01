from mujoco_py import load_model_from_path, MjSim, MjViewer
import numpy as np
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot as plt
#import pickle
import os


def babbling_fcn():
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

	simulation_time=5*60.0
	timestep=0.005
	babble_phase_time=3

	run_samples=int(np.round(simulation_time/timestep))
	babble_phase_samples=int(np.round(babble_phase_time/timestep))
	babbling_kinematics=np.zeros((run_samples,4))
	babbling_actuations=np.zeros((run_samples,3))

	#while True:
	sim.set_state(sim_state)
	for ii in range(run_samples):
	    current_kinematics_array=np.array([sim.data.qpos[0], sim.data.qvel[0], sim.data.qpos[1], sim.data.qvel[1]])
	    if (ii%babble_phase_samples)==0:
	        sim.data.ctrl[:] = np.random.uniform(0,1,control_vector_length)
	    sim.step()
	    babbling_kinematics[ii,:]=current_kinematics_array
	    babbling_actuations[ii,:]=sim.data.ctrl
	    #viewer.render()
	print("min and max joint 0, min and max joint 1:")
	print(np.min(babbling_kinematics[:,0]), np.max(babbling_kinematics[:,0]), np.min(babbling_kinematics[:,2]), np.max(babbling_kinematics[:,2]))
	return babbling_kinematics, babbling_actuations
	#np.save("babbling_kinematics",babbling_kinematics)
	#np.save("babbling_actuations",babbling_actuations)

def inverse_mapping_fcn(babbling_kinematics, babbling_actuations):
	"""
	this function used the babbling data to create an inverse mapping using a
	MLP NN
	"""
	number_of_samples=babbling_actuations.shape[0]

	train_ratio=1 # from 0 to 1, 0 being all test and 1 being all train
	kinematics_train=babbling_kinematics[:int(np.round(train_ratio*number_of_samples)),:]
	kinematics_test=babbling_kinematics[int(np.round(train_ratio*number_of_samples))+1:,:]
	actuations_train=babbling_actuations[:int(np.round(train_ratio*number_of_samples)),:]
	actuations_test=babbling_actuations[int(np.round(train_ratio*number_of_samples))+1:,:]
	number_of_samples_test=actuations_test.shape[0]

	#training the model
	print("training the model")
	mlp = MLPRegressor(hidden_layer_sizes=(15), activation = "logistic")
	mlp.fit(kinematics_train, actuations_train)
	#pickle.dump(mlp,open("mlp_model.sav", 'wb'))

	# running the model
	print("running the model")
	#mlp=pickle.load(open("mlp_model.sav", 'rb')) # loading the model
	est_actuations=mlp.predict(babbling_kinematics)

	# plotting the results
	plt.figure()
	plt.subplot(311)
	plt.plot(range(babbling_actuations.shape[0]), babbling_actuations[:,0], range(babbling_actuations.shape[0]), est_actuations[:,0])

	plt.subplot(312)
	plt.plot(range(babbling_actuations.shape[0]), babbling_actuations[:,1], range(babbling_actuations.shape[0]), est_actuations[:,1])

	plt.subplot(313)
	plt.plot(range(babbling_actuations.shape[0]), babbling_actuations[:,2], range(babbling_actuations.shape[0]), est_actuations[:,2])
	plt.show(block=False)
	return mlp
	#import pdb; pdb.set_trace()



def create_task_kinematics_fcn(task_length, number_of_cycles, mlp):
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
	task_kinematics=np.transpose(np.concatenate(([[q0], [np.gradient(q0)/timestep], [q1], [np.gradient(q1)/timestep]]),axis=0))


	# running the model
	print("running the model")
	#mlp=pickle.load(open("mlp_model.sav", 'rb')) # loading the model
	est_task_actuations=mlp.predict(task_kinematics)

	# plotting the results
	plt.figure()
	plt.subplot(311)
	plt.plot(range(task_kinematics.shape[0]), est_task_actuations[:,0])

	plt.subplot(312)
	plt.plot(range(task_kinematics.shape[0]), est_task_actuations[:,1])

	plt.subplot(313)
	plt.plot(range(task_kinematics.shape[0]), est_task_actuations[:,2])
	plt.show(block=False)
	#np.save("task_kinematics",task_kinematics)
	#np.save("est_task_actuations",est_task_actuations)
	#import pdb; pdb.set_trace()
	return task_kinematics, est_task_actuations


def run_task_fcn(task_kinematics, est_task_actuations):
	"""
	this function runs the predicted actuations generatred from running
	the inverse map on the desired task kinematics
	"""

	#loading data
	#print("loading data")
	#task_kinematics=np.load("task_kinematics.npy")
	#est_task_actuations=np.load("est_task_actuations.npy")


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

	#np.save("real_task_kinematics",real_task_kinematics)
	#np.save("real_task_actuations",real_task_actuations)
	plt.figure()
	plt.subplot(411)
	plt.plot(range(number_of_task_samples), task_kinematics[:,0], range(number_of_task_samples), real_task_kinematics[:,0])
	plt.subplot(412)
	plt.plot(range(number_of_task_samples), task_kinematics[:,1], range(number_of_task_samples), real_task_kinematics[:,1])
	plt.subplot(413)
	plt.plot(range(number_of_task_samples), task_kinematics[:,2], range(number_of_task_samples), real_task_kinematics[:,2])
	plt.subplot(414)
	plt.plot(range(number_of_task_samples), task_kinematics[:,3], range(number_of_task_samples), real_task_kinematics[:,3])
	plt.show(block=True)
	return real_task_kinematics, real_task_actuations
	 #   if os.getenv('TESTING') is not None:
 #       break
