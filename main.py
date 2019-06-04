
# next is to add accel and see the difference
# add stiffness too
import numpy as np
from matplotlib import pyplot as plt
from all_functions import *

[babbling_kinematics, babbling_activations] = babbling_fcn(simulation_minutes = 5)
model = inverse_mapping_fcn(kinematics= babbling_kinematics, activations= babbling_activations)



#[model, errors]= in_air_adaptation_fcn(model = model, babbling_kinematics = babbling_kinematics, babbling_activations = babbling_activations, number_of_refinements = 10)


input("press anykey to exit")
#import pdb; pdb.set_trace()
