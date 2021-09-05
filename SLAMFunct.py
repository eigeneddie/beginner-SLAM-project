from controller import Robot
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import copy
from HardwareInterfaceFunct import *
from Visualization import *

#Important Global Definitions of Robot
timestep = 64
landmark_amount = 8
wheel_radius = 0.025
wheel_circum = 2*3.14*wheel_radius
radian_lidar_unit = 2*math.pi/256
degree_lidar_unit = 360/256    
#encoder unit
encoder_unit = wheel_circum/6.28
distance_between_wheels = 0.090
max_speed = 6.28 #Angular velocity

#plot preferences, interactive plotting mode
plt.axis([-1, 2, -1, 2])
plt.ion()
plt.show()

# Function to read landmarks data
# file : file name that has a list of landmark that has three attributes : id of landmark, xPos, yPos
# splitter : the splitter between attribute of landmark
# return : the landmarks dictionary (hash table)
def read_landmarks(file,splitter): 
    #Opening the Raw Data of the landmarks list
    rawDat = open(file)
    #Preparing a variable as a dictionary, which is a hash table
    landmarks = dict()
    #Looping for all row at the rawDat
    for row in rawDat :
        #Pick each row
        line_full = row.split("\n")
        #Split to each attribute
        attributes = line_full[0].split(splitter)
        #Include it in a dictionary as hash table.
        #The key is the landmark id
        #The value is the x_landmark position and y_landmark position
        landmarks[int(attributes[0])] = [float(attributes[1]), float(attributes[2])]
    return landmarks
    
def particleInitialization(numberOfParticles, totalLandmarks): 
    # Structuring a particle and the landmarks:
    #   j-th property = robot pose + M landmarks

    # Make N particles of these:
    # 1. State
    #    properties: x, y, theta, weight, & path (snail trail).     
    # 2. Landmark
    #    properties: mu, std, observed?

    allParticles = [] # List of all particles (State + Landmark)

    for j in range(numberOfParticles):
        
        # Properties for the j-th particle (1 particle = robot pose + M landmarks)
        stateParticle = dict()
        
        stateParticle['x'] = 0 # x, y, theta begins at origin
        stateParticle['y'] = 0
        stateParticle['theta'] = 0
        stateParticle['weight'] = 1.0/float(numberOfParticles) # every particle weight the same and sums up to one
        stateParticle['history'] = [] # to plot trail of EACH particles
        allLandmarks = dict()

        for i in range(totalLandmarks):
            # Property for the i-th landmark
            landmarkProperty = dict()
            landmarkProperty['mu'] = [0,0] # we don't know where it is in x,y
            landmarkProperty['sigma'] = np.zeros([2,2]) # covariance matrix of a property
            landmarkProperty['observed'] = False # assume we haven't seen the landmark yet
            
            # i-th landmark for the j-th particle estimate
            allLandmarks[i+1] = landmarkProperty # so that we label the landmark from 1 (not 0)
        
        stateParticle['landmarks'] = allLandmarks # assign M landmark properties as a state particle

        allParticles.append(stateParticle) # we got the j-the particle

    return allParticles # return N amount of robot and all landmark estimates

def sample_motion_model(odometry, particles):
    # Updates the particle positions, based on old positions, the odometry
    # measurements and the motion noise 
    rot1 = odometry['r1']
    trans = odometry['t']
    rot2 = odometry['r2']

    alpha = [0, 0, 0, 0]

    cov_rot1 = alpha[0]*(abs(rot1)) + alpha[1]*trans
    cov_trans = alpha[2]*trans + alpha[3]*(abs(rot1) + abs(rot2))
    cov_rot2 = alpha[0]*(abs(rot2)) + alpha[1]*trans
    #update particle 
    for particle in particles :
        noisy_rot1 = rot1 + np.random.normal(0, cov_rot1)
        noisy_trans = trans + np.random.normal(0,cov_trans)
        noisy_rot2 = rot2 + np.random.normal(0,cov_rot2)
        #For Visualization Purpose   
        particle['history'].append([particle['x'],[particle['y']]])
        #Update the Particles
        particle['x'] = particle['x'] + noisy_trans * np.cos(particle['theta'] + noisy_rot1)
        particle['y'] = particle['y'] + noisy_trans * np.sin(particle['theta'] + noisy_rot1)
        particle['theta'] = particle['theta'] + noisy_rot1 + noisy_rot2
        #To limiting it between -Pi and Pi
        if(particle['theta'] > math.pi):
            particle['theta'] = particle['theta'] - 2*math.pi
        elif(particle['theta'] < -1*math.pi):
            particle['theta'] = particle['theta'] + 2*math.pi
    return



def measurement_model(particle, landmark):
    #Compute the expected measurement for a landmark
    #and the Jacobian with respect to the landmark.

    px = particle['x']
    py = particle['y']
    ptheta = particle['theta']

    lx = landmark['mu'][0]
    ly = landmark['mu'][1]

    #calculate expected range measurement
    meas_range_exp = np.sqrt( (lx - px)**2 + (ly - py)**2 )
    meas_bearing_exp = math.atan2(ly - py, lx - px) - ptheta

    h = np.array([meas_range_exp, meas_bearing_exp])

    # Compute the Jacobian H of the measurement function h 
    #wrt the landmark location
    
    H = np.zeros((2,2))
    H[0,0] = (lx - px) / h[0]
    H[0,1] = (ly - py) / h[0]
    H[1,0] = (py - ly) / (h[0]**2)
    H[1,1] = (lx - px) / (h[0]**2)

    return h, H

def eval_sensor_model(sensor_data, particles):
    #Correct landmark poses with a measurement and
    #calculate particle weight

    #sensor noise
    Q_t = np.array([[0.1, 0],\
                    [0, 0.1]])

    #measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']
    bearings = sensor_data['bearing']

    #update landmarks and calculate weight for each particle
    for particle in particles:

        landmarks = particle['landmarks']

        px = particle['x']
        py = particle['y']
        ptheta = particle['theta'] 

        #loop over observed landmarks 
        for i in range(len(ids)):

            #current landmark
            lm_id = ids[i]
            landmark = landmarks[lm_id]
            
            #measured range and bearing to current landmark
            meas_range = ranges[i]
            meas_bearing = bearings[i]

            if not landmark['observed']:
                # landmark is observed for the first time
                
                # initialize landmark mean and covariance. You can use the
                # provided function 'measurement_model' above
                lx = px + meas_range * np.cos(ptheta + meas_bearing)
                ly = py + meas_range * np.sin(ptheta + meas_bearing)
                landmark['mu'] = [lx, ly]
                
                #get expected measurement and Jacobian wrt. landmark position
                h, H = measurement_model(particle, landmark)
                #initialize covariance for this landmark
                H_inv = np.linalg.inv(H)
                landmark['sigma'] = H_inv.dot(Q_t).dot(H_inv.T)

                landmark['observed'] = True

            else:
                # landmark was observed before

                # update landmark mean and covariance. You can use the
                # provided function 'measurement_model' above. 
                # calculate particle weight: particle['weight'] = ...
                h, H = measurement_model(particle, landmark)
                #Calculate measurement covariance and Kalman gain
                S = landmark['sigma']
                Q = H.dot(S).dot(H.T) + Q_t
                K = S.dot(H.T).dot(np.linalg.inv(Q))

                #Compute the difference between the observed and the expected measurement
                delta = np.array([meas_range - h[0], angle_diff(meas_bearing,h[1])])
                #update estimated landmark position and covariance
                landmark['mu'] = landmark['mu'] + K.dot(delta)
                landmark['sigma'] = (np.identity(2) - K.dot(H)).dot(S)

                # compute the likelihood of this observation
                fact = 1 / np.sqrt(math.pow(2*math.pi,2) * np.linalg.det(Q))
                expo = -0.5 * np.dot(delta.T, np.linalg.inv(Q)).dot(delta)
                weight = fact * np.exp(expo)
                particle['weight'] = particle['weight'] * weight


    #normalize weights
    normalizer = sum([p['weight'] for p in particles])

    for particle in particles:
        particle['weight'] = particle['weight'] / normalizer
    
    return

def resamplingProcedure(particles):
    # Russian roullete scheme. One round trip of the roullete accumulates to probability of 1.
    # So, we set the initial step of the pointer in the russian roullete as follows:

    # Define new particle sample space
    newParticles = []

    # Initial interval within the first particle's weight
    step = 1/float(len(particles))
    
    # random pointer withing the first weight interval
    u = np.random.uniform(0, step)

    # access the current weight within the pointer's range
    c = particles[0]['weight']

    # increment
    i = 0

    for particle in particles:

        # when the pointer exceeds the value of the current weight, switch particles weight range
        while u > c:
            i = i + 1
            c = c + particles[i]['weight']

        # add the particle to the new set of particles by copying it.
        newParticle = copy.deepcopy(particles[i])
        newParticle['weight'] = 1.0/len(particles)
        newParticles.append(newParticle)

        #increase the threshold
        u = u + step
    # hint: To copy a particle from particles to the new_particles
    # list, first make a copy:
    # new_particle = copy.deepcopy(particles[i])
    # ...
    # new_particles.append(new_particle)
    return newParticles