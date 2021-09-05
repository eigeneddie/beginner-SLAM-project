from controller import Robot
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import copy
from HardwareInterfaceFunct import *
from SLAMFunct import *

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

#Function to visualize the result of SLAM
#Param : List of Particles and List of Real Landmarks
#Return : the Plotting for Particles that has survived and the Map Plotting
def plot_particles(particles, landmarks):
    #Define the limit of the plotting
    #Defining some spaces for Text Information
    space_text = 2
    space_text2 = 3.5-2.5
    lim = [-1, 2, -1, 2]
    lim = [lim[0], lim[1]+space_text, lim[2], lim[3]]
    
    #List for the REAL Landmark Position
    lx = []
    ly = []
    #Looping for the real landmarks
    for j in range(len(landmarks)):
        lx = np.append(lx,landmarks[j+1][0])
        ly = np.append(ly,landmarks[j+1][1])

    #Make a dictionary to save the best particle (highest weight)
    estimated_pose = dict()    
    #List of Robot Particles Position
    px = []
    py = []
    pthet = []
    #Looping for each particles, save it to the list
    for particle in particles:
        px = np.append(px, particle['x'])
        py = np.append(py, particle['y'])
        pthet = np.append(pthet, particle['theta'])
    
    #Find the Best Particle
    best_p = SearchBest(particles)
    #Save it to dictionary
    estimated_pose['x'] = best_p['x']
    estimated_pose['y'] = best_p['y']
    estimated_pose['theta'] = best_p['theta']
    Information_best = "Best Position\nx = {:.2f}\ny = {:.2f}\ntheta = {:.2f} rad".format(estimated_pose['x'], estimated_pose['y'], estimated_pose['theta'])
    #List of Path listed in the Best Particle
    path = best_p['history']
    pathx = []
    pathy = []
    #Looping for the path
    for dot in path:
        pathx = np.append(pathx, dot[0])
        pathy = np.append(pathy, dot[1])    
    
    ####Plot the data
    plt.clf()
    #Plot the Particles
    plt.plot(px,py,'b.')
    #Plot the error of landmarks approximation and Planning the Information Text
    Information_Landmarks_Best = "Landmarks Approx Position\n"
    Information_Landmarks_Real = "Landmarks Real Position\n"
    for i in range(len(landmarks)):
        landmark = best_p['landmarks'][i+1]
        ellipse = error_ellipse(landmark['mu'], landmark['sigma'])
        plt.gca().add_artist(ellipse)
        #Information Planning
        Information_Landmarks_Best += "ID : {} X: {:.2f} y:{:.2f}\n".format((i+1), landmark['mu'][0], landmark['mu'][1])
        Information_Landmarks_Real += "ID : {} X: {:.2f} y:{:.2f}\n".format((i+1), landmarks[i+1][0], landmarks[i+1][1])
    #Plot the particle of Robot Pose with blue dot
    plt.plot(pathx,pathy, 'b-')
    #Plot the Real Landmark Position with square
    plt.plot(lx,ly,'gx', markersize = 12)
    #Plot the best Particle Robot Pose with Quiver
    plt.quiver(estimated_pose['x'], estimated_pose['y'], math.cos(estimated_pose['theta']), math.sin(estimated_pose['theta']), angles='xy',scale_units='xy')
    #Add Text Informations
    plt.text(lim[1]-space_text, lim[3]-space_text2, Information_Landmarks_Best, fontsize = 6)
    plt.text(lim[1]-space_text, lim[3]/2-0.5, Information_best, fontsize = 8)
    plt.text(lim[1]-space_text, lim[2], Information_Landmarks_Real, fontsize = 6)
    #Limit the Axis Plot
    plt.axis(lim)
    plt.pause(0.0001)

#Function to find the Best Particle
#Param : List of Particles
#Return : The best particle
def SearchBest(particles):
    #find particle with highest weight with selection sort 
    maximum_weight = 0    
    for particle in particles:
        if particle['weight'] > maximum_weight:
            maximum_weight = particle['weight']
            best_particle = particle
    return best_particle

def angle_diff(angle1, angle2):
    return np.arctan2(np.sin(angle1-angle2), np.cos(angle1-angle2))

#Function to Plot an Ellipse Distribution with mean and sigma
#Param : The Mean(X and y) and Sigma of a distribution
#return : An Object For Plotting
def error_ellipse(position, sigma):

    covariance = sigma[0:2,0:2]
    eigenvals, eigenvecs = np.linalg.eig(covariance)

    #get largest eigenvalue and eigenvector
    max_ind = np.argmax(eigenvals)
    max_eigvec = eigenvecs[:,max_ind]
    max_eigval = eigenvals[max_ind]

    #get smallest eigenvalue and eigenvector
    min_ind = 0
    if max_ind == 0:
        min_ind = 1

    min_eigvec = eigenvecs[:,min_ind]
    min_eigval = eigenvals[min_ind]

    #chi-square value for sigma confidence interval
    chisquare_scale = 2.2789  

    #calculate width and height of confidence ellipse
    width = 2 * np.sqrt(chisquare_scale*max_eigval)
    height = 2 * np.sqrt(chisquare_scale*min_eigval)
    angle = np.arctan2(max_eigvec[1],max_eigvec[0])

    #generate covariance ellipse
    error_ellipse = Ellipse(xy=[position[0],position[1]], width=width, height=height, angle=angle/np.pi*180)
    error_ellipse.set_alpha(0.25)

    return error_ellipse
