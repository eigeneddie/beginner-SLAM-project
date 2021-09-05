from controller import Robot
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import copy
from HardwareInterfaceFunct import *
from SLAMFunct import *
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

#Function to run the robot
#This function is an infinite Loop, which is the main function
def run_robot(robot):
    #Setup the motor
    [left_motor,right_motor] = motorSetup(robot)
    # Set up the Position Sensor
    [left_ps,right_ps] = EncoderSetup(robot)
    #Setup Distance Sensor
    [left_ds,right_ds] = distanceSetup(robot)
    #Setup IMU
    imu = imuSetup(robot)
    #setup Lidar
    lidar = LidarSetup(robot)
    
    #Prepare the Lists for Odometry
    ps_values = [0,0]
    dist_values = [0,0]
    Odom_values = [0,0,0]
    robot_pose = [0,0,0]
    last_robot_pose = [0,0,0]
    last_ps_values = [0,0]
    
    #Prepare Variable for Obstacle Avoidance
    Avoid_Counter = 0
    
    #Prepare Variables For LIDAR
    Sensing = np.zeros((landmark_amount,2))    
    ids = []
    dists = []
    bears = []
    landmark_mem = np.zeros((landmark_amount,2))

    #The GOAL Dictionaries for feeding it to SLAM
    sensor_and_odom = dict()
    
    #### Read Landmark
    directory = "D:/!SEMESTER9/KSC/TugasRobot/Tubes/fastSLAM_framework"
    print("Reading landmark positions")
    landmarks = read_landmarks((directory + "/data/world.dat"), " ")
    ### Init Paricles
    num_particles = 100
    num_landmarks = len(landmarks)
    #create particle set
    particles = particleInitialization(num_particles, num_landmarks)

    #INFINITE LOOP
    while robot.step(timestep) != -1:
        ######################################
        #ODOMETRY PROCESS
        #Read Wheeled Encoder
        ps_values = ReadEncoder(left_ps,right_ps)
        print("--------------------------")
        #Calculate Distances
        dist_values = CalculateDist(ps_values,last_ps_values)        
        #compute linear and angular velocity for robot
        robot_pose = CalculatePose(dist_values, robot_pose,imu)  
        #Calculate Odometry Information
        Odom_values = CalculateOdom(robot_pose,last_robot_pose)
        print("Odometry Values : {}".format(Odom_values))
        sensor_and_odom['odom'] = {'r1':float(Odom_values[0]), 'r2':float(Odom_values[2]), 't':float(Odom_values[1])}
        #update last values
        last_ps_values = updatePs(ps_values)
        last_robot_pose = updatePose(robot_pose)
        
        ########################################
        #SENSING LANDMARK PROCESS
        #Zero it for Sensing Landmark
        ids = []
        dists = []
        bears = []
        #Sensing Landmarks from Lidars
        #Get the Lidar Measurements
        lidar_val = lidar.getRangeImageArray()
        [Sensing, landmark_mem] = ReadLandmark(lidar_val,lidar, robot_pose, landmark_mem)
        #Processing the List of Landmarks measured in one timestep
        for i in range(Sensing.shape[0]):
            if(Sensing[i][0] != 0):
                ids = np.append(ids, int(i+1))
                dists = np.append(dists, float(Sensing[i][0]))
                bears = np.append(bears, float(Sensing[i][1]))
                print("Detect landmark ",i,"at ", Sensing[i][1], " deg with value of ", Sensing[i][0])
        sensor_and_odom["sensor"] = {'id':ids, 'range' : dists, 'bearing' : bears}
        
        ############################################
        #SLAM Algorithm
        #predict particles by sampling from motion model with odometry info
        sample_motion_model(sensor_and_odom['odom'], particles)

        #evaluate sensor model to update landmarks and calculate particle weights
        eval_sensor_model(sensor_and_odom['sensor'], particles)

        #plot filter state
        plot_particles(particles, landmarks)

        #calculate new set of equally weighted particles
        particles = resamplingProcedure(particles)

        ############################################
        #SPEED PLANNING AND OBSTACLE AVOIDANCE
        #Obstacle Avoidances Planning to get the left and right speed
        [left_speed, right_speed, Avoid_Counter] = Avoid(lidar_val,Avoid_Counter)
        
        ############################################
        #ACTUATE
        #Actuate the Motor from Speed Planning
        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)       
        

if __name__ == "__main__":
    # create the Robot instance.
    robot = Robot()  
    # Run the Main Function
    run_robot(robot)

