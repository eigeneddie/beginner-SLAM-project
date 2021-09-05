from controller import Robot
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import copy
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

#Function to calculate global landmark position from odometry and LIDAR 
#Param : robot position now, distance to the landmark, bearing to the landmark
#return : landmark global position
def landmark_pose_odom(robot_pose, distance, bearing):
    landmark_pose = [0,0]  
    #Calculate the local frame from distance and bearing
    local_landmark_x = distance*math.cos(bearing)
    local_landmark_y = distance*math.sin(bearing)
    #Calculate the global frame of the landmark
    landmark_pose[0] = local_landmark_x*(math.cos(robot_pose[2])) - local_landmark_y*(math.sin(robot_pose[2])) + robot_pose[0]
    landmark_pose[1] = local_landmark_x*(math.sin(robot_pose[2])) + local_landmark_y*(math.cos(robot_pose[2])) + robot_pose[1] 
    return landmark_pose

#Function to setup the Two-Wheeled Motors
#Param : Robot instance
# Return : left motor and right motor objects        
def motorSetup(robot):
    #Created motor instances
    left_motor = robot.getMotor('motor1')
    right_motor = robot.getMotor('motor2')
    #Set velocity to 0
    left_motor.setPosition(float('inf'))
    left_motor.setVelocity(0.0)
    right_motor.setPosition(float('inf'))
    right_motor.setVelocity(0.0)   
    return left_motor,right_motor

#Function to setup the LIDAR
#Param : Robot Instance
#return : LIDAR Object
def LidarSetup(robot):
    lidar = robot.getLidar("lidar")
    lidar.enable(timestep)
    lidar.enablePointCloud()    
    return lidar

#Function to setup the Encoder at both wheels
#Param : Robot Instance
#return : left and right encoder objects
def EncoderSetup(robot):
    left_ps = robot.getPositionSensor('ps_1')    
    left_ps.enable(timestep)
    right_ps = robot.getPositionSensor('ps_2')    
    right_ps.enable(timestep)
    return left_ps,right_ps

#Function to setup the IMU 
#Param : Robot Instance
#return : IMU Sensor Object
def imuSetup(robot):
    imu = robot.getInertialUnit('imu')
    imu.enable(timestep)    
    return imu

#Function to setup the Distance Sensors
#Param : Robot Instance
#return : Left and Right Distance sensor objects
def distanceSetup(robot):
    left_ds = robot.getDistanceSensor('ds_1')
    left_ds.enable(timestep)
    right_ds = robot.getDistanceSensor('ds_2')
    right_ds.enable(timestep)
    return left_ds,right_ds
    
#Function to Read the Wheeled Encoders
#Param : Left and Right Encoders Objects
#return : list of values measured from left and right encoders
def ReadEncoder(left_ps,right_ps):
    ps_values = [0,0]
    ps_values[0] = left_ps.getValue()
    ps_values[1] = right_ps.getValue()        
    return ps_values

#Function to Read landmarks
#Param : Lidar Object, Robot Position Now, List of landmarks that has been approxed before
#return : List of landmark measured (which landmark, distance, measure) and Update the List of Landmark 
def ReadLandmark(lidar_val, lidar,robot_pose, landmark_mem):
    #Initiate the Sensing Array with the size of the amount of landmark x 2
    Sensing = np.zeros((landmark_amount,2))
    #Initiate Some Counting variables
    j = 0
    min = 99
    iBefore = -1
    #Initiate Landmark Global Position
    landmark_pose = [0,0]
    #Loop inside the Lidar Measurements
    for i in range (len(lidar_val)):
        #We limit the range measured on 0.5 meters for decreasing the error
        if( (lidar_val[i][3] <= 0.5) and (lidar_val[i][3] > 0) ):
            #If the i is closed with before, then that means it is the same object as before
            if(i != iBefore+1):
               #If not, then it is a new object, set the minimum value measured
               min = 99
            #Get the distance to the landmark from the outest layer
            distance = lidar_val[i][3]
            #We will always choose the minimum value of the distance because lidar tends to error at higher value
            if(distance < min):
                #Transform to a Bearing
                bearing = (130-i)*radian_lidar_unit
                if(bearing > math.pi):
                    bearing = bearing-2*math.pi
                #Update the minimum value
                min = distance
                #Calculate the Landmark Global Position based on robot position, distance, and bearing
                landmark_pose = landmark_pose_odom(robot_pose,distance,bearing)
                #Here is the code for determining the Id of the landmark
                #We will loop at the landmark position that has been APPROXIMATED with odometry information before, we call it memories
                #And based on that, we will compare the landmark position detected and landmark position MEMORIES
                for k in range(len(landmark_mem)):
                    #If we found the position of zero at this memories, it means it is the new landmark id!
                    if((landmark_mem[k][0] == 0) and (landmark_mem[k][1] == 0)):
                        j = k
                        #Update the RETURN values
                        Sensing[j][0] = distance
                        Sensing[j][1] = bearing
                        landmark_mem[k][0] = landmark_pose[0]
                        landmark_mem[k][1] = landmark_pose[1]
                        break #AND Break!
                    #We will try to compare the landmark position now and the memories 
                    else :
                        #Distance of the position landmark now and the memories
                        temp_dist = math.sqrt(math.pow((landmark_mem[k][0]-landmark_pose[0]),2) + math.pow((landmark_mem[k][1]-landmark_pose[1]),2))
                        if(temp_dist < 0.5): #If the distance is closed enough
                            j = k
                            #Update that landmark. It means it has been scanned before, not a new ID!
                            Sensing[j][0] = distance
                            Sensing[j][1] = bearing
                            landmark_mem[k][0] = landmark_pose[0]
                            landmark_mem[k][1] = landmark_pose[1]
                            break     
            iBefore = i
    return Sensing,landmark_mem

#Function to Read Yaw Position from IMU
#Param : IMU Object
#return : Yaw Position of robot in -pi to pi radian
def ReadYaw(imu):
    yaw = (imu.getRollPitchYaw())[2]  
    return yaw
        
#Function to Calculate the distance travelled by encoders
#Param : Rotary Encoder values now and Rotary Encoder values before
#return : The list of distance travelled for both wheel in meter    
def CalculateDist(ps_values,last_ps_values):
    dist_values = [0,0]
    for ind in range(2):
        #Calculate the Differences
        diff = ps_values[ind] - last_ps_values[ind]
        #Ignore when the differences are too small
        if diff<0.001:
            diff = 0
            ps_values[ind] = last_ps_values[ind]
        #Calculate the distance travelled
        dist_values[ind] = diff * encoder_unit #To Meter
    return dist_values

#Function to Calculate the position of the robot in global frame
#Param : List of distance travelled for both wheels, robot position before, IMU object
#return : Robot position now
def CalculatePose(dist_values, robot_pose,imu):    
    #Calculate the Angular and linear speed of the centre of the robot
    v = (dist_values[0] + dist_values[1])/2.0
    w = (dist_values[0] - dist_values[1])/distance_between_wheels
    #Read the IMU Sensor for getting the direction of the robot    
    dt = 1
    robot_pose[2] = ReadYaw(imu)
    #Calculate the x and y components of the linear speed of the centre of the robot
    vx = v*math.cos(robot_pose[2])
    vy = v*math.sin(robot_pose[2])
    #Increment or Decrement the robot position to get the position now    
    robot_pose[0] += (vx*dt)
    robot_pose[1] += (vy*dt)    
    return robot_pose

#Function to update the list of Wheeled encoder Values Before
#Param : List of Wheeled Encoder Values now
#return : List of Wheeled Encoder Values Before
def updatePs(ps_values):
    last_ps_values = [0,0]   
    for ind in range(2):
        last_ps_values[ind] = ps_values[ind]    
    return last_ps_values

#Function to update the Robot Position Before
#Param : Robot Position now
#return : Robot Position Before
def updatePose(robot_pose):
    last_robot_pose = [0,0,0]   
    for ind in range(3):
        last_robot_pose[ind] = robot_pose[ind]    
    return last_robot_pose

#Function to Calculate the Odometry Informations of Robot
#Param : Robot position now and Robot position before
#return : Odometry Values that consists of Delta Rotation1, Delta Translation, Delta Rotation2
def CalculateOdom(robot_pose, last_robot_pose):
    Odom_values = [0,0,0]
    # DeltaRot1, DeltaTrans, DeltaRot2
    # DeltaRot2 will be good if imu is good
    Odom_values[0] = (math.atan2((robot_pose[1]-last_robot_pose[1]) , (robot_pose[0]-last_robot_pose[0]))) - last_robot_pose[2]
    Odom_values[1] = math.sqrt(math.pow((robot_pose[1]-last_robot_pose[1]),2) + math.pow((robot_pose[0]-last_robot_pose[0]),2))
    Odom_values[2] = robot_pose[2] - last_robot_pose[2] - Odom_values[0]
    return Odom_values

#Function to do Obstacle Avoidances
#Using A sonar because of its efficiency in time and computation (Not Like LIDAR)
#Param : Left and right distance sensors objects and A counter variable
#Return : The left and right wheel's speed and counter variable
def Avoid(lidar_val,Avoid_Counter):
    #If obstacle object ahead has been detected before
    if Avoid_Counter > 0:
        Avoid_Counter -= 1
        #Rotate the motor until counter is ended
        leftSpeed = -max_speed
        rightSpeed = max_speed
    #If there is no obstacle ahead detected
    else:  # read sensors
        #Detect objects ahead
        for i in range(120,140): #Ahead between indeks 120 to 140
            if ( (lidar_val[i][3] < 0.15) and (lidar_val[i][3] > 0) ):
                Avoid_Counter = 10
        #Drive Forward
        leftSpeed = max_speed
        rightSpeed = max_speed
        
    return leftSpeed, rightSpeed, Avoid_Counter
