import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError

from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Int8

class depthImage:
  def __init__(self):
    rospy.Subscriber("/vehicle1/rangecamera",Image,self.dimage_callback)
    self.bridge = CvBridge()
    self.height = -1
    self.width = -1
    self.depth_image = []
  def dimage_callback(self,data):
    try:
      self.depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="32FC1")
    except CvBridgeError as e:
      print(e)
    self.height, self.width = self.depth_image.shape
    cv2.imshow("depth", self.depth_image/10) #normalization
    cv2.waitKey(3)
  def getSize(self):
    return self.width,self.height



class imageGrabber:
  def __init__(self):
    rospy.Subscriber("vehicle1/camera1",Image,self.image_callback)
    self.bridge = CvBridge()
    self.height=-1
    self.width=-1
    self.channels=-1
    self.cv_image = []
  def image_callback(self,data):
    try:
      self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    self.height, self.width, self.channels = self.cv_image.shape
    #cv2.imshow("camera", self.cv_image)
    #cv2.waitKey(3)
  def getSize(self):
    return self.width,self.height



class GetPose:
  def __init__(self):
    rospy.Subscriber("/vehicle1/pose", Pose, self.p_callback)
    self.p = []
  def p_callback(self,data):
    q1 = data.orientation.x
    q2 = data.orientation.y
    q3 = data.orientation.z
    q4 = data.orientation.w
    newx = data.position.x
    newy = data.position.y
    newz = data.position.z
    newphi = math.atan2(2*(q1*q4+q2*q3),1-2*(q2*q2+q1*q1))
    newtheta = math.asin(2*(q2*q4-q1*q3))
    newpsi = math.atan2(2*(q3*q4+q1*q2),1-2*(q2*q2+q3*q3))
    self.p = [newx, newy, newz, newphi, newtheta, newpsi]



class GetVelocity:
  def __init__(self):
    rospy.Subscriber("/vehicle1/velocity", Twist, self.v_callback)
    self.v = []
  def v_callback(self,data):
    newu = data.linear.x
    newv = data.linear.y
    neww = data.linear.z
    newp = data.angular.x
    newq = data.angular.y
    newr = data.angular.z
    self.v = [newu, newv, neww, newp, newq, newr]



class Laser:
  def __init__(self):
    rospy.Subscriber("/vehicle1/multibeam", LaserScan, self.LS_callback)
    self.laser = []
  def LS_callback(self,data):
    self.laser = data.ranges 



class Trajectory:
#show given trajectory (update position)
  def __init__(self):
    self.tr_pub = rospy.Publisher("/vehicle2/dataNavigator", Odometry, queue_size=1)
  def tr_publish(self,data):
    x, y, z, phi, theta, psi = data
    t_msg = Odometry()
    t_msg.pose.pose.position.x = x
    t_msg.pose.pose.position.y = y
    t_msg.pose.pose.position.z = z
    t_msg.pose.pose.orientation.w = np.cos(phi/2)*np.cos(psi/2)*np.cos(theta/2)+np.sin(phi/2)*np.sin(theta/2)*np.sin(psi/2)
    t_msg.pose.pose.orientation.x = np.sin(phi/2)*np.cos(psi/2)*np.cos(theta/2)-np.cos(phi/2)*np.sin(theta/2)*np.sin(psi/2) 
    t_msg.pose.pose.orientation.y = np.cos(phi/2)*np.cos(psi/2)*np.sin(theta/2)+np.sin(phi/2)*np.cos(theta/2)*np.sin(psi/2)
    t_msg.pose.pose.orientation.z = np.cos(phi/2)*np.sin(psi/2)*np.cos(theta/2)-np.sin(phi/2)*np.sin(theta/2)*np.cos(psi/2)
    self.tr_pub.publish(t_msg)





class UwsimEnv(gym.Env):


  def __init__(self):
    
    # Subscriber	
    self.IG = imageGrabber()	
    self.DI = depthImage()
    self.L = Laser()
    self.State_p = GetPose()
    self.State_v = GetVelocity()
       
    # Publisher
    self.Thruster_pub = rospy.Publisher("/vehicle1/thrusters_input",Float64MultiArray ,queue_size=1)
    self.reset_pub = rospy.Publisher("/vehicle1/resetdata",Odometry ,queue_size=1)
    self.pause_pub = rospy.Publisher("/pause",Int8,queue_size=1)

    # observation spaces
    # state: x, y, z, psi, theta, gamma, u, v, w, p, q, r  (or IG.cv_image, DI.depth_image, L.laser) 

    # action space
    self.action_space = spaces.Box(low = -np.array(np.ones(5)), high = np.array(np.ones(5)))


    self.seed()


  def seed(self, seed = None):

    self.np_random, seed = seeding.np_random(seed)
    return ([seed])

  def step(self, action):
        
    #publish action
    tau1, tau2, tau3, tau4, tau5 = np.clip(action, self.action_space.low, self.action_space.high)
    a_msg = Float64MultiArray()
    a_msg.data = [tau1, tau2, tau3, tau4, tau5]
    self.Thruster_pub.publish(a_msg)
    rospy.sleep(0.1)

    #subscribe new state
    self.state = np.append(self.State_p.p, self.State_v.v)

    #calculate costs
    costs = 1###

    return self._get_obs(), -costs, False, None

  def reset(self):

    #set initial parameter
    msg = Odometry()
    self.state = np.array([1,1,7.5,0,0,1.27,0.1,0.2,0.3,0.4,0.5,0.6])###

    x, y, z, phi, theta, psi, u, v, w, p, q, r = self.state

    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.position.z = z # 4.5
    msg.pose.pose.orientation.w = np.cos(phi/2)*np.cos(psi/2)*np.cos(theta/2)+np.sin(phi/2)*np.sin(theta/2)*np.sin(psi/2)
    msg.pose.pose.orientation.x = np.sin(phi/2)*np.cos(psi/2)*np.cos(theta/2)-np.cos(phi/2)*np.sin(theta/2)*np.sin(psi/2) 
    msg.pose.pose.orientation.y = np.cos(phi/2)*np.cos(psi/2)*np.sin(theta/2)+np.sin(phi/2)*np.cos(theta/2)*np.sin(psi/2)
    msg.pose.pose.orientation.z = np.cos(phi/2)*np.sin(psi/2)*np.cos(theta/2)-np.sin(phi/2)*np.sin(theta/2)*np.cos(psi/2)

    msg.twist.twist.linear.x = u
    msg.twist.twist.linear.y = v
    msg.twist.twist.linear.z = w
    msg.twist.twist.angular.x = p
    msg.twist.twist.angular.y = q
    msg.twist.twist.angular.z = r

    self.reset_pub.publish(msg)

    #publish reset_flag
    flag = Int8()
    flag.data = 1
    self.pause_pub.publish(flag)
    return self._get_obs()
        
  def _get_obs(self):
    return self.state

