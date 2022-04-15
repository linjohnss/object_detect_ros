#! /usr/bin/python

# rospy for the subscriber
from turtle import pu
import rospy
# ROS Image message
from sensor_msgs.msg import CompressedImage
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import numpy as np
import time
import threading


# Instantiate CvBridge
bridge = CvBridge()
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
pub = rospy.Publisher("/output/image_raw/compressed", CompressedImage, queue_size=1)
tsk = [] 

def callback(msg):
    print("Received an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print(e)
    else:
        gray_image = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        t = threading.Thread(target = hog_detect(gray_image, cv2_img))
        t.start()
        t.join()
        # tsk.append(t)
        
def fast_feature(gray_image, cv2_img):
    fast = cv2.FastFeatureDetector_create(threshold=100,nonmaxSuppression=True,type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
    kp = fast.detect(gray_image,None)
    kp_image = cv2.drawKeypoints(cv2_img, kp, None, color=(0, 255, 0), flags=0)
    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = "jpeg"
    msg.data = np.array(cv2.imencode('.jpg', cv2_img)[1]).tostring()
    # Publish new image
    pub.publish(msg)


def hog_detect(gray_image, cv2_img):
    boxes, weights = hog.detectMultiScale(gray_image, winStride=(8,8), padding = (32, 32))
        
    for (x, y, w, h) in boxes:  
        # display the detected boxes in the colour picture
        kp_image = cv2.rectangle(cv2_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = "jpeg"
    msg.data = np.array(cv2.imencode('.jpg', cv2_img)[1]).tostring()
    # Publish new image
    pub.publish(msg)


def main():
    rospy.init_node('image_listener')
    # Define your image topic
    image_topic = "/camera/color/image_raw/compressed"
    # Set up your subscriber and define its callback
    rospy.Subscriber(image_topic, CompressedImage, callback, queue_size=1)
    # Spin until ctrl + c
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down ROS Image feature detector module"
        # for tt in tsk:
        #     tt.join()

if __name__ == '__main__':
    main()