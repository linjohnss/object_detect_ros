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
# hog = cv2.HOGDescriptor()
# hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
pub = rospy.Publisher("/output/image_raw/compressed", CompressedImage, queue_size=100)
net = cv2.dnn.readNet("/home/point/robot_ws/src/facedetect_ros/src/weights/yolov3-tiny.weights", "/home/point/robot_ws/src/facedetect_ros/src/configuration/yolov3-tiny.cfg")
classes = []
with open("/home/point/robot_ws/src/facedetect_ros/src/configuration/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

font = cv2.FONT_HERSHEY_SIMPLEX
frame_id = 0
starting_time = time.time()

tsk = [] 

def callback(msg):
    print("Received an image!")
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
    except CvBridgeError, e:
        print(e)
    else:
        t = threading.Thread(target = yolo_v3(cv2_img))
        t.start()
        tsk.append(t)

def yolo_v3(frame):
    # frame_id += 1
    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (640, 480), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Visualising data
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                # Object detected
                center_x = int(detection[0] * 640)
                center_y = int(detection[1] * 480)
                w = int(detection[2] * 640)
                h = int(detection[3] * 480)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)

    msg = CompressedImage()
    msg.header.stamp = rospy.Time.now()
    msg.format = "jpeg"
    msg.data = np.array(cv2.imencode('.jpg', frame)[1]).tostring()
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
    except rospy.ROSInterruptException:
        print "Shutting down ROS Image feature detector module"
        for tt in tsk:
            tt.join()
if __name__ == '__main__':
    main()