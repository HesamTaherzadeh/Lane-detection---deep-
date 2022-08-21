import numpy as np
import cv2
import time
import sys
import tensorflow as tf
import Camera



# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []

def detect_lanes(frame):
    small_img = cv2.resize(frame, (160,80))
    small_img = np.array(small_img)
    small_img = small_img[None,:,:,:]
    prediction = model.predict(small_img)[0] * 255
    lanes.recent_fit.append(prediction)
     # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis = 0)

    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))
    lane_image = cv2.resize(lane_drawn, (640,480))
    image = cv2.resize(frame, (640,480))
    lane_image = lane_image.astype(np.uint8)
    result = cv2.addWeighted(image, 1, lane_image, 1, 0)
    return result


lanes = Lanes()
camera = Camera.Camera(True)
model = tf.keras.models.load_model('LLDNet.h5')
camera.show_camera(detect_lanes, "/home/devspec5/Desktop/new/solidWhiteRight.mp4")
