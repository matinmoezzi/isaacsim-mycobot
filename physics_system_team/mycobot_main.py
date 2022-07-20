from pymycobot.mycobot import MyCobot
from pymycobot import PI_PORT, PI_BAUD
from DOMAIN import SPEED_DEFAULT,JOINT_LOWER_BOUND,JOINT_UPPER_BOUND
from gym import spaces

from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision

import numpy as np
import argparse
import math
import time
import sys
import cv2

# openAI gym boundbox
class MyCobot():
    def __init__(self):
        self.mc = MyCobot(PI_PORT, PI_BAUD)
        self.action_space = spaces.Box(
            np.array(JOINT_LOWER_BOUND).astype(np.float32),
            np.array(JOINT_UPPER_BOUND).astype(np.float32)
        )

        self.cv_parse_init()
    
    def step_end_effector(self,
                          coords: list,
                          speed=SPEED_DEFAULT) -> None:
        self.mc.send_coords(coords,speed=speed)

    def step(self,
             joint_angles: list,
             speed = SPEED_DEFAULT,
             angles_check = False) -> None:
        # Check if the joint_angles input was out of domain.
        if not self.action_space(joint_angles):
            raise ValueError("You have input joint angles which are out of domain.")
            exit()
        
        # Manipulate the to set destination
        self.mc.send_angles(joint_angles,speed=speed)

        if angles_check:
            while not self.mc.is_in_position(joint_angles,0):
                # Option 1:
                self.mc.send_angles(joint_angles,speed=math.ceil(speed/3))
                # Option 2:
                # time.sleep(5)

        # if collision?

    def get_observation(self,
                        headless=False):
                        
        counter, fps = 0, 0
        # start_time = time.time()

        # Start capturing video input from the camera
        cap = cv2.VideoCapture(int(self.args.cameraId))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.frameWidth)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.frameHeight)

        # Visualization parameters
        row_size = 20  # pixels
        left_margin = 24  # pixels
        text_color = (0, 0, 255)  # red
        font_size = 1
        font_thickness = 1
        fps_avg_frame_count = 10

        # Initialize the object detection model
        base_options = core.BaseOptions(
            file_name=self.args.model, use_coral=bool(self.args.enableEdgeTPU), num_threads=self.args.numThreads)
        detection_options = processor.DetectionOptions(
            max_results=3, score_threshold=0.3)
        options = vision.ObjectDetectorOptions(
            base_options=base_options, detection_options=detection_options)
        detector = vision.ObjectDetector.create_from_options(options)

        # Continuously capture images from the camera and run inference
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                sys.exit(
                    'ERROR: Unable to read from webcam. Please verify your webcam settings.'
                )

            counter += 1
            image = cv2.flip(image, 1)

            # Convert the image from BGR to RGB as required by the TFLite model.
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cap.release()
            cv2.destroyAllWindows()
            
            return rgb_image
            # # Create a TensorImage object from the RGB image.
            # input_tensor = vision.TensorImage.create_from_array(rgb_image)

            # # Run object detection estimation using the model.
            # detection_result = detector.detect(input_tensor)

            # # Draw keypoints and edges on input image
            # image = utils.visualize(image, detection_result)

            # # Calculate the FPS
            # if counter % fps_avg_frame_count == 0:
            #     end_time = time.time()
            # fps = fps_avg_frame_count / (end_time - start_time)
            # start_time = time.time()

            # # Show the FPS
            # fps_text = 'FPS = {:.1f}'.format(fps)
            # text_location = (left_margin, row_size)
            # cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
            #             font_size, text_color, font_thickness)

            # Stop the program if the ESC key is pressed.
            if cv2.waitKey(1) == 27:
                break
            cv2.imshow('object_detector', image)

        # cap.release()
        # cv2.destroyAllWindows()

    def reset(self,
              origin_angles = [0,0,0,0,0,0],
              speed = SPEED_DEFAULT) -> None:
        self.mc.send_angles(origin_angles,speed=speed)

    def cv_parse_init(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument(
            '--model',
            help='Path of the object detection model.',
            required=False,
            default='efficientdet_lite0.tflite')
        parser.add_argument(
            '--cameraId', help='Id of camera.', required=False, type=int, default=0)
        parser.add_argument(
            '--frameWidth',
            help='Width of frame to capture from camera.',
            required=False,
            type=int,
            default=640)
        parser.add_argument(
            '--frameHeight',
            help='Height of frame to capture from camera.',
            required=False,
            type=int,
            default=480)
        parser.add_argument(
            '--numThreads',
            help='Number of CPU threads to run the model.',
            required=False,
            type=int,
            default=4)
        parser.add_argument(
            '--enableEdgeTPU',
            help='Whether to run the model on EdgeTPU.',
            action='store_true',
            required=False,
            default=False)
        self.args = parser.parse_args()

