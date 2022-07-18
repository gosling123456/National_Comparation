# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Main script to run the object detection routine."""
import argparse
import sys
import time
import  RPi.GPIO as GPIO
import cv2
from object_detector import ObjectDetector
from object_detector import ObjectDetectorOptions
import utils

from concurrent.futures import ThreadPoolExecutor
import threading

SensorRight = 16
SensorLeft  = 12

PWMA   = 18
AIN1   = 22
AIN2   = 27

PWMB   = 23
BIN1   = 25
BIN2   = 24

BtnPin  = 19
Gpin    = 5
Rpin    = 6

def car_up(speed,t_time):
    t_up(speed,t_time)
    #t_stop(0.1)

def car_left(speed,t_time):
    t_left(speed,t_time)
    t_stop(0.1)
def car_right(speed,t_time):
    t_right(speed,t_time)
    t_stop(0.1)
def car_down(speed,t_time):
    t_down(speed,t_time)
    t_stop(0.1)

#智能小车运动函数 
def t_up(speed,t_time):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2,False)#AIN2
    GPIO.output(AIN1,True) #AIN1

    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2,False)#BIN2
    GPIO.output(BIN1,True) #BIN1
    time.sleep(t_time)
        
def t_stop(t_time):
    L_Motor.ChangeDutyCycle(0)
    GPIO.output(AIN2,False)#AIN2
    GPIO.output(AIN1,False) #AIN1

    R_Motor.ChangeDutyCycle(0)
    GPIO.output(BIN2,False)#BIN2
    GPIO.output(BIN1,False) #BIN1
    time.sleep(t_time)
        
def t_down(speed,t_time):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2,True)#AIN2
    GPIO.output(AIN1,False) #AIN1

    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2,True)#BIN2
    GPIO.output(BIN1,False) #BIN1
    time.sleep(t_time)

def t_left(speed,t_time):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2,True)#AIN2
    GPIO.output(AIN1,False) #AIN1

    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2,False)#BIN2
    GPIO.output(BIN1,True) #BIN1
    time.sleep(t_time)

def t_right(speed,t_time):
    L_Motor.ChangeDutyCycle(speed)
    GPIO.output(AIN2,False)#AIN2
    GPIO.output(AIN1,True) #AIN1

    R_Motor.ChangeDutyCycle(speed)
    GPIO.output(BIN2,True)#BIN2
    GPIO.output(BIN1,False) #BIN1
    time.sleep(t_time)
    
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)       # 按物理位置给GPIOs编号
GPIO.setup(Gpin, GPIO.OUT)     # 设置绿色Led引脚模式输出
GPIO.setup(Rpin, GPIO.OUT)     # 设置红色Led引脚模式输出
# 设置输入BtnPin模式，拉高至高电平(3.3V) 
GPIO.setup(BtnPin, GPIO.IN, pull_up_down=GPIO.PUD_UP)    
GPIO.setup(SensorRight,GPIO.IN)
GPIO.setup(SensorLeft,GPIO.IN)

GPIO.setup(AIN2,GPIO.OUT)
GPIO.setup(AIN1,GPIO.OUT)
GPIO.setup(PWMA,GPIO.OUT)

GPIO.setup(BIN1,GPIO.OUT)
GPIO.setup(BIN2,GPIO.OUT)
GPIO.setup(PWMB,GPIO.OUT)
    
L_Motor= GPIO.PWM(PWMA,100)
L_Motor.start(0)
R_Motor = GPIO.PWM(PWMB,100)
R_Motor.start(0)
    

    
pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix='ThreadPool')

def run(model: str, camera_id: int, width: int, height: int, num_threads: int,
        enable_edgetpu: bool) -> None:
  """Continuously run inference on images acquired from the camera.

  Args:
    model: Name of the TFLite object detection model.
    camera_id: The camera id to be passed to OpenCV.
    width: The width of the frame captured from the camera.
    height: The height of the frame captured from the camera.
    num_threads: The number of CPU threads to run the model.
    enable_edgetpu: True/False whether the model is a EdgeTPU model.
  """

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Start capturing video input from the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  # Visualization parameters
  row_size = 20  # pixels
  left_margin = 24  # pixels
  text_color = (0, 0, 255)  # red
  font_size = 1
  font_thickness = 1
  fps_avg_frame_count = 10

  # Initialize the object detection model
  options = ObjectDetectorOptions(
      num_threads=num_threads,
      score_threshold=0.3,
      max_results=3,
      enable_edgetpu=enable_edgetpu)
  detector = ObjectDetector(model_path=model, options=options)

  # Continuously capture images from the camera and run inference
  count = 0

  while cap.isOpened():
    start2=time.time()
    start=time.time()
    success, image = cap.read()
    end=time.time()
    #print('video,',start-end)
    if not success:
      sys.exit(
          'ERROR: Unable to read from webcam. Please verify your webcam settings.'
      )

    counter += 1
    image = cv2.flip(image, 1)
    end=time.time()
    #print('video2,',start-end)
    # Run object detection estimation using the model.
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    start=time.time()
    detections = detector.detect(rgb_image)
    #print(detections)
    end=time.time()
    #print('lantency',start-end)     
    # Draw keypoints and edges on input image
    image,res = utils.visualize(image, detections,counter)
    x,y,x_p =0,0,0
    for detection in detections:
        person = detection.categories[0].label
        if person == "person":
                #print(person)
            (left,top,right,bottom)= detection.bounding_box
            #print((x,y,z,w))
            x = (left + right)/2
            y = (top + bottom)/2
            x_p = int(round(x))
            print(x_p)
    x_Lower = 70
    x_Upper = 230
    flag = True
    #flag =True
    if "person" not in res:
        t_stop(0)
    if counter %10 == 0:
        if"person" in res:
            # 判断X方向范围来判断机器人的运动
            if x_p > x_Lower and x_p < x_Upper:
                try:
                    #pool.submit(car_up,70,0.4)
                    t_up(80,0.4)
                    #t_stop(1)
                except KeyboardInterrupt:  # 当按下Ctrl+C时，将执行子程序destroy()。
                    GPIO.cleanup()               
            elif x_p < x_Lower:
                try:
                #utils.t_left(20,3)
                    #pool.submit(car_right,50,0.4)
                    t_right(50,0.5)
                    t_stop(0)
                    print("leftleftleftleftleftleftleftleftleft")
                except KeyboardInterrupt:  # 当按下Ctrl+C时，将执行子程序destroy()。
                    GPIO.cleanup()
            elif x_p > x_Upper:
                try:
                    #pool.submit(car_left,50,0.4)
                    t_left(50,0.5)
                    t_stop(0)
                #utils.t_right(30,3)
                    print("rightrightrightrightrightrightright")
                except KeyboardInterrupt:  # 当按下Ctrl+C时，将执行子程序destroy()。
                    GPIO.cleanup()
       


    # Calculate the FPS
    if counter % fps_avg_frame_count == 0:
        end_time = time.time()
        fps = fps_avg_frame_count / (end_time - start_time)
        start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)
    
    # Stop the program if the ESC key is pressed.
    if cv2.waitKey(1) == 27:
      break
    cv2.imshow('object_detector', image)
    # car movement
    #utils.movement(detections)
    end2=time.time()
    #print('all',start2-end2)
  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
      default=300)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      type=int,
      default=200)
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
  args = parser.parse_args()

  run(args.model, int(args.cameraId), args.frameWidth, args.frameHeight,
      int(args.numThreads), bool(args.enableEdgeTPU))



if __name__ == '__main__':
  #pool.submit(infrad_avoid)
  #pool.submit(main)
  main()
  #infrad_avoid()
  
  

