import pygame

#pygame.init()
pygame.mixer.init() # 初始化混音器模块（pygame库的通用做法，每一个模块在使用时都要初始化pygame.init()为初始化所有的pygame模块，可以使用它也可以单初始化这一个模块）
pygame.mixer.music.load("/home/pi/pi4b_tensorflow_lite/raspberry_pi_object/tt.mp3")  # 加载音乐  
pygame.mixer.music.set_volume(0.8)# 设置音量大小0~1的浮点数
pygame.mixer.music.play() # 播放音频
