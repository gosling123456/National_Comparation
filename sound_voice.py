# -*- coding：utf-8 -*-
# -*- author：zzZ_CMing  CSDN address:https://blog.csdn.net/zzZ_CMing
# -*- 2018/07/12; 15:19
# -*- python3.5
import time

import threading

import urllib
import webbrowser

import pyaudio
import wave
import requests
import random
from concurrent.futures import ThreadPoolExecutor
import pygame

# 开启录音功能
# 麦克风采集的语音输入
# 输入文件的path
pool = ThreadPoolExecutor(max_workers=4)


def play_sound(path):
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()


def get_audio(filepath):
    filepath = "input.wav"
    CHUNK = 256
    FORMAT = pyaudio.paInt16
    CHANNELS = 1  # 声道数
    RATE = 16000  # 采样率
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = filepath
    try:
        p = pyaudio.PyAudio()
    except Exception as e:
        print(e)
    # 打开一个新的音频stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("*" * 10, "开始录音：请在5秒内输入语音")
    frames = []  # 存放录音数据
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    print("*" * 10, "录音结束\n")

    stream.stop_stream()
    stream.close()
    p.terminate()
    # 将音频数据保存到wav文件之中
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


# 自己的APPID AK SK
APP_ID = '26273092'
API_KEY = 'XmOEfSbzqrmmqmptosR52fgp'
SECRET_KEY = 'etEMsii801j5XZGGXxtfpx80nLfeVKSK'
from aip import AipSpeech

# 准备好APPID、APIKEY、SecretKey
# 创建识别对象
client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)


# 将上一步录音 通过调用百度api实现语音识别

# 读取文件
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        read_sound = fp.read()
        return read_sound


def predict():
    # 调用百度AI的接口, 识别本地文件
    return client.asr(get_file_content('input.wav'), 'wav', 16000, {
        'dev_pid': 1537,
    })


# 免费、无需注册、只需要发送get请求就可实现聊天的青云客智能机器人，直接调用接口即可。
def talkWithRobot(msg):
    url = 'http://api.qingyunke.com/api.php?key=free&appid=0&msg={}'.format(urllib.parse.quote(msg))
    html = requests.get(url)
    return html.json()["content"]


flag_respond = False
flag = False
flag_music = False


# 随机播放音乐
def rand_music():
    time.sleep(2)
    musil_list = ['小兔子乖乖.mp3', '蓝精灵.mp3', '拔萝卜.mp3', '世上只有妈妈好.mp3', '两只老虎.mp3', '健康歌.mp3', '我有一只小毛驴.mp3', '外婆的澎湖湾.mp3',
                  '鲁冰花.mp3', '爱我你就抱抱我.mp3']
    key = random.randint(0, 9)
    print(musil_list[key])
    play_sound("./music/" + musil_list[key])
    flag_sound = False
    if flag_music:
        # if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()


# 将机器人返回的对话 调用语音合成接口 返回语音
def communication(str_Result):
    res = talkWithRobot(str_Result)
    if '未获取到相关信息' not in res:

        print(res)
        result = client.synthesis(res, 'zh', 1, {
            'per': 4,
            'spd': 3,  # 速度
            'vol': 7  # 音量
        })
        if not isinstance(result, dict):
            with open('test.mp3', 'wb') as f:
                f.write(result)

        # 播放合成后的语音
        play_sound("test.mp3")


def commun(str_Result):
    return str_Result


def awake():
    awake_list = ['awake_one.mp3', 'awake_two.mp3', 'awake_three.mp3']

    key = random.randint(0, 2)
    print(awake_list[key])
    play_sound("./sound/" + awake_list[key])
    return awake_list[key]


def test():
    while True:

        if flag:
            print(1)
        else:
            break


def start_or_stop(str_Result):
    global flag
    if "休息" in str_Result:
        play_sound("./sound/stop_work.mp3")
        # global flag
        flag = False

    elif "工作" in str_Result:
        print("工作")

        flag = True
        play_sound("./sound/start_work.mp3")


def openmusic(str_Result):
    global flag_music
    global flag_respond
    if "菲菲" in str_Result or "飞飞" in str_Result:

        flag_respond = True
        print("唤醒")
        awake()
    elif "说话" in str_Result and flag_respond:
        flag_respond = False
    elif "工作" in str_Result or "休息" in str_Result and flag_respond:
        start_or_stop(str_Result)
    elif "音乐" in str_Result and flag_respond:
        rand_music()
    elif "闭嘴" in str_Result and flag_respond:
        flag_music = True
        pygame.mixer.music.pause()
    else:
        if flag_respond:
            communication(str_Result)


while True:
    in_path = "input.wav"
    get_audio(in_path)
    results = predict()  # 语音识别返回的结果是字典格式
    Result = results["result"]  # 将字典中的”result“值取出
    str_Result = "".join(Result)
    print(str_Result)  # 调用识别方法, 并输出# 讲列表格式转换为字符串格式
    openmusic(str_Result)