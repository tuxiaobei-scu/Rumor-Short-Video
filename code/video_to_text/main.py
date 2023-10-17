'''


'''
from multiprocessing import Pool
from PIL import Image
import sys
from imageocr import imageocr
# from Extract_audio import extract_audio
from Imghash import imghash
from Extract_frame import V2frame
# from wav2text import wav2text
import os
import shutil
from Sentence_distance import sentence_distance
import time
import json
import cv2
audio_path = ''
frame_path = ''
datasave_path = './data'
video_path = './videos'
# 清空文件夹


def clear_dir_new(path):
    try:
        shutil.rmtree(path)
        os.mkdir(path)
    except Exception as e:
        print(e)


def clear_dir(path):
    try:
        shutil.rmtree(path)
        # os.mkdir(path)
    except Exception as e:
        print(e)
# 环境初始化


def enc_init():
    print('[enc_init] working')
    clear_dir_new(audio_path)
    clear_dir_new(frame_path)


def get_string_from_list(A):
    if type(A) == list:
        ret = ''
        for i in A:
            ret += get_string_from_list(i)
        return ret
    if type(A) == str:
        return A
    return ""


ocr_ret = []


def ocr_frame(frame):
    img_path = os.path.join(frame_path, str(frame)+'.jpg')
    now_ocr_result = imageocr.work(img_path)
    ocr_ret.append({'frame': frame, 'result': now_ocr_result})


def get_frame(x):
    return x['frame']


def OCR(path):
    imgH = imghash()
    cropimgH = imghash()
    # 返回结果
    ret = []

    # 当前帧
    now_frame = -15
    # 上一次执行ocr的帧
    last_frame = 0
    # 上一次的ocr结果拼接字符串
    last_ocr_result_string = "ocr at first"
    # 限
    k = 8
    # 历史最高k点
    kmax = 20
    # 匹配标识
    marchflag = False

    while True:
        now_frame += 15

        img_path = os.path.join(frame_path, str(now_frame)+'.jpg')

        if not os.path.exists(img_path):
            print('[OCR all] done', now_frame)
            break

        # 相似度高，无需ocr
        if not imgH.work(Image.open(img_path), k):
            print("continue " + str(now_frame))
            continue

        print('[OCR working] ocr at frame', now_frame)

        # 进行ocr
        now_ocr_result = imageocr.work(img_path, cropimgH)

        # 将识别结果添加
        ret.append({'frame': now_frame, 'result': now_ocr_result})
        # print('[OCR done] ocr at frame', now_frame)
    return ret


# 处理视频
def video_work(path, video_id):
    st = time.time()
    print('[video_work] working')
    enc_init()
    v2f = V2frame()
    # w2t = wav2text()
    # 分离wav
    # print('[extract audio]', path, audio_path)
    # extract_audio(path, audio_path)
    # 分离帧
    print('[extract frame]', path, audio_path)
    v2f.work(path, frame_path)

    # 进行wav处理
    # print('[wav2text] working')
    # wav_result = w2t.work(audio_path+'/a.wav')
    # print('[wav_result]', wav_result)

    # 进行ocr处理
    ocr_result = OCR(frame_path)
    print('ocr_result', ocr_result)

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 保存结果
    wav_result = {}
    clear_dir(audio_path)
    clear_dir(frame_path)
    print("time:     " + str(time.time() - st))
    return {"video_id": video_id, 'wav_result': wav_result, 'ocr_result': ocr_result, "fps": fps}


if __name__ == '__main__':
    file = sys.argv[1]
    r = open(file, "r").readlines()
    for name in r:
        name = name.strip()
        data_path = os.path.join(datasave_path, name[:-4]+'.json')
        if os.path.exists(data_path):
            # print("continue " + name)
            continue
        if name == '':
            continue
        path = os.path.join(video_path, name)
        audio_path = './audio/' + name[:-4]
        try:
            os.mkdir(audio_path)
        except:
            pass
        frame_path = './frame/' + name[:-4]
        try:
            os.mkdir(frame_path)
        except:
            pass
        result = video_work(path, name[:-4])

        with open(data_path, 'w') as f:
            f.write(json.dumps(result, ensure_ascii=False))
