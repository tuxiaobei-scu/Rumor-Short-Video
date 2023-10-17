import random
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
import cv2
from PIL import Image
import math
import sys


class imageocr:
    ocr_detection = pipeline(
        Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
    ocr_recognition = pipeline(
        Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-general_damo')

    def crop_image(img, position):
        def distance(x1, y1, x2, y2):
            return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))
        position = position.tolist()
        for i in range(4):
            for j in range(i+1, 4):
                if (position[i][0] > position[j][0]):
                    tmp = position[j]
                    position[j] = position[i]
                    position[i] = tmp
        if position[0][1] > position[1][1]:
            tmp = position[0]
            position[0] = position[1]
            position[1] = tmp

        if position[2][1] > position[3][1]:
            tmp = position[2]
            position[2] = position[3]
            position[3] = tmp

        x1, y1 = position[0][0], position[0][1]
        x2, y2 = position[2][0], position[2][1]
        x3, y3 = position[3][0], position[3][1]
        x4, y4 = position[1][0], position[1][1]

        corners = np.zeros((4, 2), np.float32)
        corners[0] = [x1, y1]
        corners[1] = [x2, y2]
        corners[2] = [x4, y4]
        corners[3] = [x3, y3]

        img_width = distance((x1+x4)/2, (y1+y4)/2, (x2+x3)/2, (y2+y3)/2)
        img_height = distance((x1+x2)/2, (y1+y2)/2, (x4+x3)/2, (y4+y3)/2)

        corners_trans = np.zeros((4, 2), np.float32)
        corners_trans[0] = [0, 0]
        corners_trans[1] = [img_width - 1, 0]
        corners_trans[2] = [0, img_height - 1]
        corners_trans[3] = [img_width - 1, img_height - 1]

        transform = cv2.getPerspectiveTransform(corners, corners_trans)
        dst = cv2.warpPerspective(
            img, transform, (int(img_width), int(img_height)))
        return dst

    def bubble_sort(A):
        for i in range(1, A.shape[0]):
            for j in range(0, A.shape[0]-i):
                if A[j+1][7] < A[j][1] or ((not A[j][7] < A[j+1][1]) and A[j][0] > A[j+1][0]):
                    B = np.copy(A[j])
                    A[j] = A[j+1]
                    A[j+1] = B
        return A

    def work(img_path, cropimgH):
        ret = []
        image_full = cv2.imread(img_path)
        # print(img_path)
        try:
            det_result = imageocr.ocr_detection(image_full)
            det_result = det_result['polygons']

            det_result = imageocr.bubble_sort(det_result)
        except:
            return ret

        for i in range(det_result.shape[0]):
            # print("ssssssssss " + str(i))
            try:
                pts = det_result[i].reshape([4, 2])
                image_crop = imageocr.crop_image(image_full, pts)
                img = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
                # img.save("tests/" + str(random.randint(1, 10000)) + '.jpg')
                if not cropimgH.work(img, 16):
                    # print("continue " + str(img_path) + " " + str(i))
                    continue
                # img.save("tests/" + str(img_path)[35:-4] + "-" + str(i) + '.jpg')
                result = imageocr.ocr_recognition(image_crop)
                # print("box: %s" % ','.join([str(e) for e in list(pts.reshape(-1))]))
                # print("text: %s" % result['text'])
                # print( type(result['text']) )
                ret.append(result['text'])
            except:
                print("fail " + str(img_path) + " " + str(i))
        return ret


if __name__ == '__main__':
    print(imageocr.work('/home/tuxiaobei/video_to_text/frame/171.jpg'))
    print(imageocr.work('/home/tuxiaobei/video_to_text/frame/271.jpg'))
    print(imageocr.work('/home/tuxiaobei/video_to_text/frame/371.jpg'))
    print(imageocr.work('/home/tuxiaobei/video_to_text/frame/471.jpg'))
    print(imageocr.work('/home/tuxiaobei/video_to_text/frame/571.jpg'))
    print(imageocr.work('/home/tuxiaobei/video_to_text/frame/671.jpg'))
    print(imageocr.work('/home/tuxiaobei/video_to_text/frame/771.jpg'))
    print(imageocr.work('/home/tuxiaobei/video_to_text/frame/871.jpg'))
    print(imageocr.work('/home/tuxiaobei/video_to_text/frame/971.jpg'))
    print(imageocr.work('/home/tuxiaobei/video_to_text/frame/1071.jpg'))
