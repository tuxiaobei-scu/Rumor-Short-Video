# encoding:utf-8
import json
import os

last_str = 'last_str'
text = ''


def Levenshtein_Distance(str1, str2):
    """
    计算字符串 str1 和 str2 的编辑距离
    :param str1
    :param str2
    :return:
    """
    matrix = [[i + j for j in range(len(str2) + 1)]
              for i in range(len(str1) + 1)]

    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            if (str1[i-1] == str2[j-1]):
                d = 0
            else:
                d = 1

            matrix[i][j] = min(matrix[i-1][j]+1, matrix[i]
                               [j-1]+1, matrix[i-1][j-1]+d)

    # print('[Levenshtein_Distance]', str1, str2, matrix[len(str1)][len(str2)])
    return matrix[len(str1)][len(str2)]


def add_sentence(s1, sim=0.3):
    global text
    if len(text) <= len(s1):
        dis = Levenshtein_Distance(s1, text)
        if dis <= sim*len(s1):
            return False
    else:
        for i in range(len(text) - len(s1) - 1, 0, -1):
            dis = Levenshtein_Distance(s1, text[i:i+len(s1)])
            if dis <= sim*len(s1):
                return False
    text += s1
    return True


def check_chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


def get_string_from_list(A):
    global last_str
    if type(A) == list:
        ret = ''
        for i in A:
            ret += get_string_from_list(i)
        return ret
    if type(A) == str:
        print('[str]', A)
        if Levenshtein_Distance(A, last_str) < min(len(A), len(last_str))*0.4:
            return ""
        if not check_chinese(A):
            # 去除无汉字的结果
            return ""
        print("-"*100)
        last_str = A
        return A
    return ""


def refine_ocr(ocr_result):
    last_frame_string = 'last_frame_string'
    ret_string = ''
    for frame_result in ocr_result:
        # print(frame_result)
        frame_result_string = get_string_from_list(frame_result['result'])
        print('[sentence]', frame_result_string)
        # print('-'*10)
        try:
            if Levenshtein_Distance(frame_result_string, last_frame_string) > min(len(frame_result_string), len(last_frame_string))*0.4:
                ret_string += frame_result_string
                last_frame_string = frame_result_string
        except:
            pass

    print('='*10)
    print(ret_string)
    return ret_string


def calc(data):
    wav_result = data['wav_result']['sentences']
    ocr_result = data['ocr_result']
    fps = data['fps']
    if len(wav_result) == 0:
        wav_result = [{
            "text": "",
            "start": 0,
            "end": 0
        }]
    if len(ocr_result) == 0:
        ocr_result = [{
            "frame": 0,
            "result": []
        }]
    i = j = 0
    while i < len(wav_result) and j < len(ocr_result):
        if wav_result[i]['start'] < ocr_result[j]['frame']/fps:
            add_sentence(wav_result[i]['text'])
            i += 1
        else:
            k = ocr_result[j]['result']
            for s in k:
                if check_chinese(s[0]):
                    add_sentence(s[0])
            j += 1
    while i < len(wav_result):
        add_sentence(wav_result[i]['text'])
        i += 1
    while j < len(ocr_result):
        k = ocr_result[j]['result']
        for s in k:
            if check_chinese(s[0]):
                add_sentence(s[0])
        j += 1


def work(json_path, save_path):
    files = os.listdir(json_path)
    global text
    i = 50
    for file in files:
        text = ''
        file_path = os.path.join(json_path, file)
        J = None
        with open(file_path, 'r', encoding='utf-8') as f:
            J = json.loads(f.read())
        print(J['video_id'])

        calc(J)
        data_save_path = os.path.join(
            save_path, J['video_id']+'.txt')
        with open(data_save_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(text)
        print('-'*100)
        i -= 1
        if i < 0:
            break


if __name__ == '__main__':
    json_save_path = 'text'
    data_save_path = 'output'
    work(json_save_path, data_save_path)