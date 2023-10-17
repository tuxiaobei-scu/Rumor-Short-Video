import json
import numpy
r = open("test.json", "r", encoding="utf-8").readlines()

data = []
for p in r:
    d = json.loads(p)
    if d['annotation'] == "假":
        data.append(1)
    elif d['annotation'] == "真":
        data.append(0)
    else:
        data.append(2)


def calc_metrics(predict):
    global data
    k = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    # print(k)
    
    for i in range(len(data)):
        k[data[i]][predict[i]] += 1
        
        # if predict[i] == 1 and data[i] != 1:
        #    print(i + 1, data[i])

    Accuracy = (k[0][0] + k[1][1] + k[2][2]) / (k[0][0] + k[0][1] + k[0]
                                                [2] + k[1][0] + k[1][1] + k[1][2] + k[2][0] + k[2][1] + k[2][2])
    if k[0][0] == 0:
        P1 = 0
        R1 = 0
    else:
        P1 = k[0][0] / (k[0][0] + k[1][0] + k[2][0])
        R1 = k[0][0] / (k[0][0] + k[0][1] + k[0][2])
    if k[1][1] == 0:
        P2 = 0
        R2 = 0
    else:
        P2 = k[1][1] / (k[0][1] + k[1][1] + k[2][1])
        R2 = k[1][1] / (k[1][0] + k[1][1] + k[1][2])
    if k[2][2] == 0:
        P3 = 0
        R3 = 0
    else:
        P3 = k[2][2] / (k[0][2] + k[1][2] + k[2][2])
        R3 = k[2][2] / (k[2][0] + k[2][1] + k[2][2])
    Precision = (P1 + P2 + P3) / 3
    Recall = (R1 + R2 + R3) / 3
    F1 = (2 * Precision * Recall) / (Precision + Recall)
    return Accuracy, Precision, Recall, F1


def check_model(model_name):
    # print(model_name)
    predict = []
    r = open("data/" + model_name + ".txt", "r").readlines()
    for p in r:
        p = p.split(' ')
        a = [float(p[0]), float(p[1]), float(p[2])]
        a_array = numpy.array(a)
        predict.append(a_array.argmax())
    return calc_metrics(predict)


video_models = ["c3d", "tsn", "slowfast", "swin"]
audio_models = ["EcapaTdnn", "PANNS_CNN14", "Res2Net", "ResNetSE", "TDNN"]
text_models = ["AttentiveConvNet", "DPCNN", "DRNN", "FastText",
               "TextCNN", "TextRCNN", "TextRNN", "Transformer"]
our_models = ["Our"]
print("Method,Accuracy,Precision,Recall,F1")
for model in video_models:
    Accuracy, Precision, Recall, F1 = check_model(model)
    print("%s,%.5f,%.5f,%.5f,%.5f" %
          (model, Accuracy, Precision, Recall, F1))
