import json
r = open("test.json", "r", encoding="utf-8").readlines()

data = []
for p in r:
    d = json.loads(p)
    if d['annotation'] == "假":
        data.append(1)
    else:
        data.append(0)





def ROC(data, thresholds):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(data)):
        if data[i] == 1:
            if predict[i] > thresholds:
                TP += 1
            else:
                FN += 1
        else:
            if predict[i] > thresholds:
                FP += 1
            else:
                TN += 1
    FPR = FP / (FP + TN)
    TPR = TP / (TP + FN)
    return FPR, TPR


video_models = ["c3d", "tsn", "slowfast", "swin"]
audio_models = ["EcapaTdnn", "PANNS_CNN14", "Res2Net", "ResNetSE", "TDNN"]
text_models = ["AttentiveConvNet", "DPCNN", "DRNN", "FastText",
               "TextCNN", "TextRCNN", "TextRNN", "Transformer"]
our_models = ["Our"]

for model_name in our_models:
    predict = []
    r = open("data/" + model_name + ".txt", "r").readlines()
    for p in r:
        k = p.split(" ")
        a = float(k[0])
        b = float(k[1])
        c = float(k[2])
        predict.append(b / (a + b + c))
    thresholds = 0
    k = open("roc-audio/" + model_name + "_roc.csv", "w")
    k.write("FPR,TPR,thresholds\n")
    auc = 0
    FPR_last = 1
    TPR_last = 1
    while thresholds <= 1:
        FPR, TPR = ROC(data, thresholds)
        auc += (FPR_last - FPR) * TPR_last
        FPR_last = FPR
        TPR_last = TPR
        k.write(str(FPR) + "," + str(TPR) + "," + str(thresholds) + "\n")
        thresholds += 0.001
    print(model_name + ',' + str(auc))
