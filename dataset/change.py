
import jieba
import json
d = open("train.json", "r", encoding="utf-8").readlines()
for k in d:
    data = json.loads(k)
    video_id = data['video_id']
    text = open("text/{}.txt".format(video_id), "r", encoding="utf-8").read()
    text = jieba.lcut(text)
    res = {
        "doc_keyword": [],
        "doc_topic": [],
        "doc_label": [data['annotation']],
        "doc_token": text,
    }
    open("data_train.json", "a", encoding="utf-8").write(
        json.dumps(obj=res, ensure_ascii=False) + "\n")
