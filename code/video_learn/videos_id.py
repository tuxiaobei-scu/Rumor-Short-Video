import json
import os
d = open("datasets/data.json", "r").readlines()
for p in d:
    p = json.loads(p)
    if os.path.exists("datasets/video_feature/douyin_" + p['video_id'] + ".csv"):
        open("datasets/data_video.json",
             "a").write(json.dumps(p, ensure_ascii=False) + "\n")
