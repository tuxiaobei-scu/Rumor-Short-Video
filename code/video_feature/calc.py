from mmaction.apis import inference_recognizer, init_recognizer

config_path = 'tsn.py'
checkpoint_path = 'checkpoint.pth'  # can be a local path
# you can specify your own picture path
model = init_recognizer(config_path, checkpoint_path,
                        device="cuda:0")  # device can be 'cuda:0'

p = open('data/test.txt', 'r').readlines()
f = open('vector/tsn.txt', 'w')
for video in p:
    video_id = video.split(' ')[0]
    print(video_id)
    result = inference_recognizer(model, 'data/videos/' + video_id)

