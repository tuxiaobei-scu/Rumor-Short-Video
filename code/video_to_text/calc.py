import threading
import os
video_path = "./videos"
files = os.listdir(video_path)
thread_num = 1
i = 0
for i in range(thread_num):
    open("videos-%d.txt" % (i), "w").write('')
for file in files:
    if file.endswith('.mp4'):
        open("videos-%d.txt" % (i % thread_num), "a").write(file + '\n')
        i += 1
