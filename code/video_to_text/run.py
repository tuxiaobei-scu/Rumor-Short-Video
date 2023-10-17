import os
import threading
import time


def td():
    os.system("python3 main.py videos-0.txt > v0.txt")


while True:
    thread = threading.Thread(target=td)
    thread.start()
    thread.join()
