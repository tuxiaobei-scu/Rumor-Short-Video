from PIL import Image
import imagehash


class imghash:

    img_list = []

    def work(this, img, k):
        res = imagehash.phash(img)
        for i in range(len(this.img_list)-1, -1, -1):
            p = this.img_list[i]
            if p - res < k:
                return False
        this.img_list.append(res)
        if len(this.img_list) > 100:
            this.img_list.pop(0)
        return True


if __name__ == '__main__':
    a = imghash()
    print(int(str(a.work('/home/tuxiaobei/video_to_text/frame/0.jpg'))))
    print(a.work('/home/tuxiaobei/video_to_text/frame/5.jpg'))
    print(a.work('/home/tuxiaobei/video_to_text/frame/10.jpg'))
    print(a.work('/home/tuxiaobei/video_to_text/frame/15.jpg'))
    print(a.work('/home/tuxiaobei/video_to_text/frame/20.jpg'))

    print("-"*10)

    print(a.work('/home/tuxiaobei/video_to_text/frame/12.jpg'))
    print(a.work('/home/tuxiaobei/video_to_text/frame/13.jpg'))
