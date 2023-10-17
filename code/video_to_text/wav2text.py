from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


class wav2text:
    p = pipeline('auto-speech-recognition',
                 'damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch', device="cpu")

    def work(this, wav_path):
        return this.p(wav_path,)


if __name__ == "__main__":
    a = wav2text()
    r = a.work('/home/tuxiaobei/video_to_text/test/a.wav')
    print(r)
