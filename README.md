# Rumor-Short-Video 基于多模态特征融合的短视频谣言检测系统

> **本项目是论文** [Multimodal Short Video Rumor Detection System Based on Contrastive Learning](https://arxiv.org/abs/2304.08401) 的开源存储库。

## 环境依赖

本项目只能在 Linux 环境下运行，目前已验证的系统为 Ubuntu 20.04。

需要 `Pytorch` 作为机器学习基础环境。

```
pip install torch torchvision torchaudio
```

请前往 [Pytorch 官网](https://pytorch.org/) 浏览更多安装选项。

安装 mmaction2

```
pip install -U openmim
mim install mmengine
mim install mmcv
pip install mmaction2
```

安装 ModelScope

```
pip install modelscope
pip install "modelscope[cv]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
pip install "modelscope[audio]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

## 数据集

数据集请见 dataset 文件夹。

数据格式为 NDJSON（Newline Delimited JSON，换行分割的 JSON）

```json
{"video_id": "douyin_6799592761499700494", "keywords": "16名韩国护士因疫情严重辞职", "annotation": "假"}
{"video_id": "douyin_6989830805723811076", "keywords": "19岁消防员救火时被坍塌的墙体压倒而牺牲", "annotation": "真"}
{"video_id": "douyin_6732370114181598477", "keywords": "2019环邢台国际自行车赛连环大撞车", "annotation": "真"}
```

原始短视频请从百度网盘获取。链接：https://pan.baidu.com/s/1PH2vSfGxq2aHTlb0KUj7CA?pwd=0803

## 多模态文字提取

见 `code/video_to_text` 文件夹

首先请将视频放置在 `videos` 文件夹下

执行 `python calc.py` 生成待处理文件列表。

执行

```
python3 main.py videos-0.txt > v0.txt
```

即可自动开启文字提取任务。

将会以 `json` 格式输出提取结果。提取结果样例：

```json
{
    "video_id": "douyin_6559701594739313923",
    "wav_result": {
        "text": "喜欢你看我时候一脸的乖巧，曾经的亲吻我的睫毛。喜欢你，每次给我温柔的依靠特边，所有我的骄傲心跳，不何与雨恋和你问题在讲，我就能感觉到的时候，你哼出的音调是恋爱的预兆。",
        "text_postprocessed": "喜 欢 你 看 我 时 候 一 脸 的 乖 巧 曾 经 的 亲 吻 我 的 睫 毛 喜 欢 你 每 次 给 我 温 柔 的 依 靠 特 边 所 有 我 的 骄 傲 心 跳 不 何 与 雨 恋 和 你 问 题 在 讲 我 就 能 感 觉 到 的 时 候 你 哼 出 的 音 调 是 恋 爱 的 预 兆",
        "time_stamp": [
            [
                1750,
                1989
            ],
            [
                2009,
                2250
            ],
            [
                2370,
                2610
            ],
            // 省略
        ],
        "sentences": [
            {
                "text": "喜欢你看我时候一脸的乖巧,",
                "start": 1750,
                "end": 5635
            },
			// 省略
        ]
    },
    "ocr_result": [
        {
            "frame": 0,
            "result": [
                [
                    "2018.5.25"
                ],
                [
                    "TOP7"
                ],
                [
                    ""
                ],
                [
                    "周"
                ],
                [
                    "本店"
                ],
                [
                    "言"
                ],
                [
                    "By:江苏网警"
                ]
            ]
        },
        // 省略
    ],
    "fps": 30.0
}
```



## 多模态文字融合

见路径 `code/text-fusion`。

将文字数据 `json` 放置在 `text` 目录下，执行。

```
python calc.py
```

即可自动融合文字信息。

## 视频特征提取

见 `code/video_feature` 路径wget

下载模型文件

```
wget -O checkpoint.pth https://download.openmmlab.com/mmaction/v1.0/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb_20220906-cd10898e.pth
```

将原始视频连接至 `data/videos`

执行

```python
python calc.py
```

## 对比学习训练

在原始数据集中，我们有很多样本，编号为 0，1，2（非谣言，谣言，辟谣）。为了提高我们在向量数据库中检索的准确性，我们需要能很好地通过向量区分出这些样本。我们随机将样本分为两两一对，作为一个数据训练，其标签为0=两个样本标签不同，1=两个样本标签不同。在这种情况下，我们可以使用对比性损失： 标签为1的相似对被拉到一起，，因此它们在向量空间中是接近的。不相似的对，如果比定义的边际更接近，则在向量空间中被推开。

也就是我们实现了相同标签的样本在向量空间中尽可能聚集，同时与不同标签的样本相隔的距离尽可能远。

我们使用cosine_distance（也就是1-cosine_similarity）作为我们基础性的对比损失函数，边距为0.5。也就是说，不相同标签的样本应该有至少0.5的余弦距离（相当于0.5的余弦相似度差异）。

对比性损失的改进版本是OnlineContrastiveLoss，它寻找哪些消极对的距离低于最大的积极对，哪些积极对的距离高于消极对的最低距离。也就是说，这种损失会自动检测一批中的困难案例，并只对这些案例计算损失。

执行

```
python main.py
```

进行标签的对比性训练

执行

```
python main_find.py
```

进行可解释性（视频检索的对比性训练）

## 模型评估

见 `code/val` 目录。

对比模型简介：

(1)视频分类模型

- C3D：C3D模型是一种使用3D卷积网络学习时空特征的模型，它可以对视频序列进行特征提取和分类。C3D模型的主要思想是将3D卷积和3D池化应用于视频帧，从而同时捕获空间和时间维度的信息。C3D模型的结构类似于VGG网络，但是将所有的2D卷积和2D池化替换为3D卷积和3D池化。C3D模型具有高效、简单、紧凑的特点，可以在多个视频理解任务上取得较好的效果。
- TSN：TSN模型是一种用于视频分类的网络模型，它基于2D-CNN，通过稀疏采样视频帧的方式来提取视频的全局信息和局部信息。TSN模型包含两个分支：空间分支和时间分支，空间分支用于提取每帧图像的静态特征，时间分支用于提取相邻帧之间的动态特征。两个分支的特征通过后期融合的方式来进行分类或检测。TSN模型是一种简单而有效的视频理解方法，可以适应不同长度和复杂度的视频。TSN模型是一种利用稀疏采样和双分支结构的视频CNN模型，可以同时学习空间和时间特征。
- Slowfast：Slowfast模型是一种用于视频理解的双模CNN模型，它由两个分支组成：Slow分支和Fast分支。Slow分支使用较少的帧数和较大的通道数来学习空间语义信息，Fast分支使用较多的帧数和较少的通道数来学习运动信息。两个分支通过侧向连接来融合特征，最后进行分类或检测。Slowfast模型受到了灵长类动物视网膜神经细胞种类的启发，可以有效地利用不同速度的视频信息，提高了视频识别的性能。Slowfast模型是一种利用双速率分治策略的视频CNN模型，可以同时学习空间和运动特征。
- VideoSwin：VideoSwin模型是一种基于Swin Transformer的视频分类模型，它利用了Swin Transformer的多尺度建模和高效局部注意力特性，可以在视频Transformer中引入局部性的归纳偏置，提高了速度-准确度的权衡。VideoSwin模型在动作识别和时序建模等视频理解任务上取得了最先进的性能，超过了其他的Transformer模型，例如ViViT和TimeSformer等。VideoSwin模型是一种利用局部注意力机制的视频Transformer模型，可以有效地处理视频序列数据。

(2)音频分类模型

- TDNN：即时延神经网络，是一种用于处理序列数据的神经网络，可以包含多帧的特征输入，通过一维卷积和空洞卷积来提取时序信息。TDNN模型最初用于语音识别问题，可以构建一个时间不变的声音模型，适应不同长度和方言的语音。
- ECAPA-TDNN：该模型是一种用于说话人识别的网络模型，它基于TDNN（Time Delay Neural Network），通过引入SE（Squeeze-Excitation）模块、多层特征融合和通道注意力统计池化等技巧来提高说话人特征的表达能力。ECAPA-TDNN模型在多个说话人验证任务上取得了优异的性能，超越了传统的x-vector和ResNet等模型。
- Res2Net：该模型是一种基于ResNet的卷积神经网络模型，它通过在残差块内部构造分层的类残差连接，来增强多尺度特征的表示能力和感受野范围。Res2Net模型将输入特征图分成多个子集，每个子集经过一个3x3卷积，并与前一个子集的输出相加，形成一个分层的连接结构。Res2Net模块可以提取不同尺度的时频特征，并增加每个网络层的感受野范围，从而提高音频分类的准确率和鲁棒性。Res2Net模型用于音频分类的一些应用场景包括说话人识别，语音情感识别，声纹识别等。
- ResNetSE：该模型是一种在ResNet模型的基础上引入了SE模块（Squeeze-and-Excitation module）的卷积神经网络模型，它通过对每个残差块的输出特征图进行通道间的自适应权重调整，来增强特征的表达能力和选择性。SE模块包括两个步骤：squeeze和excitation。squeeze步骤是对每个特征图进行全局平均池化，得到一个通道维度的向量，表示每个通道的全局信息。excitation步骤是对这个向量进行两层全连接操作，得到一个通道维度的权重向量，表示每个通道的重要性。然后将这个权重向量乘以原始的特征图，得到加权后的特征图。ResNetSE模型可以在不增加计算量和参数量太多的情况下，可以有效地捕获音频信号的通道间关系，增强音频特征的表达能力和选择性。
- PANNS_CNN14：该模型是一种用于音频模式识别的预训练音频神经网络，它在大规模的AudioSet数据集上进行了训练。它使用对数-梅尔谱图作为输入特征，并由14层卷积层和3层全连接层组成。

(3)文本分类模型

- AttentiveConvNet：该模型是一种在卷积神经网络中引入注意力机制的模型，它可以动态地调整卷积操作的视野，使得每个单词或短语的特征表示不仅包含局部上下文信息，还包含全局或跨文本的信息。AttentiveConvNet模型可以应用于自然语言理解的任务，如文本分类、文本蕴含、答案选择等。AttentiveConvNet模型有两个版本：light版本和advanced版本。light版本是基于标准的卷积层和池化层，通过在每个卷积窗口内计算注意力权重来加权求和得到特征表示。advanced版本是在light版本的基础上增加了残差连接和多头注意力机制，以提高特征的多样性和表达能力。
- DPCNN：该模型是一种利用深度金字塔结构和注意力机制进行文本特征提取和分类的模型。它可以通过堆叠卷积层和池化层来提取文本的长距离依赖关系和抽象特征。使用region embedding层将word embedding转换为能够覆盖多个词的特征表示。使用等长卷积层来增强每个词或短语的特征表示，使其包含更广的上下文信息。1/2池化层来降低文本的长度，扩大卷积操作的有效覆盖范围，捕获更多的全局信息。使用固定数量的特征图来减少计算复杂度，保持语义空间的一致性。使用残差连接来支持训练深层网络，解决梯度消失和网络退化的问题。
- DRNN：该模型是一种深层循环神经网络，它可以通过增加隐藏层的层数来增强模型的表达能力和学习长距离依赖的能力12。DRNN模型的主要特点有：每个时刻上的循环体重复多次，形成一个深层的循环结构。每一层循环体中参数是共享的，但不同层之间的参数可以不同。循环体可以是任何类型的递归神经单元，如RNN，LSTM，GRU等。
- FastText：该模型是一种Facebook AI Research在2016年开源的文本分类和词向量训练的工具，它的特点是模型简洁，训练速度快，文本分类准确率也令人满意12。使用所有的单词和相应的n-gram特征作为输入，可以捕捉到词序信息。简单的平均词向量作为文本的表示，可以减少计算复杂度。使用层次softmax作为输出层，可以加速训练过程，适用于大规模的类别数。用预训练的词向量作为输入层，可以提高文本分类的效果。
- TextCNN：该模型是一种用于文本分类的卷积神经网络，它的基本思想是利用不同大小的卷积核来提取句子中的局部特征，然后通过最大池化层和全连接层得到最终的分类结果。可以有效地提取句子中的局部特征，捕捉词序和语义信息。可以使用预训练的词向量，增强模型的泛化能力。
- TextRNN：该模型是一种用于文本分类的循环神经网络（RNN），它的基本思想是利用RNN将句子中的每个词的词向量编码成一个固定长度的向量，然后将该向量输入到一个全连接层得到分类结果。可以有效地处理序列结构，考虑到句子的上下文信息。
- TextRCNN：该模型是一种结合了循环神经网络（RNN）和卷积神经网络（CNN）的文本分类模型，它的基本思想是利用双向RNN来获取句子的上下文信息，然后将词向量和上下文向量拼接起来，通过一个卷积层和一个最大池化层提取特征，最后通过一个全连接层得到分类结果。可以有效地利用双向RNN捕捉句子中的上下文信息，考虑词序和语义信息。可以通过最大池化层自动判断哪些特征在文本分类过程中更重要，减少噪声和冗余信息。可以使用预训练的词向量，增强模型的泛化能力。
- Transformer：该模型是一种基于注意力机制的深度学习模型，它主要用于自然语言处理领域，如机器翻译、文本摘要等。Transformer模型的核心思想是“注意力即是一切”，它摒弃了传统的循环神经网络（RNN）或卷积神经网络（CNN）结构，而是完全使用自注意力机制来捕捉序列中的依赖关系。Transformer模型由编码器（Encoder）和解码器（Decoder）两部分组成，每部分都包含6个子层。编码器负责将输入序列（如源语言句子）编码成一个高维的向量表示，解码器负责根据编码器的输出和已生成的目标序列（如目标语言句子）来生成下一个词。

模型对比结果：

| 模型             | 使用模态                           | 准确率     | 精确率     | 召回率     | F1 值      | AUC 面积   |
| ---------------- | ---------------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| C3D              | 图像                               | 0.5817     | 0.5775     | 0.5767     | 0.5771     | 0.7361     |
| TSN              | 图像                               | 0.6495     | 0.6542     | 0.6428     | 0.6484     | 0.8090     |
| Slowfast         | 图像                               | 0.6991     | 0.6986     | 0.6942     | 0.6964     | 0.8459     |
| VideoSwin        | 图像                               | 0.7156     | 0.7156     | 0.7147     | 0.7151     | 0.8564     |
| TDNN             | 音频                               | 0.4092     | 0.3012     | 0.3810     | 0.3364     | 0.6443     |
| ECAPA-TDNN       | 音频                               | 0.4624     | 0.4574     | 0.4644     | 0.4609     | 0.6961     |
| Res2Net          | 音频                               | 0.4587     | 0.4672     | 0.4572     | 0.4622     | 0.6427     |
| ResNetSE         | 音频                               | 0.4514     | 0.4259     | 0.4408     | 0.4332     | 0.6395     |
| PANNS_CNN14      | 音频                               | 0.4092     | 0.2765     | 0.4097     | 0.3302     | 0.6282     |
| AttentiveConvNet | 图像音频融合文本                   | 0.6587     | 0.7262     | 0.6403     | 0.6806     | 0.8894     |
| DPCNN            | 图像音频融合文本                   | 0.7872     | 0.7874     | 0.7837     | 0.7855     | 0.8841     |
| DRNN             | 图像音频融合文本                   | 0.7523     | 0.7530     | 0.7495     | 0.7512     | 0.8698     |
| FastText         | 图像音频融合文本                   | 0.8184     | 0.8161     | 0.8119     | 0.8140     | 0.9140     |
| TextCNN          | 图像音频融合文本                   | 0.8367     | 0.8391     | 0.8349     | 0.8370     | 0.9280     |
| TextRCNN         | 图像音频融合文本                   | 0.8275     | 0.8253     | 0.8247     | 0.8250     | 0.9232     |
| TextRNN          | 图像音频融合文本                   | 0.8367     | 0.8321     | 0.8316     | 0.8318     | 0.9185     |
| Transformer      | 图像音频融合文本                   | 0.8018     | 0.8066     | 0.8013     | 0.8040     | 0.9079     |
| **Our Model**    | **图像+图像音频融合文本+外部知识** | **0.8771** | **0.8737** | **0.8739** | **0.8738** | **0.9507** |

## 许可协议

本项目采用 [MIT](https://github.com/tuxiaobei233/Rumor-Short-Video/blob/main/LICENSE) 许可协议开源，在使用本项目的源代码时请遵守许可协议。
