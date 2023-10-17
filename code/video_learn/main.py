"""
This scripts demonstrates how to train a sentence embedding model for question pair classification
with cosine-similarity and a simple threshold.
As dataset, we use Quora Duplicates Questions, where we have labeled pairs of  questions beeing either duplicates (label 1) or non-duplicate (label 0).
As loss function, we use OnlineConstrativeLoss. It reduces the distance between positive pairs, i.e., it pulls the embeddings of positive pairs closer together. For negative pairs, it pushes them further apart.
An issue with constrative loss is, that it might push sentences away that are already well positioned in vector space.
"""

import json
from torch.utils.data import DataLoader
from sentence_transformers import losses, util
from sentence_transformers import LoggingHandler, SentenceTransformer, evaluation
from sentence_transformers.readers import InputExample
import logging
from transformers import AutoModel
from datetime import datetime
import csv
import os
from zipfile import ZipFile
import random

# Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)
# /print debug information to stdout


# As base model, we use DistilBERT-base that was pre-trained on NLI and STSb data
model = SentenceTransformer('output_model/best2', is_video=True)
num_epochs = 30
train_batch_size = 8

# As distance metric, we use cosine distance (cosine_distance = 1-cosine_similarity)
distance_metric = losses.SiameseDistanceMetric.COSINE_DISTANCE

# Negative pairs should have a distance of at least 0.5
margin = 0.5

dataset_path = 'quora-IR-dataset'
model_save_path = 'output/MSVRS-' + \
    datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

os.makedirs(model_save_path, exist_ok=True)
"""
# Check if the dataset exists. If not, download and extract
if not os.path.exists(dataset_path):
    logger.info("Dataset not found. Download")
    zip_save_path = 'quora-IR-dataset.zip'
    util.http_get(
        url='https://sbert.net/datasets/quora-IR-dataset.zip', path=zip_save_path)
    with ZipFile(zip_save_path, 'r') as zip:
        zip.extractall(dataset_path)


######### Read train data  ##########
# Read train data
train_samples = []
i = 500
with open(os.path.join(dataset_path, "classification/train_pairs.tsv"), encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        sample = InputExample(
            texts=[row['question1'], row['question2']], label=int(row['is_duplicate']))
        train_samples.append(sample)
        i -= 1
        if i == 0:
            break
"""

train_videos = []
for video in open("datasets/train.json", "r", encoding="utf-8").readlines():
    train_videos.append(json.loads(video.strip()))
train_siz = 16


evaluators = []

ir_queries = {}  # Our queries (qid => question)
ir_corpus = {}  # Our corpus (qid => question)

r = open("datasets/test.json", "r", encoding="utf-8").readlines()
for p in r:
    d = json.loads(p)
    if d['annotation'] == "假":
        ir_queries[d['video_id']] = 1
    elif d['annotation'] == "真":
        ir_queries[d['video_id']] = 0
    else:
        ir_queries[d['video_id']] = 2

r = open("datasets/train.json", "r", encoding="utf-8").readlines()
for p in r:
    d = json.loads(p)
    if d['annotation'] == "假":
        ir_corpus[d['video_id']] = 1
    elif d['annotation'] == "真":
        ir_corpus[d['video_id']] = 0
    else:
        ir_corpus[d['video_id']] = 2
ir_evaluator = evaluation.RumourChecker(
    ir_queries, ir_corpus)

evaluators.append(ir_evaluator)

seq_evaluator = evaluation.SequentialEvaluator(
    evaluators, main_score_function=lambda scores: scores[-1])

logger.info("Evaluate model without training")
seq_evaluator(model, epoch=0, steps=0, output_path=model_save_path)


train_loss = losses.OnlineContrastiveLoss(
    model=model, distance_metric=distance_metric, margin=margin)
# train_loss = losses.CosineSimilarityLoss(model=model)
# Train the model

for i in range(num_epochs):
    model.save("output_model/50-epoch-%d" % i)
    logger.info("Epoch: {}".format(i))
    random.shuffle(train_videos)
    video_batch = [train_videos[i:i+train_siz]
                   for i in range(0, len(train_videos), train_siz)]
    train_samples = []

    for batch in video_batch:
        for i in range(len(batch) - 1):
            for j in range(i + 1, len(batch)):
                if batch[i]["annotation"] == batch[j]["annotation"]:
                    label = 1
                else:
                    label = 0
                sample = InputExample(
                    texts=[batch[i]["video_id"], batch[j]["video_id"]], label=label)
                train_samples.append(sample)
    train_dataloader = DataLoader(
        train_samples, shuffle=True, batch_size=train_batch_size)
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=seq_evaluator,
              epochs=1,
              warmup_steps=0,
              output_path=model_save_path,
              use_amp=True,
              scheduler='warmupcosine'
              )
