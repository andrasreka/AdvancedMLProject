
from simpletransformers.classification import ClassificationModel
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


comments = [
    "This is a lovely book. I would recommend it to anyone",
    "This book is just a list of numbers. Useless.",
    "OMG, what a suspenseful read",
    "More interesting than the Bible",
    "This is, without a doubt, a more touching story than twilight",
    "A great read. Captivating. I couldn’t put it down anymore, when I have found out that 0.629 is there",
]

model = ClassificationModel(
    "roberta", "outputs_reddit_based_on_twitter/checkpoint-24000", use_cuda=False
)
pred, _ = model.predict(comments)
print(pred)
