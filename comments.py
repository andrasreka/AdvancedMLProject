
from simpletransformers.classification import ClassificationModel
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


comments = [
    "This is a lovely book. I would recommend it to anyone",
    "This book is borring. Just a bunch of numbers",
    "This is, without a doubt, a better story than twilight",
    "A great read. Captivating. I couldn’t put it down anymore, when I have found out that 0.629 is there"
]

model = ClassificationModel(
    "roberta", "/outputs_reddit_based_on_twitter/checkpoint-24000", use_cuda=False
)
pred, _ = model.predict(comments)
print(pred)