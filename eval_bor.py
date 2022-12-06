from simpletransformers.classification import ClassificationModel
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd

model = ClassificationModel(
    "roberta", "sarcasm", # "models/twitter_roberta_base_irony_mustard",
    use_cuda=False
)

# eval_df = pd.read_csv('data/val_MUSTARD.csv')
eval_df = pd.read_csv('data/reddit_val.csv').head(5000)
result, model_outputs, wrong_predictions = model.eval_model(
    eval_df, 
    acc = accuracy_score,
    f1 = f1_score,
    recall = recall_score,
    precision = precision_score 
    )

print(result)
