from simpletransformers.classification import ClassificationModel
import sklearn
import pandas as pd

model = ClassificationModel(
    "roberta", "models/twitter_roberta_base_irony_plus_mustard",
    use_cuda=False
)

eval_df = pd.read_csv('data/val_MUSTARD.csv')
model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)