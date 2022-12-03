from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging
prefix = 'data/'


logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

train_df = pd.read_csv(prefix + 'train.txt', sep='\t', header=None) 
train_df = train_df[train_df.columns[1:3]]
train_df.columns = ['labels', 'text']
train_df = train_df[['text', 'labels']]

eval_df = pd.read_csv(prefix + 'test.txt', sep='\t', header=None)
eval_df = eval_df[eval_df.columns[1:3]]
eval_df.columns = ['labels', 'text']
eval_df = eval_df[['text', 'labels']]


# Optional model configuration
model_args = ClassificationArgs(num_train_epochs=1)

# Create a ClassificationModel
model = ClassificationModel(
    "roberta", "roberta-base", args=model_args, use_cuda=False
)

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

# Make predictions with the model
predictions, raw_outputs = model.predict(["Sam was a Wizard"])