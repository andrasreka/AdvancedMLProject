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
# # Preparing train data
# train_data = [
#     ["Aragorn was the heir of Isildur", 1],
#     ["Frodo was the heir of Isildur", 0],
# ]
# train_df = pd.DataFrame(train_data)
# train_df.columns = ["text", "labels"]

# # Preparing eval data
# eval_data = [
#     ["Theoden was the king of Rohan", 1],
#     ["Merry was the king of Rohan", 0],
# ]
# eval_df = pd.DataFrame(eval_data)
# eval_df.columns = ["text", "labels"]


# Optional model configuration
model_args = {
    'data_dir': 'data/',
    'model_type':  'distilroberta',
    'model_name': 'distilroberta-base',
    'output_dir': 'outputs2/',
    'cache_dir': 'cache/',
    'do_train': True,
    'do_eval': True,
    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 128,
    'output_mode': 'classification',
    'train_batch_size': 12,
    'eval_batch_size': 12,

    'gradient_accumulation_steps': 1,
    'num_train_epochs': 1,
    'weight_decay': 0,
    'learning_rate': 4e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.06,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,

    'logging_steps': 50,
    'evaluate_during_training': False,
    'save_steps': 2000,
    'eval_all_checkpoints': True,

    'overwrite_output_dir': False,
    'reprocess_input_data': True,
    'notes': 'Using twitter dataset'
}

# Create a ClassificationModel
model = ClassificationModel(
    "roberta", "roberta-base", args=model_args, use_cuda=False
)

# Train the model
model.train_model(train_df, show_running_loss=True)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

# Make predictions with the model
predictions, raw_outputs = model.predict(["Sam was a Wizard"])