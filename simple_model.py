from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import logging

prefix = 'data/'
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

# train_df = pd.read_csv(prefix + 'train.txt', sep='\t', header=None) 
# train_df = train_df[train_df.columns[1:3]]
# train_df.columns = ['labels', 'text']
# train_df = train_df[['text', 'labels']]

# val_df = pd.read_csv(prefix + 'test.txt', sep='\t', header=None)
# val_df = val_df[val_df.columns[1:3]]
# val_df.columns = ['labels', 'text']
# val_df = val_df[['text', 'labels']]

train_df = pd.read_csv('data/train_MUSTARD.csv')  
val_df = pd.read_csv('data/val_MUSTARD.csv')

model_args = {
    'data_dir': 'data/',
    'output_dir': 'outputs/',
    'cache_dir': 'cache/',
    'do_train': True,
    'do_eval': True,
    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 512,
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
    'save_steps': 1000,
    'eval_all_checkpoints': True,
    'overwrite_output_dir': False,
    'reprocess_input_data': True,
}


train_df = train_df.head(5)
val_df = val_df.head(5)
# Create a ClassificationModel
model = ClassificationModel(
    "roberta", "cardiffnlp/twitter-roberta-base-irony", args=model_args, use_cuda=False
)
# model = ClassificationModel(
#     "roberta", "jkhan447/sarcasm-detection-RoBerta-base", args=model_args, use_cuda=False
# )

# model = ClassificationModel(
#     "distilbert", "xlnet/sarcasm-detection-xlnet-base-cased", args=model_args, use_cuda=False, from_tf = True
# )


# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(val_df)