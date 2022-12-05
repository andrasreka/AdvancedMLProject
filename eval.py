from simpletransformers.classification import ClassificationModel
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)

model = ClassificationModel(
    "roberta", "models/robertatwitter_mustard",
    use_cuda=False

)

eval_df = pd.read_csv('data/val_MUSTARD.csv')
result, model_outputs, wrong_predictions = model.eval_model(
    eval_df, 
    acc = accuracy_score,
    f1 = f1_score,
    recall = recall_score,
    precision = precision_score 
    )


with open("models/robertatwitter_mustard/eval.json", 'w') as file:
    json.dump(result, file, indent=4, sort_keys=True,
              separators=(', ', ': '), ensure_ascii=False,
              cls=NumpyEncoder)
