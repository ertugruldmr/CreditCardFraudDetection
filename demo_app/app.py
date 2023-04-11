import pickle
import json 
import gradio as gr
import numpy as np
import pandas as pd
import sklearn
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE


# File Paths
model_path = 'log_reg_os.sav'
component_config_path = "component_configs.json"
examples_path = "examples.txt"

# predefined
feature_order = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
                 'scaled_amount', 'scaled_time']

# Loading the files
model = pickle.load(open(model_path, 'rb'))
examples = pickle.load(open(examples_path, 'rb'))

labels = ["not Fraud", "Fraud"]#classes[target].values()

feature_limitations = json.load(open(component_config_path, "r"))


# Util function
def predict(*args):

  # preparing the input into convenient form
  features = pd.Series([*args], index=feature_order)
  #features = feature_encode(features)
  features = np.array(features).reshape(-1,len(feature_order))

  # prediction
  #model = os_loj_pure
  probabilities = model.predict_proba(features) #.predict(features)
  probs = probabilities.flatten()

  # output form
  results = {l : np.round(p, 3) for l, p in zip(labels, probs)}

  return results


inputs = list()
for col in feature_order:
  if col in feature_limitations["cat"].keys():
    
    # extracting the params
    vals = feature_limitations["cat"][col]["values"]
    def_val = feature_limitations["cat"][col]["def"]
    
    # creating the component
    inputs.append(gr.inputs.Dropdown(vals, default=def_val, label=col))
  else:
    
    # extracting the params
    min = feature_limitations["num"][col]["min"]
    max = feature_limitations["num"][col]["max"]
    def_val = feature_limitations["num"][col]["def"]
    
    # creating the component
    inputs.append(gr.inputs.Slider(minimum=min, maximum=max, default=def_val, label=col) )


# creating the app
demo_app = gr.Interface(predict, inputs, "label", examples=examples)

# Launching the demo
if __name__ == "__main__":
    demo_app.launch()