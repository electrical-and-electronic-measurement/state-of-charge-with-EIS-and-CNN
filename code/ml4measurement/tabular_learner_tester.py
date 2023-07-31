# Fastai is required by eb_ml_battery_lib
#!pip install fastai==2.5.3 -q

# import
import numpy as np
import pandas as pd

import sys
from fastai.tabular import *
from fastai.tabular.all import *

from eb_ml_colab_utils import get_root_path
from eb_ml_battery_lib import load_soc_dataset,get_EIS_tabular_dataset_polar
from eb_ml_utils import build_tabular_learner

# %%
#configuration dictionary
config ={}

# Root working folder (local or Google Drive)
# config['ROOT_DIR'] = get_root_path("batterie")
config['ROOT_DIR'] = get_root_path("..")  

# Folder with dataset in CSV format
#config['DATASETS_DIR'] = config['ROOT_DIR']+"/datasets"
config['DATASETS_DIR'] = config['ROOT_DIR']+"/datasets/EIS-vs-SOC-May2022"

# List of SoC level into dataset
#config['soc_list']=['100','090','080','070','060','050','040','030','020','010']
config['soc_list']=['100','090','080','070','060','050','040','030','020','010']

# Data acquisition files to load from dateset folder
battery_list=["02_4","02_5","02_6","02_7","02_8","02_9","03_4","03_5","03_6","03_7","03_8","03_9"]
dataset,feature_col_names=load_soc_dataset(battery_list,config["soc_list"],config['DATASETS_DIR'])
dataset.reset_index(drop=True,inplace=True)

dataset_polar,feature_col_names_polar=get_EIS_tabular_dataset_polar(dataset,feature_col_names)
splits = RandomSplitter(valid_pct=0.2)(range_of(dataset_polar))

#Path / default location for saving/loading models

model_path='../models'

#The dependent variable/target
dep_var = 'SOC_float'

#The list of categorical features in the dataset
#cat_names = ['BATTERY_ID', 'EIS_ID'] 
cat_names = []

#The list of continuous features in the dataset
#Exclude the Dependent variable 'Price'
cont_names =feature_col_names_polar


#List of Processes/transforms to be applied to the dataset
procs = [FillMissing, Categorify, Normalize]

learn = build_tabular_learner(dataset_polar,splits,model_path,dep_var,cat_names,cont_names)

lr_value=learn.lr_find()
print("LR value:  ", str(lr_value))

learn.fit_one_cycle(250,lr_value)
learn.show_results()

from sklearn.metrics import mean_squared_error
[y_pred,y_true] =learn.get_preds()
mean_squared_error(y_true,y_pred,squared=False)


learn.model

learn.summary()


