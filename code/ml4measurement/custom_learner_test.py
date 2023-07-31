import eb_ml_utils
import eb_ml_battery_lib
from "../src/eb_colab_utils.py" import *
# Data acquition file to load from dateset folder
measure_list= ["02_1","02_2","02_3","02_4","02_5","02_6","02_7","02_8","02_9"]
test_measure_list= [5]


#configuration dictionary
config ={}

# Root working folder (local or Google Drive)
# config['ROOT_DIR'] = get_root_path("batterie")
config['ROOT_DIR'] = eb_colab_utils.get_root_path("batterie")  

# Folder with dataset in CSV format
#config['DATASETS_DIR'] = config['ROOT_DIR']+"/datasets"
config['DATASETS_DIR'] = "/Users/emanuelebuchicchio/ml-4-measurement/datasets/EIS-v-SOC-may2022"

# List of SoC level into dataset
#config['soc_list']=[100,90,80,70,60,50,40,30,20,10]
config['soc_list']=[100,90,80,70,60,50,40,30,20,10]

# Folder to store trained model
#config['MODELS_DIR'] = config['ROOT_DIR']+"/models"
config['MODELS_DIR'] = config['ROOT_DIR']+"/models"

# Enable/disable image dataset generation
# Image genaration is time and resource comsuming task. 
# You need to generate image just once for every cross validation experiment
# generate_images = False
generate_images = True

# The number of epochs is a hyperparameter that defines the number times that the 
# learning algorithm will work through the entire training dataset
# n_epochs=50
n_epochs=1

model_accuracy={}

model_name="prova _IMG"
print("model name: "+model_name)

config['ExperimentName'] = model_name
config['IMAGES_PATH'] = "/Users/emanuelebuchicchio/ai4power/IMG_SAVE"
config['TEST_IMAGES_PATH'] = "/Users/emanuelebuchicchio/ai4power/TEST_IMG_SAVE"

#GENERATE IMAGE
if(generate_images):
    dataset,feature_col_names=eb_ml_battery_lib.load_soc_dataset_ec(measure_list,config["soc_list"],config['DATASETS_DIR'])
    eb_ml_battery_lib.generate_image_from_ec(dataset,feature_col_names,config['IMAGES_PATH'],config['ExperimentName'],DATA_AUGMENTATION_FACTOR=10)

    test_dataset,feature_col_names=eb_ml_battery_lib.load_soc_dataset_ec(test_measure_list,config["soc_list"],config['DATASETS_DIR'])
    eb_ml_battery_lib.generate_image_from_ec(test_dataset,feature_col_names,config['IMAGES_PATH'],config['ExperimentName'],DATA_AUGMENTATION_FACTOR=10)



# TRAINING
learn= eb_ml_utils.build_and_train_custom_learner(measure_list,test_measure_list,
data_loader_function=eb_ml_battery_lib.load_soc_dataset_ec,image_generator_function=eb_ml_battery_lib.generate_image_from_ec ,config=config,n_epochs=n_epochs)
#SAVE
weights_filename=eb_ml_utils.save_model_weights(learn,config["MODELS_DIR"],model_name)
filename_pth= weights_filename+".pth"



