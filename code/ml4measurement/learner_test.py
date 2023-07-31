import configparser
import eb_ml_utils
import eb_ml_battery_lib

# Data acquition file to load from dateset folder
battery_list= [1,2,3,4]
test_battery_list= [5]


#configuration dictionary
config ={}

# Root working folder (local or Google Drive)
# config['ROOT_DIR'] = get_root_path("batterie")
config['ROOT_DIR'] = "."  

# Folder with dataset in CSV format
#config['DATASETS_DIR'] = config['ROOT_DIR']+"/datasets"
config['DATASETS_DIR'] = "/Users/emanuelebuchicchio/ai4power/datasets/EIS-vs-SOC-2022"

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

model_name="prova_IMG"
print("model name: "+model_name)

# Folder with dataset in CSV format
#config['DATASETS_DIR'] = config['ROOT_DIR']+"/datasets"
config['DATASETS_DIR'] = config['ROOT_DIR']+"/datasets/EIS-vs-SOC-2022"

# List of SoC level into dataset
#config['soc_list']=['100','090','080','070','060','050','040','030','020','010']
config['soc_list']=['100','090','080','070','060','050','040','030','020','010']


# Folder to store trained model
#config['MODELS_DIR'] = config['ROOT_DIR']+"/models"
config['MODELS_DIR'] = config['ROOT_DIR']+"/models"

config['ExperimentName'] = model_name
config['IMAGES_PATH'] = config['ROOT_DIR']+"/"+config['ExperimentName']
config['TEST_IMAGES_PATH'] = config['ROOT_DIR']+"/test_"+config['ExperimentName']

if(generate_images):
    dataset,feature_col_names=eb_ml_battery_lib.load_soc_dataset(battery_list,config["soc_list"],config['DATASETS_DIR'])
    eb_ml_battery_lib.generate_image_files_from_eis(dataset,feature_col_names,config['IMAGES_PATH'],config['ExperimentName'],DATA_AUGMENTATION_FACTOR=10)

    test_dataset,feature_col_names=eb_ml_battery_lib.load_soc_dataset(test_battery_list,config["soc_list"],config['DATASETS_DIR'])
    eb_ml_battery_lib.generate_image_files_from_eis(test_dataset,feature_col_names,config['TEST_IMAGES_PATH'],config['ExperimentName'],DATA_AUGMENTATION_FACTOR=10)

# TRAINING
learn= eb_ml_utils.build_and_train_learner(config,n_epochs=n_epochs)
#SAVE
weights_filename=eb_ml_utils.save_model_weights(learn,config["MODELS_DIR"],model_name)
filename_pth= weights_filename+".pth"



