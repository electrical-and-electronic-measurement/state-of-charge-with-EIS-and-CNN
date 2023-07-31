from audioop import minmax
from cmath import exp
import cmath
from re import split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastai.vision.all import *
from sklearn.preprocessing import MinMaxScaler

from ml4measurement.eb_ml_utils import get_items_func,rescale_dataset,plottingfunction,build_tabular_learner,get_image_files_filtered

from LiBEIS.code.utilities import get_patterns,get_xy_values,augment_meas_data


# CONFIG PAREMETERS
CSV_FILE_PREFIX='/EIS_BATT'

#DEFAULT OFFSET FOR DATA AUGMENTATION CSV FILE NAME GENERATION 
AUGMENTATION_OFFSET = 1000
DATA_AUGMENTATION_FACTOR_OFFSET = 100

def build_and_train_battery_learner_from_EIS(battery_list,test_measure_list,config,n_epochs=50,generate_training_images=False,generate_test_images=False,rescale=False):
  """ Build and train a FastAI learner for battery SoC classification task """
  learn= build_battery_soc_model_learner(battery_list,test_measure_list,config,generate_training_images,generate_test_images)
  lr_obj = learn.lr_find()
  print(f"Valley: {lr_obj.valley:.2e}")
  learn.fine_tune(n_epochs,lr_obj.valley)
  return learn

def build_and_train_battery_learner_from_EC(battery_list,test_measure_list,config,n_epochs=50,generate_training_images=False,generate_test_images=False,rescale=True):
  """ Build and train a FastAI learner for battery SoC classification task """
  learn= build_battery_soc_model_learner_ec(battery_list,test_measure_list,config,generate_training_images,generate_test_images,rescale)
  lr_obj = learn.lr_find()
  print(f"Valley: {lr_obj.valley:.2e}")
  learn.fine_tune(n_epochs,lr_obj.valley)
  return learn


def build_battery_soc_model_learner(measure_list,test_battery_list,config,
generate_training_images=False,generate_test_images=False):
  """ Train a battery SOC classifier model """
  #Train - Validation
  dataset,eis_col_names=load_soc_dataset(measure_list,config["soc_list"],config['DATASETS_DIR'])
  if(generate_training_images):
    generate_image_files_from_eis(dataset,eis_col_names,config['IMAGES_PATH'],config['ExperimentName'],DATA_AUGMENTATION_FACTOR=10)

  #Test dataset
  test_dataset,eis_col_names=load_soc_dataset(test_battery_list,config["soc_list"],config['DATASETS_DIR'])
  if(generate_test_images):
    generate_image_files_from_eis(test_dataset,eis_col_names,config['TEST_IMAGES_PATH'],config['ExperimentName'],DATA_AUGMENTATION_FACTOR=1)
  
  splitter = config['Splitter'] # RandomSplitter(valid_pct=0.3, seed=41) RandomSplitter(valid_pct=0.3, seed=41)

  #FastAI image pipeline
  item_tfms = [Resize(224)]
  batch_tfms=[Normalize.from_stats(*imagenet_stats)]
  rePat=config['rePat'] #r'^.*_(\d+).png$'

  #Build FastAI DataBlock
  dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   get_items=get_items_func,
                   get_y=RegexLabeller(rePat),
                   splitter=splitter,
                   item_tfms=item_tfms,
                   batch_tfms=batch_tfms)

  dblock.summary(config["IMAGES_PATH"])
  dl= dblock.dataloaders(config["IMAGES_PATH"],bs=32)
  learn = cnn_learner(dl, resnet18, metrics=accuracy)
  return learn

def build_battery_soc_model_learner_ec(measure_list,test_battery_list,config,
generate_training_images=False,generate_test_images=False,rescale=True):
  """ Train a battery SOC classifier model """
  rePat=config['rePat'] #r'^.*_(\d+).png$'

  #Train - Validation
  dataset,feature_col_names=load_soc_dataset_ec(measure_list,config["soc_list"],config['DATASETS_DIR'])
  if(generate_training_images):
    generate_image_from_ec(dataset,feature_col_names,config['IMAGES_PATH'],config['ExperimentName'],DATA_AUGMENTATION_FACTOR=10)

  #Test dataset
  test_dataset,feature_col_names=load_soc_dataset_ec(test_battery_list,config["soc_list"],config['DATASETS_DIR'])
  if(generate_test_images):
    generate_image_from_ec(test_dataset,feature_col_names,config['TEST_IMAGES_PATH'],config['ExperimentName'],DATA_AUGMENTATION_FACTOR=1)
  
  splitter = config['Splitter'] # RandomSplitter(valid_pct=0.3, seed=41)

  #FastAI image pipeline
  item_tfms = [Resize(224)]
  batch_tfms=[Normalize.from_stats(*imagenet_stats)]

  #Build FastAI DataBlock
  dblock = DataBlock(blocks=(ImageBlock, CategoryBlock),
                   get_items=get_image_files_filtered,
                   get_y=RegexLabeller(rePat),
                   splitter=splitter,
                   item_tfms=item_tfms,
                   batch_tfms=batch_tfms)

  dblock.summary(config["IMAGES_PATH"])
  dl= dblock.dataloaders(config["IMAGES_PATH"],bs=32)
  learn = cnn_learner(dl, resnet18, metrics=accuracy)
  return learn    

def build_image_dataset_from_eis(measure_list,test_battery_list,config,generate_training_images=False,generate_test_images=False):
  ''' Build image dataset from EIS data. Defualt is to generate new images from EIS data.'''
  #Train - Validation
  dataset,eis_col_names=load_soc_dataset(measure_list,config['soc_list'],config['DATASETS_DIR'])
  if(generate_training_images):
    generate_image_files_from_eis(dataset,eis_col_names,config['IMAGES_PATH'],config['ExperimentName'],DATA_AUGMENTATION_FACTOR=10)

  #Test dataset
  test_dataset,eis_col_names=load_soc_dataset(test_battery_list,config['soc_list'],config['DATASETS_DIR'])
  if(generate_test_images):
    generate_image_files_from_eis(test_dataset,eis_col_names,config['TEST_IMAGES_PATH'],config['ExperimentName'],DATA_AUGMENTATION_FACTOR=1)

def load_soc_dataset(measure_list,soc_list, dataset_path,show_data=False):
  ''' Loads the dataset from the CVS file vith EIS data and returns the dataset and the EIS column names '''
  dataset = pd.DataFrame(columns=['SOC','BATTERY_ID','EIS_ID'])
  for measure_index, measure_id in enumerate(measure_list):
    battery_id=measure_id.split("_")[0]
    if show_data:
      print("measure_id: "+str(measure_id))
      print("battery_id: "+str(battery_id))
    #Create a Pandas dataframe from CSV
    df_original= pd.read_csv(dataset_path+CSV_FILE_PREFIX+str(measure_id)+"_ALL_SOC.csv",names=soc_list, low_memory=False)
    #note: csv from matlab are in format 12-64i.
    #      'i" must be replaced with "j" into the CVS file
    df = df_original.apply(lambda col: col.apply(lambda val: val.replace('i','j')))
    #Parse complex number in format: 123-56j, 432+56j
    df = df.apply(lambda col: col.apply(lambda val: complex(val)))
    df_rows=df.transpose()

    eis_col_names= []
    for colIndex in range(0,df_rows.shape[1],1):
      eis_col_names.append("Z_f"+str(colIndex))
    
    if show_data:
      print(eis_col_names)
    
    df_rows.columns=eis_col_names

    #for rowIndex, row in enumerate(df_rows):
    df_rows['SOC']=soc_list
    df_rows['EIS_ID']=measure_id
    df_rows['BATTERY_ID']=battery_id
    dataset= dataset.append(df_rows)
    if show_data:
      print(df_rows)

  return dataset,eis_col_names

def load_soc_dataset_ec(battery_list,soc_list, dataset_path,show_data=False):
  dataset = pd.DataFrame(columns=['SOC','BATTERY'])
  for battery_index, battery_value in enumerate(battery_list):
    if show_data:
      print("battery: "+str(battery_value))
    #Create a Pandas dataframe from CSV
    df= pd.read_csv(dataset_path+CSV_FILE_PREFIX+str(battery_value)+"_EC.csv",names=soc_list, low_memory=False)
    df_rows=df.transpose()

    eis_col_names= []
    for colIndex in range(0,df_rows.shape[1],1):
      eis_col_names.append("ec_param"+str(colIndex))
    
    if show_data:
      print(eis_col_names)
    
    df_rows.columns=eis_col_names

    #for rowIndex, row in enumerate(df_rows):
    df_rows['SOC']=soc_list
    df_rows['BATTERY']=battery_value
    dataset= dataset.append(df_rows)
    if show_data:
      print(df_rows)

  return dataset,eis_col_names

def generate_image_files_from_measure_table(dataset,labels,test_meas_ids,image_path,experimentName,rescale=False,mode ='real+imag'):

  
    
  row_number=dataset.shape[0]
  print("dataset row number: "+str(row_number))
  print("start image file generation. IMAGE_PATH: "+image_path)

  #Create a root folder for image dataset
  import os
  if not os.path.exists(image_path):
    os.mkdir(image_path)
    
  for row_index in range(0,row_number,1):
    soc_label=labels[row_index]
    print("soc: "+str(soc_label))
    measure_id=test_meas_ids[row_index]
    print("measure: "+measure_id)

    row = dataset[row_index]

    x_values, y_values, y2_values = get_xy_values(row,mode)

    # EIS can be rescaled to 0-1 range      
    if rescale:
        x_values,scaler= rescale_dataset(x_values)
        y_values, scaler= rescale_dataset(y_values)
        if mode=='bode':
          y2_values, scaler= rescale_dataset(y2_values)

    img_file_name = image_path+"/"+experimentName+"-"+measure_id+"_"+str(soc_label)+".png"
    print(img_file_name)
    if mode=='bode':
      plot_and_save_bode(x_values, y_values, y2_values, img_file_name)
    else:
      plot_and_save_complex(x_values, y_values, img_file_name)


def generate_image_files_from_eis(dataset,eis_col_names,IMAGES_PATH,experimentName,rescale=False,DATA_AUGMENTATION_FACTOR=1,NOISE_AMOUNT=1e-4):

  row_number=dataset.shape[0]
  print("dataset row number: "+str(row_number))
  print("start image file generation. IMAGE_PATH: "+IMAGES_PATH)

  df=dataset[eis_col_names]
  df_real= df.apply(lambda col: col.apply(lambda val: np.real(val)))
  df_img= df.apply(lambda col: col.apply(lambda val: np.imag(val)))
  #print(df_img)

  print("df_real shape: " + str(df_real.shape))
  print("df_img shape: " + str(df_img.shape))

  #Create a root folder for image dataset
  import os
  if not os.path.exists(IMAGES_PATH):
    os.mkdir(IMAGES_PATH)
    
  if not os.path.exists(IMAGES_PATH+"/"+experimentName):
    os.mkdir(IMAGES_PATH+"/"+experimentName)

  for rowIndex in range(0,row_number,1):
      soc_label=dataset["SOC"].iloc[rowIndex]
      print("soc: "+str(soc_label))
      battery_value=dataset["EIS_ID"].iloc[rowIndex]
      battery_name_str=str(battery_value)
      print("battery: "+battery_name_str)

      
      for augmentation_index in range(0,DATA_AUGMENTATION_FACTOR,1):
        print("augmentation_index: "+str(augmentation_index))
        df_real_copy = df_real.copy(deep=True)
        df_img_copy = df_img.copy(deep=True)
 
        if augmentation_index>0:
          # apply offset to image file name for file generated with data augmentation                     
          augmented_battery_value=AUGMENTATION_OFFSET+(DATA_AUGMENTATION_FACTOR_OFFSET*augmentation_index)+rowIndex
          battery_name_str=str(augmented_battery_value)

          # AWG noise must be added before rescaling    
          df_real_copy= df_real_copy + np.random.normal(0, NOISE_AMOUNT, df_real.shape)
          df_img_copy = df_img_copy+ np.random.normal(0, NOISE_AMOUNT, df_img.shape)          

        # After adding the desidered amount of noise the values of EIS can be rescaled to 0-1 range      
        if rescale:
          df_real_copy,scaler= rescale_dataset(df_real_copy)
          df_img_copy, scaler= rescale_dataset(df_img_copy)

        #Get EIS data for a single SoC value from the dataset                    
        EIS_real=df_real_copy.iloc[rowIndex,:]
        EIS_img=df_img_copy.iloc[rowIndex,:] 

        img_file_name=IMAGES_PATH+"/"+experimentName+CSV_FILE_PREFIX+battery_name_str+"_"+str(soc_label)+".png"
        print(img_file_name)
        plot_and_save_complex(EIS_real,EIS_img,img_file_name)

def compute_image_overlay(image_files):
    from PIL import Image
    ''' Compute the overlay of all the images in the list '''
    background = Image.open(image_files[0])
    # Convert image to RGBA
    background = background.convert("RGBA") 
 
    for i in range(1,len(image_files)):
        img = Image.open(image_files[i])
        img = img.convert("RGBA")
        # Make white background transparent
        datas = img.getdata()

        newData = []
        for item in datas:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)

        img.putdata(newData)
        background.alpha_composite(img)
    return background    

def get_image_file_names_for_soc(image_path,soc):
    '''get all image file names for a given soc value'''
    re_str_soc=r'^.*_('+str(soc)+').png$'
    regExfilter = re.compile(re_str_soc)
    image_files_soc=get_image_files_filtered(image_path,regExfilter)
    return image_files_soc

def generate_image_from_ec(dataset,feature_col_names,IMAGES_PATH,experimentName,rescale=True,DATA_AUGMENTATION_FACTOR=1,NOISE_AMOUNT=1e-5,image_mode="plotAndSave"):

  row_number=dataset.shape[0]
  print("dataset row number: "+str(row_number))
  print("start image file generation. IMAGE_PATH: "+IMAGES_PATH)

  df=dataset[feature_col_names]

  #normalization
  if rescale:
    [df, scaler]= rescale_dataset(df)
  print("df shape: " + str(df.shape))

  #Create a root folder for image dataset
  import os
  if not os.path.exists(IMAGES_PATH):
    os.mkdir(IMAGES_PATH)
    
  if not os.path.exists(IMAGES_PATH+"/"+experimentName):
    os.mkdir(IMAGES_PATH+"/"+experimentName)

  for rowIndex in range(0,row_number,1):
      soc_label=dataset["SOC"].iloc[rowIndex]
      print("soc: "+str(soc_label))
      EIS_id_value=dataset["EIS_ID"].iloc[rowIndex]
      print("EIS ID: "+str(EIS_id_value))
      img_file_name=IMAGES_PATH+"/"+experimentName+CSV_FILE_PREFIX+str(EIS_id_value)+"_"+str(soc_label)+".png"
      print(img_file_name)
      if(image_mode=="plotAndSave"):
        plotAndSave(df.iloc[rowIndex,:],img_file_name)
      else:
        convertToImageAndSave(df.iloc[rowIndex,:],img_file_name)


      if DATA_AUGMENTATION_FACTOR>1:
        original=df.iloc[rowIndex,:]
        augmented_battery_value=1000+(EIS_id_value*DATA_AUGMENTATION_FACTOR)
        for index in range(1,DATA_AUGMENTATION_FACTOR,1):
          EIS_id_value=augmented_battery_value+index
          df_noise= np.random.rand(original.shape[0])*NOISE_AMOUNT
          df_with_noise= original + df_noise
          img_file_name=IMAGES_PATH+"/"+experimentName+CSV_FILE_PREFIX+str(EIS_id_value)+"_"+str(soc_label)+".png"
          print(img_file_name)
          if(image_mode=="plotAndSave"):
            plotAndSave(df_with_noise,img_file_name)
          else:
            convertToImageAndSave(df_with_noise,img_file_name)

def plotAndSave(df,img_file_name):
    fig, ax, _ = plottingfunction(range(0,df.shape[0]),df,show=False)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)
    fig.savefig(img_file_name)
    matplotlib.pyplot.close(fig)

def plot_and_save_complex(df_real,df_img,img_file_name):
  fig, ax, _ = plottingfunction(df_real,df_img,show=False)
  ax.get_yaxis().set_visible(False)
  ax.get_xaxis().set_visible(False)
  fig.savefig(img_file_name)
  matplotlib.pyplot.close(fig)

def plot_and_save_bode2(freqs,magnitudes,phases,img_file_name):

  #plot bode diagram and save it to a file
  fig, ax = plt.subplots(2, 1, figsize=(10, 10))
  ax[0].semilogx(freqs, magnitudes)
  #ax[0].set_ylabel('Magnitude [dB]')
  #ax[0].set_xlabel('Frequency [Hz]')
  #ax[0].set_title('Bode plot')
  ax[1].semilogx(freqs, phases)
  #ax[1].set_ylabel('Phase [deg]')
  #ax[1].set_xlabel('Frequency [Hz]')
  #ax[1].set_title('Bode plot')
  fig.get_axes()[0].get_yaxis().set_visible(False)
  fig.get_axes()[1].get_yaxis().set_visible(False)
  fig.get_axes()[0].get_xaxis().set_visible(False)
  fig.get_axes()[1].get_xaxis().set_visible(False)
  #ax.get_yaxis().set_visible(False)
  #ax.get_xaxis().set_visible(False)
  fig.savefig(img_file_name)
  matplotlib.pyplot.close(fig)


def plot_and_save_bode(freqs,magnitudes,phases,img_file_name):

  #plot bode diagram and save it to a file
  fig, ax = plt.subplots()
  ax.semilogx(freqs, magnitudes,'o-',color='red')
  ax2=ax.twinx()
  ax2.semilogx(freqs, phases,'o-',color='blue')
  ax.get_yaxis().set_visible(False)
  ax.get_xaxis().set_visible(False)
  ax2.get_yaxis().set_visible(False)
  fig.savefig(img_file_name)
  matplotlib.pyplot.close(fig)

def convertToImageAndSave(df,img_file_name):
  normalized = df/(df.max()/255.0)
  img_array = normalized.to_numpy().astype(np.uint8).T
  im = Image.fromarray(img_array)
  #plt.imshow(img_array, cmap='Greys')
  im.save(img_file_name)

def EIS_data_augmentation(dataset,eis_col_names,DATA_AUGMENTATION_FACTOR=1,NOISE_AMOUNT=1e-4):

  dataset_augmented= dataset.copy(deep=True)
  row_number=dataset.shape[0]
  print("dataset row number: "+str(row_number))

  df=dataset[eis_col_names]
  df_real= df.apply(lambda col: col.apply(lambda val: np.real(val)))
  df_img= df.apply(lambda col: col.apply(lambda val: np.imag(val)))
  #print(df_img)

  print("df_real shape: " + str(df_real.shape))
  print("df_img shape: " + str(df_img.shape))

  for rowIndex in range(0,row_number,1):
      soc_label=dataset["SOC"].iloc[rowIndex]
      print("soc: "+str(soc_label))
      measure_id=dataset["EIS_ID"].iloc[rowIndex]
      EIS_ID_str=str(measure_id)
      print("ESI ID: "+EIS_ID_str)
      battery_id=measure_id.split("_")[0]

      
      for augmentation_index in range(0,DATA_AUGMENTATION_FACTOR,1):
        print("augmentation_index: "+str(augmentation_index))
        df_real_copy = df_real.copy(deep=True)
        df_img_copy = df_img.copy(deep=True)
 
        if augmentation_index>0:
          # apply offset to image file name for file generated with data augmentation                     
          augmented_battery_value=AUGMENTATION_OFFSET+(DATA_AUGMENTATION_FACTOR_OFFSET*augmentation_index)+rowIndex
          EIS_ID_str=str(augmented_battery_value)

          # AWG noise must be added before rescaling    
          df_real_copy= df_real_copy + np.random.normal(0, NOISE_AMOUNT, df_real.shape)
          df_img_copy = df_img_copy+ np.random.normal(0, NOISE_AMOUNT, df_img.shape)          

        #Get EIS data for a single SoC value from the dataset                    
        EIS_real=df_real_copy.iloc[rowIndex,:]
        EIS_imag=df_img_copy.iloc[rowIndex,:] 
        EIS_complex=EIS_real+1j*EIS_imag
        df_commn=dataset[['SOC','BATTERY_ID','EIS_ID']] 
        df_commn['BATTERY_ID']=battery_id
        dataset_augmented.append(EIS_complex,ignore_index=True)
  return dataset_augmented

def get_EIS_tabular_dataset_polar(EIS_dataset,feature_col_names):
    '''Returns a tabular dataset with EIS data in polar coordinates and a list of feature names'''
    EIS_dataset.reset_index(inplace=True,drop=True)
    EIS_dataset['SOC_float'] = EIS_dataset.SOC.astype('float')
    df_common=EIS_dataset[['SOC_float','BATTERY_ID','EIS_ID']]

    # Complex Z(f) split in two feature in polar and rectangular format
    # Polar Tabular dataset
    from cmath import phase, polar, rect
    df=EIS_dataset[feature_col_names]
    df_phi= df.apply(lambda col: col.apply(lambda val: phase(val)))
    df_abs= df.apply(lambda col: col.apply(lambda val: abs(val)))

    dataset_polar= df_phi.join(df_abs,lsuffix='_abs' , rsuffix="_phi")
    dataset_polar= df_common.join(dataset_polar)
    #print(dataset_polar)
    polar_feature_names= list()
    for feat_name in feature_col_names:
        polar_feature_names.append(feat_name+"_phi")
        polar_feature_names.append(feat_name+"_abs")
    
    return dataset_polar,polar_feature_names

def get_EIS_tabular_dataset_rectangular(EIS_dataset,feature_col_names):
    '''Returns a tabular dataset with EIS data in polar coordinates and a list of feature names'''
    EIS_dataset.reset_index(inplace=True,drop=True)
    EIS_dataset['SOC_float'] = EIS_dataset.SOC.astype('float')
    df_common=EIS_dataset[['SOC_float','BATTERY_ID','EIS_ID']]

    # Complex Z(f) split in two feature in polar and rectangular format
    # Polar Tabular dataset
    from cmath import phase, polar, rect
    df=EIS_dataset[feature_col_names]
    df_real= df.apply(lambda col: col.apply(lambda val: np.real(val)))
    df_imag= df.apply(lambda col: col.apply(lambda val: np.imag(val)))

    dataset_rect= df_real.join(df_imag,lsuffix='_real' , rsuffix="_imag")
    dataset_rect= df_common.join(dataset_rect)
    #print(dataset_rect)
    
    rect_feature_names= list()
    for feat_name in feature_col_names:
        rect_feature_names.append(feat_name+"_real")
        rect_feature_names.append(feat_name+"_imag")

    return dataset_rect,rect_feature_names

def build_EIS_tabular_learner_rectangular(config,measure_list):
    #Path / default location for saving/loading models
    model_path = '..'

    #The dependent variable/target
    dep_var = 'SOC_float'

    #The list of categorical features in the dataset 
    cat_names = [] 

    dataset,feature_col_names=load_soc_dataset(measure_list,config["soc_list"],config['DATASETS_DIR'])
    dataset_rect,feature_col_names_rect=get_EIS_tabular_dataset_rectangular(dataset,feature_col_names)
    splits = RandomSplitter(valid_pct=0.2)(range_of(dataset_rect))
    learn = build_tabular_learner(dataset_rect,splits,model_path,dep_var,cat_names,feature_col_names_rect)
    return learn

from LiBEIS.code.utilities import augment_meas_data
def generate_EIS_images_for_experiment_plan(experiment_name,experiment_runs_list,meas_table_wide,impedance_col_name,soc_col_name,
measure_id_col_name,root_image_files_path,noise_std_dev=0.0001):
    ''' Generate the images for the experiment run list and store them in the root_image_files_path folder.
    A run represents a single execution of model traning/test/score pipeline. An experiment is a light-weight container for Runs
    
    Parameters: 
    experiment_name: the name of the experiment
    experiment_runs_list: list of tuples (pattern_extraction_mode,normalization_mode,data_augmentation_factor)
    meas_table_wide: the measurement table in wide format
    impedance_col_name: the name of the column containing the impedance values in the measurement table
    soc_col_name: the name of the column containing the soc label values in the measurement table
    measure_id_col_name: the name of the column containing the measure_id label in the measurement table
    root_image_files_path: path where to store the images (a forder will be created for each experiment)
    noise_std_dev: standard deviation of the noise to add to the images
     '''
    
    df_results = pd.DataFrame()
    num_experiment_runs = len(experiment_runs_list)
    for run_idx, experiment in enumerate(experiment_runs_list):
        print(f'Running experiment run  {run_idx + 1} of {num_experiment_runs}')

        experiment_run_name=experiment_name+"_Exp_"+str(run_idx)
        print("Current run name: "+experiment_run_name)
    
        record = generate_EIS_images_for_experiment(experiment_run_name, meas_table_wide, impedance_col_name, soc_col_name, measure_id_col_name, root_image_files_path, noise_std_dev, run_idx, experiment)
    
        df_results = pd.concat([record, df_results.loc[:]]).reset_index(drop=True)
    return df_results

def generate_EIS_images_for_experiment(experiment_run_name, meas_table_wide, impedance_col_name, soc_col_name, measure_id_col_name, root_image_files_path, noise_std_dev, run_idx, experiment):
    data_augmentation_factor=experiment[2]

    if(data_augmentation_factor>1):
        print("Data augmentation factor is greater than 1. Augmenting data...")
        augmented_meas_table_wide=augment_meas_data(meas_table_wide,
              impedance_col_name=impedance_col_name,
              meas_id_col_name=measure_id_col_name,
              data_augmentation_factor=data_augmentation_factor,
              noise_std_dev=noise_std_dev
            )
    else:
        augmented_meas_table_wide=meas_table_wide
    
    #Compute the patterns
    patterns = get_patterns(augmented_meas_table_wide, impedance_col_name,
                            mode = experiment[0].mode, 
                            kwargs = experiment[0].params)
    #Perform data normalisation
    patterns = experiment[1].normalise(patterns)
    data_augmentation_factor=experiment[2]

    image_path = root_image_files_path+"/"+experiment_run_name
    
    soc_labels = augmented_meas_table_wide[(soc_col_name)].to_list()
    meas_ids = augmented_meas_table_wide[(measure_id_col_name)].to_list()

    generate_image_files_from_measure_table(patterns,soc_labels,meas_ids,image_path,experiment_run_name, mode = experiment[0].mode)


    #Add record to dataframe
    record = pd.DataFrame({
        'Cross_validation_experiment_index' : run_idx,
        'Experiment index' : run_idx,
        'Feature extraction mode' : experiment[0].mode,
        'Feature normalisation mode' : experiment[1].name,
        'Data augmentation factor' : data_augmentation_factor,
        'Num features' : patterns.shape[1],
        'Run name' : experiment_run_name,
        'Image path' : image_path},
        index = [0])
    
    return record