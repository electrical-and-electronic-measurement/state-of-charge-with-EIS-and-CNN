from IPython import get_ipython

""" Check is the code is running in Google Colab environment  """
def is_running_in_colab():
  return 'google.colab' in str(get_ipython())

def is_running_in_jupyter():
  return 'ipykernel' in str(get_ipython())

def is_running_in_notebook():
  return is_running_in_colab() or is_running_in_jupyter()

def copy_model_to_google_drive(filename,src_path,dest_path):
  '''copy file from Colab VM  to google drive'''
  dest_filename=dest_path+"/"+filename
  #copy to external storage (outside Colab VM)
  model_source=src_path+"/"+filename
  #!cp $model_source  $dest_filename
  #copy to external storage (outside Colab VM)
  get_ipython().system('cp $model_source  $dest_filename')


def get_root_path(gDrivePath):
    """ Get path to the Google Drive root working folder """
    RunningInCOLAB = is_running_in_colab()
    if RunningInCOLAB :
        from google.colab import drive
        print("Running on COLAB")
        drive.mount('/gdrive',force_remount=True)
        ROOT_DIR = '/gdrive/MyDrive/'+gDrivePath
    
    else:
        print("NOT running on COLAB")
        ROOT_DIR=".."
    return ROOT_DIR

""" Print information about sytem CPU, GPU, and RAM """
def print_system_resource_info():
  get_ipython().system('pip install gputil psutil humanize')
  import psutil
  import humanize
  import os
  import GPUtil as GPU
  import torch

  process = psutil.Process(os.getpid())
  print("Gen RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ), " | Proc size: " + humanize.naturalsize( process.memory_info().rss))

  GPUs = GPU.getGPUs()
  
  for gpu in GPUs:
    print("GPU RAM Free: {0:.0f}MB | Used: {1:.0f}MB | Util {2:3.0f}% | Total {3:.0f}MB".format(gpu.memoryFree, gpu.memoryUsed, gpu.memoryUtil*100, gpu.memoryTotal))
    print("GPU Name: "+ torch.cuda.get_device_name())
    