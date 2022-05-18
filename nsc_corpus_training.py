# -*- coding: utf-8 -*-
"""
Created on Thu May 19 01:04:35 2022

@author: coco
"""

import nemo
import nemo.collections.asr as nemo_asr
import jiwer
from jiwer import wer
import json
import zipfile
import os 
import librosa
import torch
import torch.nn as nn
import copy
from omegaconf import DictConfig
import pytorch_lightning as pl
import time

train_manifest_path = '/home/dgxuser/Desktop/kailin/CHANNEL0/train_manifest.json'
val_manifest_path = '/home/dgxuser/Desktop/kailin/CHANNEL0/val_manifest.json'
data_dir = '/home/dgxuser/Desktop/kailin/CHANNEL0/WAVE'
script_dir = '/home/dgxuser/Desktop/kailin/CHANNEL0/SCRIPT/'
train_speakers = ['0091', '0124', '0135', '0148', '0173', '0178', '0190', '0194', '0208', '0218', '0245', '0276', '0289', '0308', '0314', '0317', '0329', '0330', '0337', '0349', '0379', '0397', '0432', '0443', '0451', '0462', '0488', '0489', '0490', '0501', '0537', '0543', '0557', '0591', '0592', '0603', '0620', '0627', '0634', '0658', '0659', '0675', '0743', '0762', '0763', '0766', '0778', '0780', '0785', '0790', '0791', '0792', '0798', '0799', '0812', '0831', '0872', '0881', '1058', '1064', '1444']
val_speakers = ['0044', '0047', '0056', '0066', '0074']


def build_manifest(manifest_path, speakers):
  count = 0
  with open(manifest_path, 'w') as fout: 
    for i in speakers: 
      path_unzipped = data_dir
      if not os.path.exists(path_unzipped):
        path = data_dir + '/SPEAKER' + i + '.zip'
        zip_ref = zipfile.ZipFile(path, 'r')
        zip_ref.extractall(path=data_dir)
      n = 0 
      wav_folder = data_dir + '/SPEAKER' + i + '/SESSION'
      while os.path.exists(wav_folder + str(n)):
        files = []
        for filename in os.listdir(wav_folder + str(n)):
          files.append(filename)
        files = sorted(files)
        script_path = script_dir + filename[:6] + '.TXT'
        with open(script_path, encoding = 'utf-8') as script:
          for filename in files:
            #getting transcript 
            script.readline()
            line = script.readline().strip()
            words = line.split()
            edited_words = []
            for word in words:
              if (word != '<SPK/>') and (word != '**'):
                edited_words.append(word.lower())
            transcript_line = ""
            for word in edited_words: 
              transcript_line = transcript_line + word + " "
            transcript_line = transcript_line[:-1]
            #getting audio file
            audio_path = os.path.join(wav_folder + str(n), filename)
            duration = librosa.core.get_duration(filename=audio_path)
            count = count+1
            # Write the metadata to the manifest
            metadata = {
                "audio_filepath": audio_path,
                "duration": duration,
                "text": transcript_line
            }
            json.dump(metadata, fout)
            fout.write('\n')
        n = n+1
  return count 

# Building Manifest Files
train_count = build_manifest(train_manifest_path, train_speakers)
val_count = build_manifest(val_manifest_path, val_speakers)
print("Number of Training Testcases(Lines): " + str(train_count))
print("Number of Validation Testcases(Lines): " + str(val_count))
print("***Done Training***")

# Config Information 
try:
    from ruamel.yaml import YAML
except ModuleNotFoundError:
    from ruamel_yaml import YAML
config_path = './configs/config.yaml'

yaml = YAML(typ='safe')
with open(config_path) as f:
    params = yaml.load(f)
    
## Initialise quartznet model 
quartznet = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="QuartzNet15x5Base-En")

def enable_bn_se(m):
    if type(m) == nn.BatchNorm1d:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

    if 'SqueezeExcite' in type(m).__name__:
        m.train()
        for param in m.parameters():
            param.requires_grad_(True)

quartznet.encoder.freeze()
quartznet.encoder.apply(enable_bn_se)
print("Model encoder has been frozen, and batch normalization has been unfrozen")

# add in manifest paths 
params['model']['train_ds']['manifest_filepath'] = train_manifest_path
params['model']['validation_ds']['manifest_filepath'] = val_manifest_path
params['model']['train_ds']['batch_size'] = 16
params['model']['validation_ds']['batch_size'] = 16

# lr decrease for fine-tuning 
new_opt = copy.deepcopy(params['model']['optim'])
new_opt['lr'] = 0.001

# Use the smaller learning rate we set before
quartznet.setup_optimization(optim_config=DictConfig(new_opt))

# Point to the data we'll use for fine-tuning as the training set
quartznet.setup_training_data(train_data_config=params['model']['train_ds'])

# Point to the new validation data for fine-tuning
quartznet.setup_validation_data(val_data_config=params['model']['validation_ds'])

#create a PyTorch Lightning trainer and call `fit` again.
#trainer = pl.Trainer(max_epochs=10, accelerator='gpu', devices=4, distributed_backend="ddp")
trainer = pl.Trainer(max_epochs=10, gpus=4, accelerator="ddp")
start = time.time()
trainer.fit(quartznet)
end = time.time()
print('Trained with 23871 files totalling 34.19 hours')
print('Validated with 2373 files totalling 3.02 hours')
print(f'Time taken for 10 epochs with 4 GPUs: {str(end-start)} seconds')


