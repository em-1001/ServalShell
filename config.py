# config.py
# reference : https://youtu.be/ISNdQcPhsts?si=F5xPY5JV92VNdKog
# original code : https://github.com/hkproj/pytorch-transformer/blob/main/config.py

from pathlib import Path

def get_config():
  return {
      "batch_size": 8,
      "num_epochs": 100,
      "lr": 1e-4,
      "seq_len": 100, # 512
      "d_model": 512,
      "lang_src": "invocation",
      "lang_tgt": "cmd",
      "model_folder": "./weights",
      "model_basename": "tmodel_",
      "preload": None,
      "tokenizer_file": "tokenizer_{0}.json",
      "experiment_name": "./tmodel",
      "cos_anneal": False
  }


def get_weights_file_path(config):
  model_folder = config["model_folder"]
  model_basename = config["model_basename"]
  model_filename = f"{model_basename}x.pth" 
  return str(Path('.') / model_folder / model_filename)

