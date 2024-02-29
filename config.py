# config.py
"""
from google.colab import drive
drive.mount('/content/drive')
"""

from pathlib import Path

def get_config():
  return {
      "batch_size": 8,
      "num_epochs": 100,
      "lr": 1e-4,
      "seq_len": 350, # 510
      "d_model": 512,
      "lang_src": "invocation",
      "lang_tgt": "cmd",
      "model_folder": "/content/drive/MyDrive/transformer/weights",
      "model_basename": "tmodel_",
      "preload": None,
      "tokenizer_file": "tokenizer_{0}.json",
      "experiment_name": "/content/tmodel"
  }


def get_weights_file_path(config):
  model_folder = config["model_folder"]
  model_basename = config["model_basename"]
  model_filename = f"{model_basename}x.pth" # x2
  return str(Path('.') / model_folder / model_filename)

