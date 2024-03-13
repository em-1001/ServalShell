# preprocess data
import sys
sys.path.append("./Tellina")
from bashlint.data_tools import bash_tokenizer, bash_parser, ast2tokens, ast2command
from nlp_tools import tokenizer
from bashlint import data_tools
from encoder_decoder import slot_filling
import json
from config import get_config

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def save_data(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def pre_processing(config):
    ds_raw = load_data('./Data/nl2bash/nl2bash.json')

    # data pre-processing
    print("data pre processing...")
    for item in ds_raw.values():
        item[config['lang_src']] = ' '.join(tokenizer.ner_tokenizer(item[config['lang_src']])[0])
        item[config['lang_tgt']] = ' '.join(bash_tokenizer(item[config['lang_tgt']], loose_constraints=True, arg_type_only=True))
        ...

    # save pre-processed data to a new JSON file
    save_data(ds_raw, './Data/nl2bash/preprocessed_data.json')

config = get_config()
pre_processing(config)
