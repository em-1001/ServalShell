# translate
# reference : https://youtu.be/ISNdQcPhsts?si=F5xPY5JV92VNdKog
# original code : https://github.com/hkproj/pytorch-transformer/blob/main/translate.py

from pathlib import Path
from config import get_config, get_weights_file_path
from model import Transformer
from tokenizers import Tokenizer
from dataset import BilingualDataset
import torch
import sys
from train import get_model, get_ds, run_validation


def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def get_ds(config):
  ds_raw = load_data('/content/nl2bash-data.json')


def translate(sentence: str):
    # Define the device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()
    tokenizer_src = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_src']))))
    tokenizer_tgt = Tokenizer.from_file(str(Path(config['tokenizer_file'].format(config['lang_tgt']))))
    model = Transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), config['seq_len'], config['seq_len'], config['d_model']).to(device)

    # Load the pretrained weights
    model_filename = "./tellina21epoch.pth" # get_weights_file_path(config)
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

    # translate the sentence
    model.eval()
    with torch.no_grad():
        sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

        enc_input_tokens = tokenizer_src.encode(sentence).ids

        enc_num_padding_tokens = config['seq_len'] - len(enc_input_tokens) - 2


        encoder_input = torch.cat(
            [
               sos_token,
               torch.tensor(enc_input_tokens, dtype=torch.int64),
               eos_token,
               torch.tensor([pad_token] * enc_num_padding_tokens, dtype=torch.int64)
           ]
        )
        source_mask = (encoder_input != pad_token).unsqueeze(0).int()
        encoder_output = model.encode(encoder_input.unsqueeze(0).to(device), source_mask.to(device))

        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id('[SOS]')).type_as(encoder_input).to(device)

        def causal_mask(size):
          mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
          return mask == 0

        # Generate the translation word by word
        while decoder_input.size(1) < config['seq_len']:
            # build mask for target and calculate output
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
            out = model.decode(encoder_output.to(device), source_mask.to(device), decoder_input.to(device), decoder_mask.to(device))

            # project next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(encoder_input).fill_(next_word.item()).to(device)], dim=1)

            # break if we predict the end of sentence token
            if next_word == tokenizer_tgt.token_to_id('[EOS]'):
                break

    # convert ids to tokens
    return tokenizer_tgt.decode(decoder_input.squeeze(0).detach().cpu().numpy())

