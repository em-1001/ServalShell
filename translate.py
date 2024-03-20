# translate
# reference : https://youtu.be/ISNdQcPhsts?si=F5xPY5JV92VNdKog
# original code : https://github.com/hkproj/pytorch-transformer/blob/main/translate.py

from pathlib import Path
from config import get_config, get_weights_file_path
from model import Transformer
from tokenizers import Tokenizer
import torch
from beam_search import greedy_search, beam_search, length_penalty


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

        if config['beam_search']:
            decode_output = beam_search(model, encoder_input.unsqueeze(0).to(device), source_mask.to(device), tokenizer_src, tokenizer_tgt, config['seq_len'], device, beam_width=config['beam_width'])
        else: 
            decode_output = greedy_search(model, encoder_input.unsqueeze(0).to(device), source_mask.to(device), tokenizer_src, tokenizer_tgt, config['seq_len'], device)  

    # convert ids to tokens
    return tokenizer_tgt.decode(decode_output.squeeze(0).detach().cpu().numpy())
