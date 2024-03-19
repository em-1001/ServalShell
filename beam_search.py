# translate
# reference : https://youtu.be/ISNdQcPhsts?si=F5xPY5JV92VNdKog
# original code : https://github.com/hkproj/pytorch-transformer/blob/main/translate.py

from pathlib import Path
from config import get_config, get_weights_file_path
from model import Transformer
from tokenizers import Tokenizer
from dataset import BilingualDataset, causal_mask
import torch
import sys
from train import get_model, get_ds, run_validation


def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def get_ds(config):
  ds_raw = load_data('/content/nl2bash-data.json')


def length_penalty(length, alpha=1.2, min_length=3):
    return ((min_length + length) / (min_length + 1))**alpha

def beam_search(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device, beam_width=3):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')    

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)

    # Initialize beams
    beams = [[torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device), 0.0]]
    eos_cnt = 0 
    eos_candidates = []
    
    for _ in range(max_len):
        next_candidates = []

        for beam_input, beam_score in beams:

            #if beam_input[-1] == eos_idx :
            #    next_candidates.append([beam_input, beam_score])
            #    continue

            # build mask for target
            decoder_mask = causal_mask(beam_input.size(1)).type_as(source_mask).to(device)

            # calculate output
            out = model.decode(encoder_output, source_mask, beam_input, decoder_mask)

            # get top k next words
            prob = model.project(out[:, -1])
            topk_scores, topk_words = torch.topk(prob, 2*beam_width-1) 
            # print(topk_scores[0], topk_words[0])

            boundary = beam_width
            loop = 0
            for score, word_idx in zip(topk_scores[0], topk_words[0]):
                if loop == boundary:
                    break

                new_beam_input = torch.cat(
                    [beam_input, torch.empty(1, 1).type_as(source).fill_(word_idx.item()).to(device)], dim=1
                )
                new_score = beam_score - score.item()  # Negative log likelihood

                if word_idx == eos_idx:
                    eos_candidates.append([new_beam_input, new_score])
                    boundary += 1
                    eos_cnt += 1
                    if eos_cnt == beam_width:
                        break
                else:
                    next_candidates.append([new_beam_input, new_score])

                loop += 1

            if eos_cnt == beam_width:
                break

        # Sort the next candidates and select top k
        next_candidates.sort(key=lambda x: x[1])
        beams = next_candidates[:beam_width]

        # Check if all beams have reached EOS
        if eos_cnt == beam_width:
            break

    # Select the beam with the highest score
    # print(eos_candidates)
    for text, score in eos_candidates:
        print(text, score, len(text[0]), score/length_penalty(len(text[0])))

    best_beam = min(eos_candidates, key=lambda x: x[1]/length_penalty(len(x[0][0])))
    return best_beam[0]



def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)


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

        # decode_output = greedy_decode(model, encoder_input.unsqueeze(0).to(device), source_mask.to(device), tokenizer_src, tokenizer_tgt, config['seq_len'], device)
        decode_output = beam_search(model, encoder_input.unsqueeze(0).to(device), source_mask.to(device), tokenizer_src, tokenizer_tgt, config['seq_len'], device, beam_width=3)  

    # convert ids to tokens
    return tokenizer_tgt.decode(decode_output.squeeze(0).detach().cpu().numpy())
