# train.py
# reference : https://youtu.be/ISNdQcPhsts?si=F5xPY5JV92VNdKog
# original code : https://github.com/hkproj/pytorch-transformer/blob/main/train.py

import sys
sys.path.append("./Tellina")
from bashlint.data_tools import bash_tokenizer, bash_parser, ast2tokens, ast2command
from nlp_tools import tokenizer
from bashlint import data_tools
from encoder_decoder import slot_filling

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dataset import BilingualDataset, causal_mask
from model import Transformer
from config import get_weights_file_path, get_config
from translate import translate


from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace, Metaspace
from tokenizers.pre_tokenizers import WhitespaceSplit

from torch.utils.tensorboard import SummaryWriter

import warnings
from tqdm import tqdm
from pathlib import Path

import json
from sklearn.model_selection import train_test_split


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


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break


def get_all_sentences(ds, lang):
    for item in ds.values():
      yield item[lang]


def get_or_build_tokenizer(config, ds, lang):
  tokenizer_path = Path(config['tokenizer_file'].format(lang))
  if not Path.exists(tokenizer_path):
    tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=1)
    tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
    tokenizer.save(str(tokenizer_path))
  else:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
  return tokenizer


def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def get_ds(config):
  ds_raw = load_data('./Data/nl2bash/preprocessed_data.json')

  # Build tokenizers
  tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
  tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

  # keep 90% for training and 10% for validation
  train_ds_raw, val_ds_raw = train_test_split(list(ds_raw.values()),train_size = 0.9)

  train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
  val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])

  max_len_src = 0
  max_len_tgt = 0

  for item in ds_raw.values():
    src_ids = tokenizer_src.encode(item[config['lang_src']]).ids
    tgt_ids = tokenizer_tgt.encode(item[config['lang_tgt']]).ids
    max_len_src = max(max_len_src, len(src_ids))
    max_len_tgt = max(max_len_tgt, len(tgt_ids))

  print(f'Max length of source sentence: {max_len_src}')
  print(f'Max length of target sentence: {max_len_tgt}')

  train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
  val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)

  return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
  model = Transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
  for p in model.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)
  return model


def train_model(config):
  # Define the device
  device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
  print("Using device:", device)
  if (device == 'cuda'):
    print(f"Device name: {torch.cuda.get_device_name(device.index)}")
    print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
  elif (device == 'mps'):
    print(f"Device name: <mps>")
  else:
    print("NOTE: If you have a GPU, consider using it for training.")
    print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
    print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)

  Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

  train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
  model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
  # Tensorboard
  writer = SummaryWriter(config['experiment_name'])

  optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.998), lr=config['lr'], eps=1e-9)

  if False: # config['cos_annealing']:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120)

  initial_epoch = 0
  global_step = 0
  if False: # config['preload']:
    model_filename = get_weights_file_path(config)
    print(f"Preloading model {model_filename}")
    state = torch.load(model_filename)
    initial_epoch = state['epoch'] + 1
    optimizer.load_state_dict(state['optimizer_state_dict'])
    global_step = state['global_step']

  loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

  for epoch in range(initial_epoch, config['num_epochs']+1):
    if False: # config['cos_annealing']:
      print(scheduler._last_lr)
    model.train()
    batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}")

    for batch in batch_iterator:

      encoder_input = batch['encoder_input'].to(device) # (B, Seq_Len)
      decoder_input = batch['decoder_input'].to(device) # (B, Seq_Len)
      encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, Seq_Len)
      decoder_mask = batch['decoder_mask'].to(device) # (B, 1, Seq_Len, Seq_Len)

      # Run the tensors through the transformer
      encoder_output = model.encode(encoder_input, encoder_mask) # (B, Seq_Len, d_model)
      decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, Seq_Len, d_model)
      proj_output = model.project(decoder_output) # (B, Seq_len, tgt_vocab_size)

      label = batch['label'].to(device) # (B, Seq_Len)

      # (B, Seq_Len, tgt_vocab_size) -> (B * Seq_Len, tgt_vocab_size)
      loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
      batch_iterator.set_postfix({f"loss": f"{loss.item():6.3f}"})

      # Log the loss
      writer.add_scalar('train loss', loss.item(), global_step)
      writer.flush()

      # Backpropagate the loss
      loss.backward()

      # Gradient Clipping
      # max_norm = 5.0
      # nn.utils.clip_grad_norm_(model.parameters(), max_norm)

      # Update the weights
      optimizer.step()
      optimizer.zero_grad()

      global_step += 1

    if False: # config['cos_annealing']:
      scheduler.step()

    # Run validation at the end of every epoch
    run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

    # Save the model at the end of every epoch
    model_filename = get_weights_file_path(config)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': global_step
    }, model_filename)

    if epoch == 1:# epoch > 0 and epoch % 3 == 0:
      nl = 'print current user name'
      nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
      translate(nl)

      nl = 'copies "file.txt" to "null.txt"'
      nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
      translate(nl)

      nl = 'finds all files with a ".txt" extension in the current directory'
      nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
      translate(nl)

      nl = 'prints "Hello, World!" on the terminal'
      nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
      translate(nl)

      nl = 'list current dictory files'
      nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
      translate(nl)

      nl = 'Prints the current working directory.'
      nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
      translate(nl)

      nl = 'gives execute permission to "script.sh"'
      nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
      translate(nl)

      nl = 'changes the owner and group of "file.txt" to "user:group"'
      nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
      translate(nl)

      nl = 'moves "file.txt" to "./bin"'
      nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
      translate(nl)

      nl = 'deletes a file named "file.txt"'
      nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
      translate(nl)

      nl = 'creates a directory named "my_folder"'
      nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
      translate(nl)

      nl = 'changes to the "Documents" directory'
      nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
      translate(nl)

      nl = 'displays the content of "file.txt"'
      nl = ' '.join(tokenizer.ner_tokenizer(nl)[0])
      translate(nl)


if __name__ == '__main__':
  warnings.filterwarnings('ignore')
  config = get_config()
  train_model(config)
