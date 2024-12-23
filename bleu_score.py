# !pip install torchtext==0.6.0

import torch
from torch.utils.data import Dataset, DataLoader
from train import get_ds, get_model
from config import get_config 
from beam_search import greedy_search, beam_search, length_penalty
from torchtext.data.metrics import bleu_score


device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"

config = get_config()
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)


model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
model_filename = get_weights_file_path(config) 
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])
model.eval()

trgs = []
pred_trgs = [] 
index = 0

with torch.no_grad():
  for batch in val_dataloader:
    encoder_input = batch["encoder_input"].to(device) # (b, seq_len)
    encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq_len)
    assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

    if config['beam_search']:
      model_out = beam_search(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, config['seq_len'], device, beam_width=config['beam_width'])[0][0].squeeze(0)
    else:
      model_out, _ = greedy_search(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, config['seq_len'], device)

    source_text = batch["src_text"][0].split(' ')
    target_text = batch["tgt_text"][0].split(' ')
    model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy()).split(' ')

    pred_trgs.append(model_out_text)
    trgs.append([target_text])

    index += 1
    if (index + 1) % 100 == 0:
      print(f"[{index + 1}/{len(val_dataloader)}]")
      print(f"예측: {model_out_text}")
      print(f"정답: {target_text}") 

bleu = bleu_score(pred_trgs, trgs, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])
print(f'Total BLEU Score = {bleu*100:.2f}')

individual_bleu1_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1, 0, 0, 0])
individual_bleu2_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[0, 1, 0, 0])
individual_bleu3_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[0, 0, 1, 0])
individual_bleu4_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[0, 0, 0, 1])

print(f'Individual BLEU1 score = {individual_bleu1_score*100:.2f}')
print(f'Individual BLEU2 score = {individual_bleu2_score*100:.2f}')
print(f'Individual BLEU3 score = {individual_bleu3_score*100:.2f}')
print(f'Individual BLEU4 score = {individual_bleu4_score*100:.2f}')

cumulative_bleu1_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1, 0, 0, 0])
cumulative_bleu2_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1/2, 1/2, 0, 0])
cumulative_bleu3_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1/3, 1/3, 1/3, 0])
cumulative_bleu4_score = bleu_score(pred_trgs, trgs, max_n=4, weights=[1/4, 1/4, 1/4, 1/4])

print(f'Cumulative BLEU1 score = {cumulative_bleu1_score*100:.2f}')
print(f'Cumulative BLEU2 score = {cumulative_bleu2_score*100:.2f}')
print(f'Cumulative BLEU3 score = {cumulative_bleu3_score*100:.2f}')
print(f'Cumulative BLEU4 score = {cumulative_bleu4_score*100:.2f}')  
