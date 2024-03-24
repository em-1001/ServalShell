from dataset import causal_mask
import torch


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

            decoder_mask = causal_mask(beam_input.size(1)).type_as(source_mask).to(device)

            out = model.decode(encoder_output, source_mask, beam_input, decoder_mask)

            prob = model.project(out[:, -1])
            topk_scores, topk_words = torch.topk(prob, 2*beam_width-1)

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

        next_candidates.sort(key=lambda x: x[1])
        beams = next_candidates[:beam_width]

        if eos_cnt == beam_width:
            break

    # Select the beam with the highest score
    #for text, score in eos_candidates:
    #    print(text, score, len(text[0]), score/length_penalty(len(text[0])))

    best_beam = sorted(eos_candidates, key=lambda x: x[1]/length_penalty(len(x[0][0])))
    return [best_beam[i][0] for i in range(beam_width)]



def greedy_search(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
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
