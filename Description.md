# model.py
## src_mask & tgt_mask
In the context of the transformer model, the difference between src_mask and tgt_mask lies in their usage within the model.

1. src_mask (Encoder Mask):  
- src_mask is primarily used in the encoder part of the transformer model.  
- It is employed to mask out padding tokens in the input sequences.  
- The purpose is to ensure that the model doesn’t attend to the padding tokens, which are added to the input sequences to make them uniform in length but don’t carry any meaningful information.  
- Padding token positions are masked with zeros in the src_mask.  
