# model.py
## src_mask & tgt_mask
In the context of the transformer model, the difference between src_mask and tgt_mask lies in their usage within the model.

1. src_mask (Encoder Mask):  
- **src_mask** is primarily used in the encoder part of the transformer model.  
- It is employed to mask out padding tokens in the input sequences.  
- The purpose is to ensure that the model doesn’t attend to the padding tokens, which are added to the input sequences to make them uniform in length but don’t carry any meaningful information.  
- Padding token positions are masked with zeros in the **src_mask**.  

2. tgt_mask (Decoder Mask):  
- **tgt_mask** is mainly used in the decoder part of the transformer model.  
- It is applied to mask out future tokens in the decoder’s self-attention mechanism and also in the cross-attention mechanism between the decoder and the encoder.    
- The objective is to prevent the model from accessing future information during training and generation.  
- Future token positions in the decoder are masked with zeros in the **tgt_mask**.

By utilizing these masks appropriately, the transformer model can effectively process input sequences while also ensuring that it generates outputs correctly by attending to relevant parts of the input and avoiding attending to irrelevant or future information.

The **src_mask** and **tgt_mask** that are returned on dataset.py go through the following operation.

