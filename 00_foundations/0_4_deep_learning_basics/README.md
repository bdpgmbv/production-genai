# 0.4 Deep learning basics

One notebook per topic. Every model is trained in the notebook, on a laptop CPU, on real data (SST-2 movie reviews, WikiText-2), in seconds to a couple of minutes. Built from the same PyTorch modules production code uses; the "from scratch" items assemble them so nothing is hidden.

| # | Notebook | What you will see |
|---|---|---|
| 01 | [PyTorch tensors](01_pytorch_tensors.ipynb) | 200 re-ranking scores: Python loop vs `@`; shape, dtype, device — and the error when devices mix |
| 02 | [autograd](02_autograd.ipynb) | `backward()` fills gradients that match the derivative by hand; accumulation and `no_grad` |
| 03 | [nn.Module](03_nn_module.ipynb) | A sentiment classifier: parameters listed, `state_dict` saved and restored, train vs eval |
| 04 | [Training loop from scratch](04_training_loop_from_scratch.ipynb) | The 8-step loop on SST-2: loss falling, ~76% validation accuracy in 2 s |
| 05 | [RNN](05_rnn.ipynb) | A plain RNN stuck at 50% — the vanishing gradient, live |
| 06 | [LSTM](06_lstm.ipynb) | Gates fix it: ~70%; 4× the parameters per cell |
| 07 | [GRU](07_gru.ipynb) | Two gates, one state: ~72% for three quarters of the parameters |
| 08 | [Seq2seq](08_seq2seq.ipynb) | Date normalisation learned from examples; the one-vector bottleneck |
| 09 | [Seq2seq + attention](09_seq2seq_attention.ipynb) | 100% exact match, and a printed trace of which input character each output looked at |
| 10 | [Transformer: positional encoding](10_transformer_positional_encoding.ipynb) | Proof that a transformer layer is order-blind without positions |
| 11 | [Transformer: multi-head attention](11_transformer_multi_head_attention.ipynb) | `nn.MultiheadAttention` reproduced by hand; four heads looking at four places |
| 12 | [Transformer: encoder](12_transformer_encoder.ipynb) | A from-scratch encoder on 8k sentences: below the LSTM — transformers are data-hungry |
| 13 | [Transformer: decoder](13_transformer_decoder.ipynb) | The causal mask; proof the future cannot leak; next-token targets |
| 14 | [Transformer from scratch](14_transformer_from_scratch.ipynb) | Full encoder-decoder on the date task: 94% in 21 s |
| 15 | [nanoGPT](15_nanogpt.ipynb) | A 4-layer char GPT on WikiText-2: loss 5.9 → 1.6 in two minutes, then it writes |
| 16 | [BERT fine-tuning](16_bert_fine_tuning.ipynb) | BERT-tiny with `Trainer`: 50% → 76% in 15 s |
| 17 | [Break → Fix](17_break_fix.ipynb) | Same input, five different answers: `model.eval()` was never called |
