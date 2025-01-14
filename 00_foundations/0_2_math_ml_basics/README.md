# 0.2 Math / ML basics

One notebook per topic. Each one starts from a real production situation, shows the code, and reads the output back. Real embeddings (OpenAI), real probabilities (OpenAI logprobs, Ollama), real data (20 Newsgroups) — the math is always run, never just written.

| # | Notebook | What you will see |
|---|---|---|
| 01 | [Vectors](01_vectors.ipynb) | "charged twice" and "duplicate payment" become vectors and land close together |
| 02 | [dot product](02_dot_product.ipynb) | Four articles scored against a query with `@`; the same number by hand |
| 03 | [cosine similarity](03_cosine_similarity.ipynb) | A long policy document wins every dot product; cosine fixes the ranking |
| 04 | [norms](04_norms.ipynb) | OpenAI vectors at length 1, TF-IDF vectors not; normalise, then dot = cosine |
| 05 | [Matrix multiplication](05_matrix_multiplication.ipynb) | 4 million scores: Python loop vs one `@`, same numbers, hundreds of times faster |
| 06 | [softmax](06_softmax.ipynb) | Router scores → percentages; how scaling the scores changes confidence |
| 07 | [attention math](07_attention_math.ipynb) | `softmax(QKᵀ/√d)V` by hand equals PyTorch's `scaled_dot_product_attention` |
| 08 | [Probability](08_probability.ipynb) | The model's real top-5 next tokens with their probabilities |
| 09 | [sampling: temperature](09_sampling_temperature.ipynb) | Real logprobs re-softmaxed at T=0.2/1/2; 5 calls at T=0 vs T=1.8 |
| 10 | [sampling: top-k](10_sampling_top_k.ipynb) | 20% of probability in the tail; Ollama with k=1 vs k=50 |
| 11 | [sampling: top-p](11_sampling_top_p.ipynb) | p=0.9 keeps 1 token when certain, 5 when open; OpenAI top_p 0.1 vs 1.0 |
| 12 | [sampling: min-p](12_sampling_min_p.ipynb) | Threshold relative to the top token; Ollama min_p on vs off (with the garbage it removes) |
| 13 | [Gradient descent](13_gradient_descent.ipynb) | ms-per-token recovered from 500 logged requests, loss falling step by step |
| 14 | [loss functions: cross-entropy](14_loss_functions_cross_entropy.ipynb) | Confident-and-wrong costs 4, unsure costs 1; an untrained LM's loss ≈ ln(vocab) |
| 15 | [Train/val/test split](15_train_val_test_split.ipynb) | 60/20/20 with the category mix preserved, no overlap |
| 16 | [overfitting](16_overfitting.ipynb) | 20 Newsgroups: 100% train vs 51% validation at 30 posts, gap shrinking with data |
| 17 | [regularization](17_regularization.ipynb) | Ridge alpha sweep: U-shaped validation error, chosen at the bottom; `weight_decay` in PyTorch |
| 18 | [Embedding spaces](18_embedding_spaces.ipynb) | Nearest neighbours of "delivery", "invoice", "password" |
| 19 | [PCA](19_pca.ipynb) | 12 words in 2-D with groups apart; 1536 → 256 keeps the top result |
| 20 | [UMAP](20_umap.ipynb) | 90 messages → three islands matching three topics, no labels used |
| 21 | [t-SNE](21_tsne.ipynb) | Same islands, two perplexities |
| 22 | [Break → Fix](22_break_fix.ipynb) | Query from one embedding model, index from another: silent nonsense → one model per index |
