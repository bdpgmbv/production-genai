# 0.3 Classical NLP

One notebook per topic. Each one starts from a real production situation, shows the code, and reads the output back. Real tokenizers (BERT, tiktoken, trained BPE/SentencePiece), real models (spaCy), real data (20 Newsgroups), real pretrained vectors (GloVe).

| # | Notebook | What you will see |
|---|---|---|
| 01 | [Tokenization: word](01_tokenization_word.ipynb) | `split()` vs `word_tokenize` on money, contractions, abbreviations; a word count that finally adds up |
| 02 | [Tokenization: subword](02_tokenization_subword.ipynb) | "chargeback" → `charge` + `##back`; no word is ever unknown |
| 03 | [Tokenization: BPE](03_tokenization_bpe.ipynb) | A BPE tokenizer trained on a support corpus; the first merges it learns |
| 04 | [Tokenization: WordPiece](04_tokenization_wordpiece.ipynb) | BERT's `##` pieces; longest-match-first written by hand and matched to the library |
| 05 | [Tokenization: SentencePiece](05_tokenization_sentencepiece.ipynb) | English and Japanese from one tokenizer, exactly reversible |
| 06 | [Tokenization: tiktoken](06_tokenization_tiktoken.ipynb) | Tokens per word for English, German and code; a cost-and-limit check before the call |
| 07 | [Stemming](07_stemming.ipynb) | refunds/refunded/refunding → `refund`; "universal" = "university" (the cost) |
| 08 | [Lemmatization](08_lemmatization.ipynb) | charged → charge, were → be; a readable "top issues" list |
| 09 | [Stopwords](09_stopwords.ipynb) | Default removal deletes "not"; a list with negations kept |
| 10 | [TF-IDF](10_tfidf.ipynb) | idf of "the" vs "charge"; the short on-topic article wins the search |
| 11 | [BM25](11_bm25.ipynb) | "refund" × 20 loses to a real refund article — saturation |
| 12 | [n-grams](12_ngrams.ipynb) | "not working" as a feature; unigrams vs bigrams measured on real posts |
| 13 | [NER](13_ner.ipynb) | ORG, MONEY, DATE, GPE, PERSON out of a ticket and into a CRM record |
| 14 | [POS tagging](14_pos_tagging.ipynb) | The verbs are the requested actions: cancel ×3, refund, update |
| 15 | [Text classification with sklearn](15_text_classification_sklearn.ipynb) | 20 Newsgroups routed at ~86% F1, 0.08 ms per ticket, confidence threshold, 2 MB pickle |
| 16 | [Word2Vec](16_word2vec.ipynb) | Trained on 3,000 posts: engine → torque, oil; orbit → comet, spacecraft |
| 17 | [GloVe](17_glove.ipynb) | 400k pretrained words; king − man + woman ≈ queen |
| 18 | [FastText](18_fasttext.ipynb) | "motorcyle", "orbitting", "spacecrafts" get vectors though never seen |
| 19 | [Break → Fix](19_break_fix.ipynb) | `fit_transform` in serving code: 86% offline → 34% in production → one shipped pipeline |
