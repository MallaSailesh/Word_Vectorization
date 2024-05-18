# Word Embedding Comparison: SVD vs. Skip Gram with Negative Sampling

This project involves implementing and comparing word embedding models using Singular Value Decomposition (SVD) and Skip Gram with Negative Sampling. The analysis focuses on discerning differences in the quality of embeddings produced and their effectiveness in downstream tasks.

## 1. Introduction

Many NLP systems employ modern distributional semantic algorithms, known as word embedding algorithms, to generate meaningful numerical representations for words. These algorithms aim to create embeddings where words with similar meanings are represented closely in a mathematical space. Word embeddings fall into two main categories: frequency-based and prediction-based.

- **Frequency-based embeddings**: Utilize vectorization methods such as Count Vector, TF-IDF Vector, and Cooccurrence Matrix.
- **Prediction-based embeddings**: Exemplified by Word2Vec, utilize models like Continuous Bag of Words (CBOW) and Skip-Gram (SG).

## 2. Training Word Vectors

### 2.1 Singular Value Decomposition (SVD)

Implemented a word embedding model and trained word vectors by first building a Co-occurrence Matrix followed by the application of SVD.

### 2.2 Skip Gram with Negative Sampling

Implemented the Word2Vec model and trained word vectors using the Skip Gram model with Negative Sampling.

## 3. Corpus

Train the model on the given CSV files linked here: [News Classification Dataset](https://iiitaphyd-my.sharepoint.com/personal/advaith_malladi_research_iiit_ac_in/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fadvaith%5Fmalladi%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FiNLP%5FA2%2FiNLP%2DA2%2Ezip&parent=%2Fpersonal%2Fadvaith%5Fmalladi%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FiNLP%5FA2&ga=1).

**Note**: Used the Description column of the `train.csv` for training word vectors. The label/index column is used for the downstream classification task.

## 4. Downstream Task

After successfully creating word vectors using the above two methods, evaluated the word vectors by using them for a downstream classification task. Used the same RNN and RNN hyperparameters across vectorization methods for the downstream task.

## 5. Analysis

Compared and analyzed which of the two word vectorizing methods performs better using performance metrics such as accuracy, F1 score, precision, recall, and the confusion matrix on both the train and test sets. Wrote a detailed report on why one technique might perform better than the other, including the possible shortcomings of both techniques (SVD and Word2Vec).

## 6. Hyperparameter Tuning

Experimented with three different context window sizes. Reported performance metrics for all three context window configurations. Mentioned which configuration performs the best and discussed possible reasons for it.

## Execution

To execute any file, use:
```sh
python3 <filename>
```

To load the pretrained models:
```sh
torch.load("<filename>.pt")
```

## Loading Pretrained Models

### Word Embeddings

Loading `svd-word-vectors.pt` and `skip-gram-word-vectors.pt` gives us a dictionary. From this dictionary, we can access:
- `words_to_ind` using `dic["words_to_ind"]`
- `word_embeddings` using `dic["word_embeddings"]`

To get the word embedding for a token:
1. Get the index (`idx`) using `words_to_ind[token]`.
2. Get the word embedding using `word_embeddings[idx]`.

### Classification Models

Loading `svd-classification-model.pt` and `skip-gram-classification-model.pt` gives us a model which provides the class index when given a sentence.
- `svd-classification` means the model is trained using word embeddings obtained by the SVD method.
- Similarly, `skip-gram-classification` refers to the model trained using word embeddings obtained by the Skip Gram with Negative Sampling method.

## Links to .pt Files

1. [svd-word-vectors.pt](https://drive.google.com/file/d/1nP2wy3ZoRSUJAYbYV-E-kOoqWapScju7/view?usp=sharing)
2. [skip-gram-word-vectors.pt](https://drive.google.com/file/d/1xixFBTIy1apjy-x9P9UJrxHgHPe3iYXY/view?usp=sharing)
3. [svd-classification-model.pt](https://drive.google.com/file/d/1gs2DzwtFox2FgpyWM82YoZBWiLduiY8V/view?usp=sharing)
4. [skip-gram-classification-model.pt](https://drive.google.com/file/d/1AW5UQOqZYfYFk8-iQaYHjgylb_CXKRxY/view?usp=sharing)



