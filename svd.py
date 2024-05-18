import pandas as pd 
import numpy as np 
import spacy
from spacy.symbols import ORTH
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds 
import torch

train_data = pd.read_csv("./corpus/train.csv")
test_data = pd.read_csv("./corpus/test.csv")

train_data_length = len(train_data['Description'][:10000])

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer.add_special_case("<start>", [{ORTH: "<start>"}])
nlp.tokenizer.add_special_case("<end>", [{ORTH: "<end>"}])

tokenized_sentences = []
all_tokens = []

for i in range(train_data_length):
    sentence = "<start> " + train_data["Description"][i] + " <end>"
    doc = nlp(sentence)
    tokens = [token.text for token in doc if token.text.strip() != ''] # removing spaces
    tokenized_sentences.append(tokens)
    all_tokens.extend(tokens)

all_tokens = list(set(all_tokens))
print(len(all_tokens))
words_to_ind = {word: i for i, word in enumerate(all_tokens)}

# Now build the co-occurence matrix
co_occurence_matrix = np.zeros( (len(all_tokens), len(all_tokens)) )
window_size = 5
embedding_length = 300

for sentence in tokenized_sentences:
    for i, word in enumerate(sentence):
        for j in range(max(0, i-window_size), min(len(sentence), i+window_size+1)):
            if i!=j: 
                co_occurence_matrix[words_to_ind[word]][words_to_ind[sentence[j]]] += 1

co_occurence_matrix = csr_matrix(co_occurence_matrix)

# Perform SVD on co-occurence matrix 
U, S, V = svds(co_occurence_matrix, k=embedding_length)
word_embeddings = U
 
# save the word embeddings in svd-word-vectors.py 
data_to_save = {
    'word_embeddings': word_embeddings,
    'words_to_ind': words_to_ind
}
torch.save(data_to_save, 'svd-word-vectors.pt')