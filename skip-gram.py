import pandas as pd , numpy as np , random, time
import spacy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


train_data = pd.read_csv("./corpus/train.csv")
test_data = pd.read_csv("./corpus/test.csv")

train_data_length = len(train_data['Description'][:15000]) # modify it later

nlp = spacy.load('en_core_web_sm')

tokenized_sentences = []
vocab = []

for i in range(train_data_length):
    sentence = train_data["Description"][i]
    doc = nlp(sentence)
    tokens = [token.text for token in doc if token.text.strip() != ''] # removing spaces
    tokens = ["<start>"] + tokens + ["<end>"]
    tokenized_sentences.append(tokens)
    vocab.extend(tokens)

vocab = list(set(vocab))
vocab_size = len(vocab)
print(vocab_size)
words_to_ind = {word: i for i, word in enumerate(vocab)}



class SkipGramModel_NegativeSampling(nn.Module):

    def __init__(self, tokenized_sentences, window_size, embedding_length, number_of_negative_samples):
        super().__init__()

        self.neighbours = self.get_neighbours(tokenized_sentences, window_size)

        self.data , self.target = self.get_data(number_of_negative_samples)

        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
        # Initialise the matrix with random values 
        self.embedding_matrix = nn.Embedding(vocab_size, embedding_length).to(self.device)
         # fills the data uniformly between the values -1 and 1
        self.embedding_matrix.weight.data.uniform_(-1, 1)
        self.context_matrix = nn.Embedding(vocab_size, embedding_length).to(self.device)
        self.context_matrix.weight.data.uniform_(-1, 1) 

        self.criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits Loss
        self.optimizer = torch.optim.Adam(list(self.embedding_matrix.parameters()) + list(self.context_matrix.parameters())) 

    # Get the neighbours of all words
    def get_neighbours(self, tokenized_sentences, window_size):
        neighbours = {}
        for sentence in tokenized_sentences:
            for i, word in enumerate(sentence):
                for j in range(max(0, i-window_size), min(len(sentence), i+window_size+1)):
                    if i!=j: 
                        if words_to_ind[word] not in neighbours.keys():
                            neighbours[words_to_ind[word]] = [words_to_ind[sentence[j]]]
                        else:
                            neighbours[words_to_ind[word]].append(words_to_ind[sentence[j]])
        return neighbours

    # Generate Positive anf Negative samples
    def get_data(self, n):
        # list of lists with 1 positive samples followed by n negative samples for each instance of a    word
        generated_data = [] 
        target = []

        for idx, neighbours in self.neighbours.items(): 
            excluded_list = np.array([idx] + neighbours)
            for value in neighbours:
                generated_data.append([idx, value]) # positive sample
                target.append(1)
                for _ in range(n):
                    rndm_num = idx
                    while rndm_num in excluded_list:
                        rndm_num = random.randrange(0, vocab_size) 
                    generated_data.append([idx, rndm_num]) # negative sample
                    target.append(0)
        
        return np.array(generated_data), np.array(target)

    def train(self, epochs, batch_size=1024):

        # Convert numpy array to pytorch tensors
        data_tensor = torch.tensor(self.data, dtype=torch.long)
        target_tensor = torch.tensor(self.target, dtype=torch.float)

        # Create DataLoader for batching 
        dataset =  TensorDataset(data_tensor,  target_tensor)
        dataLoader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs): 
            print(f"Epoch: {epoch}")
            for batch_data, batch_target in dataLoader:

                batch_data = batch_data.to(self.device)
                batch_target = batch_target.to(self.device)
                word1_index = batch_data[:, 0]
                word2_index = batch_data[:, 1]

                # Get embeddings
                word1_embedding = self.embedding_matrix(word1_index)
                word2_context = self.context_matrix(word2_index)

                # Calculate dot product and apply sigmoid
                dot_product = torch.sum(word1_embedding * word2_context, dim=1)
                prediction = torch.sigmoid(dot_product)

                # Calculate loss
                loss = self.criterion(prediction, batch_target)

                # Backpropagation and update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


model = SkipGramModel_NegativeSampling(tokenized_sentences, 5, 300, 5)
start_time = time.time()
model.train(5)
end_time = time.time()
print(f"Time taken: {end_time - start_time}")
word_embeddings = model.embedding_matrix.weight.data.numpy()
print(word_embeddings)

data_to_save = {
    'word_embeddings': word_embeddings,
    'words_to_ind': words_to_ind
}
torch.save(data_to_save, 'skip-gram-word-vectors.pt')

    