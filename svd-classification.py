import time
import pandas as pd , numpy as np 
import spacy
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

train_data = pd.read_csv("./corpus/train.csv")
test_data = pd.read_csv("./corpus/test.csv")

nlp = spacy.load('en_core_web_sm')

class RNN_SVD_Classifier(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(RNN_SVD_Classifier, self).__init__()

        self.word_embeddings, self.words_to_ind = self.load_embeddings()

        self.input_size = len(self.word_embeddings[0])
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.i2o = nn.Linear(self.input_size + self.hidden_size, self.output_size)
        self.sigmoid = nn.Sigmoid()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), 1e-5)

    def forward(self, input_tensor, hidden_tensor):

        combined = torch.cat((input_tensor, hidden_tensor), 0)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.sigmoid(output)
        return hidden, output  

    def load_embeddings(self):
        data_loaded = torch.load("svd-word-vectors.pt")
        word_embeddings = data_loaded["word_embeddings"]
        words_to_ind = data_loaded["words_to_ind"]
        return word_embeddings, words_to_ind

    def get_embedding(self, word):
        if word in self.words_to_ind.keys():
            return self.word_embeddings[self.words_to_ind[word]]
        else:
            return [0]*len(self.word_embeddings[0])

    def train(self, X_train, y_train, epochs=5):

        # get embeddings 
        X_train = [ np.array([self.get_embedding(token) for token in sentence], dtype=np.float32) for sentence in X_train]
        X_train_tensor = ([torch.tensor(sentence, dtype=torch.float32) for sentence in X_train])
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        start_time = time.time() 

        for epoch in range(epochs):

            total_loss = 0
            
            for i in range(len(y_train_tensor)):

                inputs = X_train_tensor[i]
                target = y_train_tensor[i]-1
                labels = [0]*(self.output_size)
                labels[target] = 1
                labels = torch.tensor(np.array(labels)).float()

                self.optimizer.zero_grad()

                cummulative_hidden = torch.zeros(self.hidden_size)
                cnt = 1
                for embedding in inputs: 
                    hidden, output = self.forward(embedding, cummulative_hidden)
                    cummulative_hidden = cummulative_hidden*cnt + hidden
                    cnt+=1
                    cummulative_hidden /= cnt

                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            end_time = time.time() 
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(y_train_tensor):.4f}, Time: {end_time-start_time}')
           
    def evaluate(self, X, y):

        X = [ np.array([self.get_embedding(token) for token in sentence], dtype=np.float32) for sentence in X]
        X_tensor = ([torch.tensor(sentence, dtype=torch.float32) for sentence in X])
        y_tensor = torch.tensor(y, dtype=torch.long)

        y_pred = []
            
        for i in range(len(y_tensor)):

            inputs = X_tensor[i]

            cummulative_hidden = torch.zeros(self.hidden_size)
            cnt = 1
            for embedding in inputs: 
                hidden, output = self.forward(embedding, cummulative_hidden)
                cummulative_hidden = cummulative_hidden*cnt + hidden
                cnt+=1
                cummulative_hidden /= cnt
            
            ind = np.argmax(output.detach().numpy())+1
            y_pred.append(ind)

        y_pred = np.array(y_pred)
        y_true = y_tensor.detach().numpy()

        print(classification_report(y_true, y_pred, zero_division=0))
        print(confusion_matrix(y_true, y_pred))


train_data = train_data[:15000]
test_data = test_data[:]
X_train = train_data["Description"].to_numpy()
# 1 or 2 or 3 or 4
y_train = train_data["Class Index"].to_numpy()
X_test = test_data["Description"].to_numpy()
y_test = test_data["Class Index"].to_numpy()


X_train_tokenized = []
for i in range(len(X_train)):
    doc = nlp(X_train[i])
    tokens = [token.text for token in doc if token.text.strip() != ''] # removing spaces
    X_train_tokenized.append(tokens)
X_test_tokenized = []
for i in range(len(X_test)):
    doc = nlp(X_test[i])
    tokens = [token.text for token in doc if token.text.strip() != ''] # removing spaces
    X_test_tokenized.append(tokens)

model = RNN_SVD_Classifier(64, 4)
model.train(X_train_tokenized, y_train)
model.evaluate(X_train_tokenized, y_train)
model.evaluate(X_test_tokenized, y_test)

data_to_save = {
    'model': model
}
torch.save(data_to_save, 'svd-classification-model.pt')

