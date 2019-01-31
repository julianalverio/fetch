import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np

torch.manual_seed(1)


def getData():
    vocab = ["pickup", "putdown", "stack", "grab"]
    training_sentences = []
    for _ in range(16):
        training_sentences.append(random.choices(vocab, k=4))
    validation_sentences = []
    for _ in range(8):
        validation_sentences.append(random.choices(vocab, k=4))
    return training_sentences, validation_sentences


def prepareInputs(training_data):
    word_to_ix = {}
    for sentence in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    prepared_training_data = []
    for sentence in training_data:
        sequence = []
        for word in sentence:
            sequence.append(word_to_ix[word])
        prepared_training_data.append(torch.tensor(sequence))
    return prepared_training_data


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, vocab_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, prepared_sentence):
        embeds = self.word_embeddings(prepared_sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


if __name__ == '__main__':
    training_data, validation_data = getData()
    training_sequence, validation_sequence = prepareInputs(training_data), prepareInputs(validation_data)
    EMBEDDING_DIM = 5
    HIDDEN_DIM = 5
    VOCAB_SIZE = 4
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(22):
        for sentence in training_sequence:
            model.zero_grad()
            model.hidden = model.init_hidden()
            tag_scores = model(sentence)
            loss = loss_function(tag_scores, sentence)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        training_scores = []
        for sentence in training_sequence:
            model.hidden = model.init_hidden()
            tag_scores = model(sentence)
            predictions = torch.argmax(tag_scores, dim=1)
            score = torch.sum(torch.eq(predictions, sentence)).item() / 4.
            training_scores.append(score)
        print('Mean Training Score:', np.mean(training_scores))

        validation_scores = []
        for sentence in validation_sequence:
            model.hidden = model.init_hidden()
            tag_scores = model(sentence)
            predictions = torch.argmax(tag_scores, dim=1)
            score = torch.sum(torch.eq(predictions, sentence)).item() / 4.
            validation_scores.append(score)
        print('Mean Validation Score', np.mean(validation_scores))
