import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

EMBEDDING_DIM = 5
HIDDEN_DIM = 5
VOCAB_SIZE = 4
TRAINING_DATA = ["pickup", "putdown", "stack", "grab"]

def prepareInputs(training_data):
    word_to_ix = {}
    for word in training_data:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    return prepare_sequence(training_data, word)


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, sentences):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, vocab_size)
        self.hidden = self.init_hidden()
        self.sentences = sentences

        self.word_to_ix = {}
        for word in self.sentences:
            if word not in self.word_to_ix:
                self.word_to_ix[word] = len(self.word_to_ix)

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def makeWord2Idx(self, sentences_list):
        word_to_ix = {}
        for sentence in sentences_list:
            words = sentence.split()
            for word in words:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
        self.word_to_ix = word_to_ix

    def preprocessSentence(self, sentences_list):
        sentence_tensors = []
        for word in sentences_list:
            word_tensor = torch.tensor(self.word_to_ix[word[0]]).cuda()
            sentence_tensors.append(word_tensor)
        sentences = torch.cat(sentence_tensors).view(len(self.sentences), 1, -1)
        return sentences

    def forward(self, sentence):
        prepared_sentence = self.preprocessSentence(sentence)
        embeds = self.word_embeddings(prepared_sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

######################################################################
# Train the model:


model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SIZE)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(300):
    for sentence, tags in TRAINING_DATA:
        model.zero_grad()
        model.hidden = model.init_hidden()

        # sentence_in = prepare_sequence(sentence, word_to_ix)
        # targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(TRAINING_DATA)

        prepared = model.preprocessSentence(TRAINING_DATA)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
    # for word i. The predicted tag is the maximum scoring tag.
    # Here, we can see the predicted sequence below is 0 1 2 0 1
    # since 0 is index of the maximum value of row 1,
    # 1 is the index of maximum value of row 2, etc.
    # Which is DET NOUN VERB DET NOUN, the correct sequence!
    print(tag_scores)


######################################################################
# Exercise: Augmenting the LSTM part-of-speech tagger with character-level features
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the example above, each word had an embedding, which served as the
# inputs to our sequence model. Let's augment the word embeddings with a
# representation derived from the characters of the word. We expect that
# this should help significantly, since character-level information like
# affixes have a large bearing on part-of-speech. For example, words with
# the affix *-ly* are almost always tagged as adverbs in English.
#
# To do this, let :math:`c_w` be the character-level representation of
# word :math:`w`. Let :math:`x_w` be the word embedding as before. Then
# the input to our sequence model is the concatenation of :math:`x_w` and
# :math:`c_w`. So if :math:`x_w` has dimension 5, and :math:`c_w`
# dimension 3, then our LSTM should accept an input of dimension 8.
#
# To get the character level representation, do an LSTM over the
# characters of a word, and let :math:`c_w` be the final hidden state of
# this LSTM. Hints:
#
# * There are going to be two LSTM's in your new model.
#   The original one that outputs POS tag scores, and the new one that
#   outputs a character-level representation of each word.
# * To do a sequence model over characters, you will have to embed characters.
#   The character embeddings will be the input to the character LSTM.
#
