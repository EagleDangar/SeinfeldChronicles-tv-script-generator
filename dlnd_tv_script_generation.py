#!/usr/bin/env python
# coding: utf-8

# # TV Script Generatio
# load in data
import helper

from collections import Counter

import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    int_to_vocab = dict(enumerate(set(text),1))
    vocab_to_int = { v:k for k,v in int_to_vocab.items() }
    
    # return tuple
    return (vocab_to_int, int_to_vocab)


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    punctuation_to_token = dict()
    punctuation_to_token["."] = "<period>"
    punctuation_to_token[","] = "<comma>"
    punctuation_to_token['"'] = "<quotation>"
    punctuation_to_token[";"] = "<semicolon>"
    punctuation_to_token["!"] = "<exclamation>"
    punctuation_to_token["?"] = "<question>"
    punctuation_to_token["("] = "<leftparent>"
    punctuation_to_token[")"] = "<rightparent>"
    punctuation_to_token["-"] = "<dash>"
    punctuation_to_token["\n"] = "<return>"
        
    return punctuation_to_token


def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    # TODO: Implement function
    
    n_batches = len(words)//batch_size
    # only full batches
    words = words[:n_batches*batch_size]
    
    input_words = []
    target_words = []
    for idx in range(len(words)-sequence_length):
        input_words.append(words[idx : idx + sequence_length])
        target_words.append(words[idx + sequence_length])
        
    input_words = torch.from_numpy(np.asarray(input_words))
    target_words = torch.from_numpy(np.asarray(target_words))
    
    dataset = TensorDataset(input_words,target_words)
    
    data_loader = DataLoader(dataset, batch_size=batch_size)
    # return a dataloader
    return data_loader


class RNN(nn.Module):
    
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
        """
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """
        super(RNN, self).__init__()
        # TODO: Implement function
        
        # set class variables
        self.output_size = output_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # define model layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first = True)
        self.fc = nn.Linear(hidden_dim, output_size)
    
    
    def forward(self, nn_input, hidden):
        """
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """  
        
        batch_size = nn_input.size(0)
        
        # embeddings and lstm_out
        vocab_embedding = self.embedding(nn_input)
        lstm_out, hidden = self.lstm(vocab_embedding, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        # fully-connected layer
        out = self.fc(lstm_out)
        
        # reshape into (batch_size, seq_length, output_size)
        out = out.view(batch_size, -1, self.output_size)
        # get last batch
        out = out[:, -1]
        

        # return one batch of output word scores and the hidden state
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        '''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        
        return hidden




def forward_back_prop(rnn, optimizer, criterion, inp, target, hidden):
    """
    Forward and backward propagation on the neural network
    :param decoder: The PyTorch Module that holds the neural network
    :param decoder_optimizer: The PyTorch optimizer for the neural network
    :param criterion: The PyTorch loss function
    :param inp: A batch of input to the neural network
    :param target: The target output for the batch of input
    :return: The loss and the latest hidden state Tensor
    """
    
    # TODO: Implement Function
    # move data to GPU, if available
    if(train_on_gpu):
        inp, target = inp.cuda(), target.cuda()
    
    hidden = tuple([each.data for each in hidden])
    
    # perform backpropagation and optimization
    
    ## intial gradient value to zero
    rnn.zero_grad()
    
    ## forward propagation on model
    output, hidden = rnn(inp, hidden)
    
    ## getting the loss and perform backward propagation
    loss = criterion(output.squeeze(),target)
    loss.backward()
    
    ## to deal with exploding gradient issue
    nn.utils.clip_grad_norm_(rnn.parameters(), 5)
    
    ## updating the gradients
    optimizer.step()

    # return the loss over a batch and the hidden state produced by our model
    return loss.item(), hidden




def train_rnn(rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
    batch_losses = []
    
    rnn.train()

    print("Training for %d epoch(s)..." % n_epochs)
    for epoch_i in range(1, n_epochs + 1):
        
        # initialize hidden state
        hidden = rnn.init_hidden(batch_size)
        
        for batch_i, (inputs, labels) in enumerate(train_loader, 1):
            
            # make sure you iterate over completely full batches, only
            n_batches = len(train_loader.dataset)//batch_size
            if(batch_i > n_batches):
                break
            
            # forward, back prop
            loss, hidden = forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)          
            # record loss
            batch_losses.append(loss)

            # printing loss stats
            if batch_i % show_every_n_batches == 0:
                print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                    epoch_i, n_epochs, np.average(batch_losses)))
                batch_losses = []

    # returns a trained rnn
    return rnn




def generate(rnn, prime_id, int_to_vocab, token_dict, pad_value, predict_len=100):
    """
    Generate text using the neural network
    :param decoder: The PyTorch Module that holds the trained neural network
    :param prime_id: The word id to start the first prediction
    :param int_to_vocab: Dict of word id keys to word values
    :param token_dict: Dict of puncuation tokens keys to puncuation values
    :param pad_value: The value used to pad a sequence
    :param predict_len: The length of text to generate
    :return: The generated text
    """
    rnn.eval()
    
    # create a sequence (batch_size=1) with the prime_id
    current_seq = np.full((1, sequence_length), pad_value)
    current_seq[-1][-1] = prime_id
    predicted = [int_to_vocab[prime_id]]
    
    for _ in range(predict_len):
        if train_on_gpu:
            current_seq = torch.LongTensor(current_seq).cuda()
        else:
            current_seq = torch.LongTensor(current_seq)
        
        # initialize the hidden state
        hidden = rnn.init_hidden(current_seq.size(0))
        
        # get the output of the rnn
        output, _ = rnn(current_seq, hidden)
        
        # get the next word probabilities
        p = F.softmax(output, dim=1).data
        if(train_on_gpu):
            p = p.cpu() # move to cpu
         
        # use top_k sampling to get the index of the next word
        top_k = 5
        p, top_i = p.topk(top_k)
        top_i = top_i.numpy().squeeze()
        
        # select the likely next word index with some element of randomness
        p = p.numpy().squeeze()
        word_i = np.random.choice(top_i, p=p/p.sum())
        
        # retrieve that word from the dictionary
        word = int_to_vocab[word_i]
        predicted.append(word)     
        
        # the generated word becomes the next "current sequence" and the cycle can continue
        current_seq = np.roll(current_seq.cpu(), -1, 1)
        current_seq[-1][-1] = word_i
    
    gen_sentences = ' '.join(predicted)
    
    # Replace punctuation tokens
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
    gen_sentences = gen_sentences.replace('\n ', '\n')
    gen_sentences = gen_sentences.replace('( ', '(')
    
    # return all the sentences
    return gen_sentences




if __name__ == '__main__':


    data_dir = './data/Seinfeld_Scripts.txt'
    text = helper.load_data(data_dir)

    # ## Pre-process all the data and save it
    # 
    # Running the code cell below will pre-process all the data and save it to file. You're encouraged to lok at the code for `preprocess_and_save_data` in the `helpers.py` file to see what it's doing in detail, but you do not need to change this code.


    # pre-process training data
    helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)

    int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()


    # Check for a GPU
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('No GPU found. Please use a GPU to train your neural network.')


    # Data params
    # Sequence Length
    sequence_length = 10  # of words in a sequence
    # Batch Size
    batch_size = 64

    # data loader - do not change
    train_loader = batch_data(int_text, sequence_length, batch_size)

    # Training parameters
    # Number of Epochs
    num_epochs = 10
    # Learning Rate
    learning_rate = 0.001

    # Model parameters
    # Vocab size
    vocab_size = len(vocab_to_int)
    # Output size
    output_size = vocab_size
    # Embedding Dimension
    embedding_dim = 350
    # Hidden Dimension
    hidden_dim = 300
    # Number of RNN Layers
    n_layers = 2

    # Show stats for every n number of batches
    show_every_n_batches = 500

    
    # defining loss and optimization functions for training
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # create model and move to gpu if available
    rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)

    if train_on_gpu:
        rnn.cuda()

    # training the model
    trained_rnn = train_rnn(rnn, batch_size, optimizer, criterion, num_epochs, show_every_n_batches)

    # saving the trained model
    helper.save_model('./modl/trained_rnn', trained_rnn)
    print('Model Trained and Saved')


    _, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
    trained_rnn = helper.load_model('./model/trained_rnn')

    # run the cell multiple times to get different results!
    gen_length = 400 # modify the length to your preference
    prime_word = 'george' # name for starting the script


    pad_word = helper.SPECIAL_WORDS['PADDING']
    generated_script = generate(trained_rnn, vocab_to_int[prime_word + ':'], int_to_vocab, token_dict, vocab_to_int[pad_word], gen_length)
    print(generated_script)



    # save script to a text file
    f =  open("generated_script_george.txt","w")
    f.write(generated_script)
    f.close()
