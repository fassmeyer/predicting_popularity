from collections import defaultdict
import numpy as np

class word2vec():
    def __init__(self):
        self.n = settings['n']
        self.lr = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']
        
    def generate_training_data(self, settings, corpus):
        # find unique word counts using dictionary
        word_counts = defaultdict(int)
        
        for row in corpus:
            for word in row:
                word_counts[word] += 1
        ## How mnay unique words in vocab?
        self.v_count = len(word_counts.keys())
        # Generate lookup dictionaries (vocab)
        self.words_list = list(word_counts.keys())
        # generate word:index
        self.word_index = dict((word, i) for i, word in enumerate(self.words_list))
        # generate index:word
        self.index_word = dict((i, word) for i, word in enumerate(self.words_list))
        
        training_data = []
        # Cycle through each sentence in corpus
        for sentence in corpus:
            sent_len = len(sentence)
            # Cycle through each word in sentence
            for i, word in enumerate(sentence):
                # convert target word to one-hot
                w_target = self.word2onehot(sentence[i])
                # cycle through context window
                w_context = []
                # Note: windw_size 2 will have range of 5 values
                for j in range(i-self.window, i+self.window+1):
                    # cirteria for context word
                    # 1. Target word cannot be context word (j != i)
                    # 2. Index must be greater or equal than 0 (j >= 0) - if not list index out of range
                    # 3. Index must be less or equal than length of sentence (j <= sent_len-1) - if not list index out of range 
                    if j != i and j <= sent_len-1 and j>= 0:
                        # Append the one-hot representation of word to w_context
                        w_context.append(self.word2onehot(sentence[j]))
                        # print(sentence[i], sentence[j])
                        # training_data contains a one-hot representation of the target word and the context word
                training_data.append([w_target, w_context])
        return np.array(training_data)
    
    
    def word2onehot(self, word):
        # word_vec - initialise a blank vector
        word_vec = [0 for i in range(0,self.v_count)] #alternative - np.zeros(self.v_count)
        # get ID of word from word_index
        word_index = self.word_index[word]
        # change value from 0 to 1 according to ID of word
        word_vec[word_index] = 1
        return word_vec
    
    
    
    def train(self, training_data):
        
        # initialising weight matrices
        self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n)) #shape: vocabulary size x dimensionality of embedding
        self.w2 = np.random.uniform(-1, 1, (self.n, self.v_count))
        # self.w1 = np.array(getW1)
        # self.w2 = np.array(getW2)
        
        # cycle through each epoch
        for i in range(self.epochs):
            # initialise loss to 0
            self.loss = 0
            
            # cycle through each training sample
            # w_t = vector for target word, w_c = vector for context word
            for w_t, w_c in training_data:
                # forward pass - pass in vector for target word (w_t) to get:
                # 1. predicted y using sofmax (y_pred) 2. matrix of hidden layer (h) 3. output layer 
                y_pred, h, u = self.forward_pass(w_t)
                
                # calculate error
                # 1. for a target word, calculate difference between y_pred and each of the context words
                # 2. sum up the differences using np.sum to give us the error for this particular target
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis = 0)
                
                # backpropagation
                # SGD to backpropagate errors - calculate loss on the output layer
                self.backprop(EI, h, w_t)
                
                # Calculate loss
                # There are 2 parts to the loss function
                # Part 1: -ve sum of all the output +
                # Part 2: length of context words * log of sum for all elements (exponential-ed) in the output layer before softmax (u)
                # Note: word.index(1) returns the index in the context word vector with value 1
                # Note: u[word.index(1)] returns the value of the output layer before softmax
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
                
            print('Epoch:', i, "Loss:", self.loss)
        
        
        
    def backprop(self, e, h, x):
        # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.outer.html
        # Column vector EI represents row-wise sum of prediction errors across each context word for the current center word
        # Going backwards, we need to take derivative of E with respect of w2
        # h - shape 10x1, e - shape 9x1, dl_dw2 - shape 10x9
        dl_dw_2 = np.outer(h, e)
        # x - shape 1x8, w2 - 5x8, e.T - 8x1
        # x - 1x8, np.dot() - 5x1, dl_dw1 - 8x5
        dl_dw_1 = np.outer(x, np.dot(self.w2, e.T))
        # Update weights
        self.w1 = self.w1 - (self.lr * dl_dw_1)
        self.w2 = self.w2 - (self.lr * dl_dw_2)
                
                
                
    def forward_pass(self, x):
        # x is one-hot vector for target word - shape |v| x 1
        # through first matrix (w1) to get hidden layer --> d x 1
        h = np.dot(self.w1.T, x)
        # dot product hidden layer with second matrix
        u = np.dot(self.w2.T, h)
        # force each element to range [0,1]
        y_c = self.softmax(u)
        return y_c, h, u
    
    
    def softmax(self, x):
        e_x = np.exp(x - np.max(x)) #why np.max?
        return e_x / e_x.sum(axis=0)
    
    
    # Get vector from word
    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w
    
    
    # Input vector, returns nearest word(s)
    def vec_sim(self, word, top_n):
        v_w1 = self.word_vec(word)
        word_sim = {}

        for i in range(self.v_count):
            # Find the similary score for each word in vocab
            v_w2 = self.w1[i]
            theta_sum = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_sum / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)

        for word, sim in words_sorted[:top_n]:
            print(word, sim)
        