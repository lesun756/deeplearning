def build_dictionary(corpus_raw):
    words = []
    # TO DO
	
    tempset = set([])
    
    for line in corpus_raw:
        temp = line.split(" ")
        for word in temp:
            singleset = set([])
            if word != "":
                singleset.add(word)
                # print(tempset)
                # print(singleset)
                # print("------")
                if singleset.issubset(tempset):
                    continue
                else:
                    tempset.add(word)
                    words.append(word)
    
    # print(words)

    # END TO DO
    return set(words)
	
def one_hot_encoding(data_point_index, vocab_size):
    # TO DO

    temp = np.zeros(vocab_size, dtype = int)
    temp[data_point_index] = 1

    # END TO DO
    return temp
	
def build_word_index_mapping(corpus_dict):   
    # TO DO
    
    ind_2_word = []
    value_list = []
    index = 0
    for word in corpus_dict:
        ind_2_word.append(word)
        value_list.append(index)
        index = index + 1
    
    word_2_ind = dict((key, value) for (key, value) in zip(ind_2_word, value_list))
    

    # END TO DO
    return word_2_ind, ind_2_word
	
def build_skip_pair(window_size, sentences):
    # TO DO
    
    data = []
    
    for line in sentences:
        
        words = []
        temp = line.split(" ")
        
        # words in one sentence
        for word in temp:
            if word != "":
                words.append(word)
                
        # build pairs for one sentence
        length = len(words)
        for i in range(length):
            pair = []
            for j in range(window_size):
                if (i-(window_size-j)) >= 0:
                    pair.append(words[i])
                    pair.append(words[i-(window_size-j)])
                    data.append(pair)
                    pair = []
            
            for j in range(window_size):
                if (i+(j+1)) < length:
                    pair.append(words[i])
                    pair.append(words[i+(j+1)])
                    data.append(pair)
                    pair = []
            
    
        

    # END TO DO
    return data
	
	
class MyEmbeddingModel(Model):
  def __init__(self, embedding_size, vocab_size):
    super(MyEmbeddingModel, self).__init__()
    #Example:
        #self.d2 = Dense(embedding_size)
        #self.d3 = Dense(vocab_size, activation = 'softmax')
    
    # TO DO
	self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size]))


    # END TO DO

  def call(self, x):
    #Example:
        #x_2 = self.d2(x_1)
        #x_3 = self.d3(x_2)
        
    # TO DO

	return tf.nn.embedding_lookup(self.embeddings, x)

    # END TO DO
    # return x_2, x_3



