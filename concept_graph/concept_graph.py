import numpy as np


class Concept_Graph(object):
    '''
    Class for representing concepts in a graph.
    '''
    def __init__(self):

        # self.__raw_matrix = np.load('./graph/vocab_distribution.npy')
        self.__raw_matrix = np.random.rand(30522, 30522)
         
        with open('./graph/vocab.txt', 'r') as vb:
            self.__raw_vocab = vb.readlines()
        vb.close()

        self.__raw_vocab = np.array([
            v[:-1]
            for v 
            in self.__raw_vocab
        ])
        return 

    @property
    def raw_matrix(self):
        return self.__raw_matrix

    @property
    def raw_vocab(self):
        return self.__raw_vocab

    def query(self, concept:str, show_n:int):
        '''
        @Arguments:

            concept [str]: query concept;

            show_n [int]: showing the n attributes with highest contribution to the concept;
        '''
        vocab_idx, = np.where(self.__raw_vocab == concept)
        vocab_idx = vocab_idx[0]
        # select the row according to vocab_idx
        vocab_dist = self.__raw_matrix[vocab_idx, :]
        
        # rank top-n
        ranked_vocab_dist = np.sort(vocab_dist).tolist()[::-1]
        ranked_vocab_idx = np.argsort(vocab_dist).tolist()[::-1]

        ranked_vocab_dist = ranked_vocab_dist[:show_n]
        ranked_vocab_idx = ranked_vocab_idx[:show_n]

        for i in range(len(ranked_vocab_dist)):
            print('{},\t{:.4f}\n'.format(self.__raw_vocab[ranked_vocab_idx[i]], ranked_vocab_dist[i]))
        return


