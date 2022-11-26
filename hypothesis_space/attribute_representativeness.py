import os
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as f


class Attribute_Representativeness(object):
    '''
    Class Attribute Representativeness Matrix.

    @Arguments:

        matrix_row [torch.tensor] (optional);

        raw_matrix [torch.tensor] (optional);

    @Public Properties:

        matrix;

        matrix_row;

        matrix_col;

        AR_conceptwise_entropy;

        AR_conceptwise_max;

        AR_conceptwise_mean;

        AR_conceptwise_var;

    @Public Methods:

        matrix_row(torch.tensor): update matrix row;

        calc_matrix_col(): calculate column matrix;

        calc_matrix_AR(): calculate AR matrix;

        calc_embedding_complexity(): calculate Kolmogorov Complexity of embeddings;

        save_csv(): save matrices respectively;
    '''
    def __init__(self, matrix_row=torch.tensor(0), raw_matrix=None):
        self.__matrix = None
        self.__log_matrix = None
        self.__matrix_row = matrix_row
        self.__matrix_col = None
        self.__raw_matrix = raw_matrix

        self.__AR_conceptwise_entropy = None 
        self.__AR_conceptwise_max = None
        self.__AR_conceptwise_mean = None
        self.__AR_conceptwise_var = None
        self.__k_complexity = None

        self.__epsilon = 1e-12
        return 

    @property
    def matrix(self):
        '''
        AR Matrix Getter.
        '''
        return self.__matrix

    @property
    def log_matrix(self):
        '''
        Log AR Matrix Getter.
        '''
        return self.__log_matrix

    @property
    def matrix_row(self):
        '''
        Matrix Row Space Getter.
        '''
        return self.__matrix_row

    @property
    def matrix_col(self):
        '''
        Matrix Column Space Getter.
        '''
        return self.__matrix_col

    @property
    def raw_matrix(self):
        '''
        Raw Matrix Getter.
        '''
        return self.__raw_matrix

    @property
    def AR_conceptwise_entropy(self):
        '''
        Conceptwise entropy of AR.
        '''
        return self.__AR_conceptwise_entropy

    @property
    def AR_conceptwise_max(self):
        '''
        Conceptwise maximum of AR.
        '''
        return self.__AR_conceptwise_max

    @property
    def AR_conceptwise_mean(self):
        '''
        Conceptwise mean of AR.
        '''
        return self.__AR_conceptwise_mean

    @property
    def AR_conceptwise_var(self):
        '''
        Conceptwise variance of AR.
        '''
        return self.__AR_conceptwise_var

    @matrix_row.setter
    def matrix_row(self, actvec:torch.tensor):
        '''
        Matrix Row Space Setter. Update Matrix Row Space swith a (set of) activation vectors. Please note that the set of activation vectors must be given samples in the same concept.

        [Usage]: Attribute_Representativeness_instance.matrix_row = actvec

        @Input:

            actvec [torch.tensor]: B * D * H * W; 

                B: number of samples in the current batch (B >= 1);

                D: dimension (channel) of the activation vectors (dimension of the attribute space) (C > 1);

                H, W: height and width of the feature map for a dimension of the activation vectors; 
        '''
        # get embedding vector
        # embedvec: B * D
        embedvec = f.adaptive_max_pool2d(actvec, output_size=(1,1)).reshape((actvec.shape[0], -1))

        # embedvec: 1 * D
        concept_embedvec = embedvec.mean(dim=0).unsqueeze(0)

        # normalize vector to obtain probabilistic distribution P(a|c): 1 * D
        p_a_given_c = concept_embedvec / (concept_embedvec.sum() + self.__epsilon)
        
        if self.__matrix_row.shape == torch.tensor(0).shape:
            # initialize matrix row
            self.__matrix_row = p_a_given_c
            # initialize raw matrix
            self.__raw_matrix = concept_embedvec
        else:
            # matrix row is initialized
            self.__matrix_row = torch.vstack((self.__matrix_row, p_a_given_c))
            # raw matrix is initialized
            self.__raw_matrix = torch.vstack((self.__raw_matrix, concept_embedvec))
        return

    def calc_matrix_col(self):
        '''
        Matrix Row Space Calculator. Please note that this function should only be called after all matrix rows (and the raw matrix) are processed.
        '''
        # Note: use raw matrix instead of matrix column
        tmp = self.__raw_matrix

        # normalize over columns for P(c'|a)
        p_c_comma_given_a = tmp / (tmp.sum(dim=0) + self.__epsilon)

        # obtain \sum_{c'\neq c} P(c'|a): |C| * D
        self.__matrix_col = 1 - p_c_comma_given_a
        return

    def calc_matrix_AR(self):
        '''
        Attriute Representativeness Matrix Calculator. Please note that this function should only be called after all matrix columns are processed.
        '''
        # R(a,c)
        self.__matrix = self.__matrix_row / (self.__matrix_col + self.__epsilon)

        # log R(a,c)
        self.__log_matrix = torch.log(self.__matrix)
        return

    def AR_stat(self):
        '''
        Second statistics over Attribute Representativeness Matrix.
        '''
        mat = self.__matrix

        self.__AR_conceptwise_entropy = - (mat * torch.log(mat)).sum(dim=1)

        self.__AR_conceptwise_max, _ = torch.max(mat, dim=1)

        self.__AR_conceptwise_mean = mat.mean(dim=1)

        self.__AR_conceptwise_var = mat.var(dim=1)
        return

    @property
    def k_complexity(self):
        '''
        Embedding Kolmogorov Complexity Getter.
        '''
        return self.__k_complexity

    def calc_embedding_complexity(self):
        '''
        Embedding Kolmogorov Complexity calculator. Here we treat the shannon entropy, i.e., E[log P(a|c)] as the indicator. Please note that this function should only be called after all matrix rows are processed.
        '''
        p_a_given_c = self.__matrix_row

        # shannon entropy E_{P(a|c)}[log P(a|c)]
        self.__k_complexity = - (p_a_given_c * torch.log(p_a_given_c)).sum(dim=1)
        return 

    def save_csv(self, labels:Union[list,dict], dataset):
        '''
        Save matrices to csv files respectively.

        @Input:

            labels [list or dictionary]: a list with ordered labels or a dictionary with order labels mapping to indexes. The order should be monotonically increasing from 1 (1-indexed) or 0 (0-indexed). 
        '''
        np_matrix = self.__matrix.numpy()
        np_log_matrix = self.__log_matrix.numpy()
        np_matrix_row = self.__matrix_row.numpy()
        np_matrix_col = self.__matrix_col.numpy()

        np_AR_conceptwise_entropy = self.__AR_conceptwise_entropy.numpy()
        np_AR_conceptwise_max = self.__AR_conceptwise_max.numpy()
        np_AR_conceptwise_mean = self.__AR_conceptwise_mean.numpy()
        np_AR_conceptwise_var = self.__AR_conceptwise_var.numpy()

        if not os.path.isdir('./matrices'):
            os.mkdir('./matrices')

        df_matrix = pd.DataFrame(self.__name_col(np_matrix, labels))
        df_log_matrix = pd.DataFrame(self.__name_col(np_log_matrix, labels))
        df_matrix_row = pd.DataFrame(self.__name_col(np_matrix_row, labels))
        df_matrix_col = pd.DataFrame(self.__name_col(np_matrix_col, labels))

        df_AR_conceptwise_entropy = pd.DataFrame(self.__name_col(np_AR_conceptwise_entropy, labels))
        df_AR_conceptwise_max = pd.DataFrame(self.__name_col(np_AR_conceptwise_max, labels))
        df_AR_conceptwise_mean = pd.DataFrame(self.__name_col(np_AR_conceptwise_mean, labels))
        df_AR_conceptwise_var = pd.DataFrame(self.__name_col(np_AR_conceptwise_var, labels))

        df_matrix.to_csv(f'./matrices/{dataset}_matrix_AR.csv', index=False)
        df_log_matrix.to_csv(f'./matrices/{dataset}_log_matrix_AR.csv', index=False)
        df_matrix_row.to_csv(f'./matrices/{dataset}_matrix_row.csv', index=False)
        df_matrix_col.to_csv(f'./matrices/{dataset}_matrix_col.csv', index=False)

        df_AR_conceptwise_entropy.to_csv(f'./matrices/{dataset}_AR_conceptwise_entropy.csv', index=[0])
        df_AR_conceptwise_max.to_csv(f'./matrices/{dataset}_AR_conceptwise_max.csv', index=[0])
        df_AR_conceptwise_mean.to_csv(f'./matrices/{dataset}_AR_conceptwise_mean.csv', index=[0])
        df_AR_conceptwise_var.to_csv(f'./matrices/{dataset}_AR_conceptwise_var.csv', index=[0])
        return 

    def __name_col(self, mat:np.array, labels:Union[list,dict]):
        '''
        Name columns from matrix.

        @Input:

            mat [np.array]: matrix;

            labels [list or dictionary]: a list with ordered labels or a dictionary with order labels mapping to indexes. The order should be monotonically increasing from 1 (1-indexed) or 0 (0-indexed). 

        @Output:

            [dict] with named columns;
        '''
        datadict = dict()

        attributes = []
        concepts = []
        values = []

        c = 0        
        for label in labels:
            if len(mat.shape) > 1:
                # a row in the matrix
                row_c = mat[c, :]

                attr_id = 1
                for attr in row_c:
                    attributes.append(attr_id)
                    concepts.append(label)
                    values.append(attr)  

                    attr_id += 1
                              
            else:
                # a dimension in the vector, need to propagate to an array
                row_c = mat[c]
                
                attributes += [0]
                concepts.append(label)
                values.append(row_c)
            
            c += 1

        datadict = {
            'Attributes': attributes,
            'Concepts': concepts,
            'Values': values,
        }
        
        return datadict

