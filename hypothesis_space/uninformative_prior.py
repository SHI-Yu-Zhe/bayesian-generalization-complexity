import itertools
from numpy import matrix

import torch

from attribute_representativeness import Attribute_Representativeness


class Uninformative_Prior(object):
    '''
    Class for implementing the uninformative prior.

    @Arguments:

        AR_row [torch.tensor]: p_a_given_c, row space matrix in C * D (concept numbers * dimension); Note: to use AR as the informative prior, we must take the entire AR matrix of the visual world;

        AR_col [torch.tensor]: sum_p_comma_given, column space matrix in C * D (concept numbers * dimension); Note: to use AR as the informative prior, we must take the entire AR matrix of the visual world;

        AR [torch.tensor]: attribute representativeness matrix in C * D (concept numbers * dimension); Note: to use AR as the informative prior, we must take the entire AR matrix of the visual world;
    '''
    def __init__(
        self, 
        AR_row:torch.tensor,
        AR_col:torch.tensor,
        AR:torch.tensor,
    ):
        self.__AR_row = AR_row
        self.__AR_col = AR_col
        self.__AR = AR

        self.__epsilon = 1e-12
        return

    @property
    def AR(self):
        '''
        Getter for attribute representativeness matrix.
        '''
        return self.__AR

    def sampling(self, matrix_row:torch.tensor, raw_matrix:torch.tensor, novel_example:torch.tensor):
        '''
        Sampling function.

        @Input:

            matrix_row [torch.tensor]: p_a_given_c, row space matrix in C * D (concept numbers * dimension); Note: to use AR as the informative prior, we must take the entire AR matrix of the visual world;

            raw_matrix [torch.tensor]: raw embedding matrix; Note: to use raw embedding matrix, we must take the entire AR matrix of the visual world; Note: matrix_row and raw_matrix must in the same shape;

            novel_example [torch.tensor]: a set of novel example (embedding vector) to generalize, in vector B * D (batchsize * dimension);

        @Output:

            sample_dict [dict{key[tuple], value[tuple]}]: prior for importance sampling, key: quardruple of concept ids (x_id, y_id, z_id), value (relateness importance for 3 hypotheses and the novel example, analogy importance for 3 hypotheses and the novel example);
        '''
        # get all concept ids
        current_concepts = int(self.__AR.shape[0])
        sample_quadruples = itertools.combinations(range(current_concepts), 3)

        newAR = Attribute_Representativeness(
            matrix_row = matrix_row, 
            raw_matrix = raw_matrix,
        )
        # add novel example to matrix row and raw matrix
        newAR.matrix_row = novel_example
        newAR.calc_matrix_col()
        newAR.calc_matrix_AR()

        self.__AR_row = newAR.matrix_row
        self.__AR_col = newAR.matrix_col 
        self.__AR = newAR.matrix

        d_relatedness, d_analogy = self.__P_hatC()

        w_r, w_a = d_relatedness[-1], d_analogy[-1]

        sample_dict = dict()

        for sq in sample_quadruples:
            x_r, y_r, z_r = d_relatedness[sq[0]], d_relatedness[sq[1]], d_relatedness[sq[2]]
            x_a, y_a, z_a = d_analogy[sq[0]], d_analogy[sq[1]], d_analogy[sq[2]]
            
            sample_dict[sq] = ((x_r * y_r * z_r * w_r) / (x_r + y_r + z_r + w_r), (x_a * y_a * z_a * w_a) / (x_a + y_a + z_a + w_a))
            
        return sample_dict

    def __P_hatC(self):
        '''
        Calculate uninformative prior for hatC.

        @Output:

            d_relatedness: uninformative prior for generalize by relatedness;

            d_analogy: uninformative prior for generalize by analogy;
        '''
        # sum_z and log P(z|c)
        ent_c = (- self.__AR_row * torch.log(self.__AR_row)).sum(dim=1)
        # standardize
        d_ent_c = ent_c - ent_c.mean()

        # sum_z and log P(c'|z)
        ent_c_comma = (- self.__AR_col * torch.log(self.__AR_col)).sum(dim=1)
        # standardize
        d_ent_c_comma = ent_c_comma - ent_c_comma.mean()

        # relatedness density
        d_relatedness = torch.exp(d_ent_c)
        d_relatedness = d_relatedness / (d_relatedness.sum() + self.__epsilon)
        # analogy density
        d_analogy = torch.exp(- d_ent_c)
        d_analogy = d_analogy / (d_analogy.sum() + self.__epsilon)

        return d_relatedness, d_analogy
    
