import torch


class K_Complexity_AR(object):
    '''
    Computing Kolmogorov Complexity by Attribute Representativeness Matrix.

    @Arguments:

        AR [torch.tensor]: attribute representativeness matrix in C * D (concept numbers * dimension);

    @UsageDEMO:
        kar = K_Complexity_AR(AR) # instantiate class
        for conceptid in concepts:
            concept, groundtruth = data[conceptid]
            conceptembed = f(concept) # get embedding vector in the batch of the concept
            kar.init_concept(conceptembed, conceptid)
            K = 1
            while True:
                pred = fc(conceptembed * kar.get_mask_K(K))
                acc = pred / groundtruth
                if acc >= thres:
                    print('Complexity for concept {} is {}.'.format(conceptid, K))
                    break

                K += 1
    '''
    def __init__(self, AR:torch.tensor):
        self.__AR = AR
        
        self.__curr_mask = None
        self.__curr_decorder = None
        return

    @property
    def AR(self):
        '''
        Getter for attribute representativeness matrix.
        '''
        return self.__AR

    @property
    def curr_mask(self):
        '''
        Getter for current mask.
        '''
        return self.__curr_mask

    @property
    def curr_decorder(self):
        '''
        Getter for current decreasing order.
        '''
        return self.__curr_decorder

    def init_concept(self, embed:torch.tensor, concept_id:int):
        '''
        Get P(z_1,z_2,dots,z_K) from embedding vector.

        @UsageHINT: initialize a concept with mask K=1. Call this function at the start of testing a concept.

        @Input:

            embed [torch.tensor]: a batch of embedding vector of concept c, in the size B * D (batchsize * dimension);

            concept_id[int]: id of concept (the row id), zero-indexed;

        @Output:

            z_mask [torch.tensor(0/1)]: a binary mask with the same size as embed;
        '''
        # initialize mask
        self.__curr_mask = torch.zeros(embed.shape)

        # get a row as attributes of concept c
        attrs = self.__AR[concept_id, :]

        # get order with descending order
        self.__curr_decorder = torch.argsort(attrs, dim=0, descending=True)
        return 

    def get_mask_K(self, K:int):
        '''
        Get mask for attribute length K.

        @Input:

            K [int]: length K;

        @Output:

            curr_mask [torch.tensor]: current mask with the same shape to embed;
        '''
        # get the list of attrids
        attrids = self.__curr_decorder[:K]

        # set attributes in attrids to 1 in the mask
        for id in attrids:
            # set the column for the attribute to 1.0
            self.__curr_mask[:, id] = 1.0
        
        return self.__curr_mask

    def get_masks(self, max_K:int):
        '''
        Get all masks for attribute length with maximum K.

        @Input:

            max_K [int]: maximum length K;

        @Output:

            final_mask [torch.tensor]: current mask with the same shape to embed; size: K * B * C, B * C is the original embedding vector, and K is a higher dimension with K masks;
        '''
        final_mask = torch.zeros(self.__curr_mask.shape).unsqueeze(0)
        single_mask = torch.zeros(self.__curr_mask.shape)
        
        for k in range(1, max_K + 1):
            attrids = self.__curr_decorder[:k]

            for id in attrids:
                single_mask[:, id] = 1.0

            if final_mask.shape[0] == 1:
                final_mask = single_mask.unsqueeze(0)
            else:
                final_mask = torch.vstack((final_mask, single_mask))
        
        return final_mask

