import sys

import numpy as np

from concept_graph import Concept_Graph

if __name__ == '__main__':
    g = Concept_Graph()
    
    # query argv[1]: concept, argv[2]: show_n
    g.query(sys.argv[1], int(sys.argv[2]))
    
