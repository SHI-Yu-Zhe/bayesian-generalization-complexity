import math
import os
import time
from cProfile import label
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class VisEnt:
    def __init__(self):
        self.__exp = lambda x: math.e ** (x)
        self.__log = lambda x: np.log(x)
        return 

    def vis_all_entropy_pixel(self, data_concept_ents:dict):
        '''
        Visualize all entropies across datasets in pixel diagram.

        @Args:

            data_concept_ents [dict(key: dataset_name, value: list(tuple(concept_name, mean, var)))];
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.grid()
        plt.title('Concept-wise Entropy')

        plt.ylabel('Variance')
        plt.xlabel('Mean')
        
        
        for dataset_name, concept_ents in data_concept_ents.items():
            tag_array = np.array([
                list(item.keys())[0]
                for item
                in concept_ents
            ])
            
            mean_array = np.array([
                list(item.values())[0][0]
                for item
                in concept_ents
            ])
            
            var_array = np.array([
                list(item.values())[0][1]
                for item
                in concept_ents
            ])

            plt.scatter(
                x = mean_array,
                y = var_array,
                marker = 'x',
                linewidths = 0.5,
                label = dataset_name,
            )

        plt.legend(loc='best')

        res_name = './res/mean_var.jpg'
        if not os.path.isdir('./res'):
            os.mkdir('./res')
        
        plt.savefig(res_name)  
        return

    def vis_all_entropy_RT(self, data_concept_ents:dict, save="./tmp.png"):
        '''
        Visualize all entropies across datasets with mouse-hover based analysis in real time.

        Keep the mouse on a scatter point to see its concept name, entropy mean, and entropy variance.

        @Args:

            data_concept_ents [dict(key: dataset_name, value: list(tuple(concept_name, mean, var)))];
        '''
        # initialize fig
        self.fig, self.ax = plt.subplots()
        plt.grid()
        plt.title('Concept-wise Entropy')

        plt.ylabel('Variance')
        plt.xlabel('Mean')   

        # self.ax.set_xticks([8, 12, 14, 15])

        self.annot = self.ax.annotate(
            "", 
            xy = (0, 0), 
            xytext = (20, 20),
            textcoords = "offset points",
            bbox = dict(boxstyle="round", fc="w"),
            arrowprops = dict(arrowstyle="->"),
        )
        self.annot.set_visible(False)
        
        for dataset_name, concept_ents in data_concept_ents.items():
            self.tag_array = np.array([
                list(item.keys())[0]
                for item
                in concept_ents
            ])
            
            self.mean_array = np.array([
                list(item.values())[0][0]
                for item
                in concept_ents
            ])

            # print(list(concept_ents[0].values())[0])
            self.var_array = np.array([
                list(item.values())[0][1]
                for item
                in concept_ents
            ])
            # update self.sc
            self.sc = plt.scatter(
                x = np.power(1.25, self.mean_array),
                y = self.var_array,
                marker = 'x',
                linewidths = 1,
                label = dataset_name,
            )

        # plt.plot([13, 15], [6, 0])
        # plt.plot([12, 14], [6, 0])

        plt.legend(loc='best')

        self.fig.canvas.mpl_connect("motion_notify_event", self.__hover)
        if save:
            plt.savefig(save)
        else:
            plt.show() 
        return

    def __update_annot(self, ind):
        pos = self.sc.get_offsets()[ind["ind"][0]]
        self.annot.xy = pos
        text = "{}\nmean = {}\nvar = {}".format(
            " ".join([self.tag_array[n] for n in ind["ind"]]),
            " ".join(["{:.4f}".format(self.mean_array[n]) for n in ind["ind"]]),
            " ".join(["{:.4f}".format(self.var_array[n]) for n in ind["ind"]]),
        )
        self.annot.set_text(text)
        return

    def __hover(self, event):
        '''
        Recall function for handling hover events.
        '''
        vis = self.annot.get_visible()
        if event.inaxes == self.ax:
            cont, ind = self.sc.contains(event)
            if cont:
                self.__update_annot(ind)
                self.annot.set_visible(True)
                self.fig.canvas.draw_idle()
            else:
                if vis:
                    self.annot.set_visible(False)
                    self.fig.canvas.draw_idle()
        return

