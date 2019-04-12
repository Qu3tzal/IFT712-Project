# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

class visualizationBuilder():
    """ This class prints out graphics visualizations of model scores """

    def __init__(self):
        """ Constructor. """
        pass

    def barChart(self, methods, scores):
        position = range(len(methods))
        plt.bar(position, scores, align='center', alpha=0.5, color="g")
        plt.xticks(position, methods)
        plt.ylabel('Accuracy')
        plt.title('Methods')
        plt.ylim(0.7)
        plt.show()

