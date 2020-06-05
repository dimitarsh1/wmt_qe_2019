#import matplotlib.pyplot as plt
import os

def show_plot(x, y):
    ''' Draw and visualize a chart
    
    :param x: the values on the x axis
    :param y: the values on the y axis
    '''
    plt.plot(x, y)
    plt.show()
    
def show_plot2(y1, y2):
    ''' Draw and visualize a chart
    
    :param y1: the values for the first line
    :param y2: the values for the second line
    '''
    x = list(range(len(y1)))
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.show()
    
def save_plot2(y1, y2, path):
    ''' Draw and visualize a chart and save
    
    :param y1: the values for the first line
    :param y2: the values for the second line
    '''
    plt.figure()
    x = list(range(len(y1)))
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.savefig(path)
    plt.close()

def export_values(y, name):
    ''' Save the values in a file

    :param y: the values for the line
    :param name: the name of the file to log values out
    '''

    fileDir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(fileDir, name + ".log"), "a") as oF:
        oF.write('\t'.join([str(i[0]) for i in y]) + "\n")
