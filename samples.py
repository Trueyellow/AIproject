# Rutger Introduction To Artificial Intelligence (CS 520) Final project
# samples.py
# Date: 04/07/2017
# Created by Kaixiang Huang, Yehan Wang
# Based on the  http://inst.eecs.berkeley.edu/~cs188/sp11/projects/classification/samples.py
import time
import os

class Datum:
    """The class helps up to get data, store or transfer data from pixels representing"""
    def __init__(self, data, width, height):
        """
        Create a new datum from file input (standard MNIST encoding).
        """
        self.height = height
        self.width = width
        if data == None:
            data = [[' ' for i in range(height)] for j in range(width)]
        self.pixels = convertToInteger(data)

    def getPixel(self, column, row):
        """
        Returns the value of the pixel at column, row as 0, or 1.
        """
        return self.pixels[column][row]

    def getPixels(self):
        """
        Returns all pixels as a list of lists.
        """
        return self.pixels


# Data processing, cleanup and display functions

def loadDataFile(filename, n, width, height):
    """
    Reads n data images from a file and returns a list of Datum objects.

    """
    data = []
    with open(filename, 'r') as image:
        for columns in range(n):
            single_data = []
            for rows in range(height):
                single_data.append(list(image.readline())[:-1])
            if len(single_data[0]) < width-1:
                break
            data.append(Datum(single_data, width, height))
    return data




def readlines(filename):
    """
    readline function to help use extract line data from file
    """
    if (os.path.exists(filename)):
        return [l[:-1] for l in open(filename).readlines()]


def loadLabelsFile(filename, n):
    """
    Reads n labels from a file and returns a list of integers.
    """
    fin = readlines(filename)
    labels = []
    for line in fin[:min(n, len(fin))]:
        if line == '':
            break
        if line == '                                                            ':
            break

        labels.append(int(line))

    return labels


def asciiGrayscaleConversionFunction(value):

    """
    Helper function for display purposes.
    """
    if (value == 0):
        return ' '
    elif (value == 1):
        return '+'
    elif (value == 2):
        return '#'


def IntegerConversionFunction(character):
    """
    Helper function for file reading.
    """
    if (character == ' '):
        return 0
    elif (character == '+'):
        return 1
    elif (character == '#'):
        return 2


def convertToInteger(data):
    """
    Helper function for file reading.
    """
    if type(data) != type([]):
        return IntegerConversionFunction(data)
    else:
        return list(map(convertToInteger, data))


# Time counter for train function
def timecounter(func):
    def wrapper(*args):
        start_time = time.time()
        train = func(*args)
        end_time = time.time()
        print('----------------The {} step tooks {} seconds to run--------------------------------'
              '--'.format(func.__name__, end_time - start_time))
        return train
    return wrapper


# calculate hit_rate and print result
def evaluate(prediction, labels):
    correct = 0
    for i in range(len(labels)):
        if prediction[i] == labels[i]:
            correct += 1
    hit_rate = float(correct)/len(labels)
    print('----------------The total number of test data is {}, and the '
          'prediction accuracy is {}---------------------\n\n\n'.format(len(labels),hit_rate))
