# This file is designed to be run with one command line argument, a .jpg of a symbol
# The output is the result of activating a neural network on the input, ideally the class of the symbol
# 
# Jody LeSage
# ID: HX85135
# CMSC471
# Spring 2016

# program constants
CLASSES = ['Smile','Hat','Hash','Heart','Dollar']   # possible outputs
N = 4                                               # hash function hashes to an N-by-N matrix of values
hiddenNodes = 5                                     # number of hidden nodes (one layer)

# stuff I wrote
from ConvertData                    import toBitmap
from HashFunctions                  import vectorize

# pybrain stuff
from pybrain.structure              import FeedForwardNetwork, LinearLayer, SigmoidLayer, SoftmaxLayer, FullConnection

# other stuff
import sys
import numpy as np

def buildNetwork():
    # make network objects
    network = FeedForwardNetwork()
    inputLayer = LinearLayer(N*N)
    hiddenLayer = SigmoidLayer(hiddenNodes)
    outputLayer = SoftmaxLayer(len(CLASSES))
    
    # connect the network
    network.addInputModule(inputLayer)
    network.addModule(hiddenLayer)
    network.addOutputModule(outputLayer)
    input_to_hidden = FullConnection(inputLayer, hiddenLayer)
    hidden_to_output = FullConnection(hiddenLayer, outputLayer)
    
    # connection weights determined by training
    input_to_hidden._setParameters([4.35074052, 0.54309903, -1.70788914, -0.37641228, 5.36652276, -5.95097706, -3.31479105, 1.48726254, 4.01124973, -2.23954635, -2.06566738, -1.05526604, 0.29454287, 0.34454901, 0.85887803, -1.54283834, -0.20189867, 2.39341244, -6.41137004, -5.21972223, 3.9786469, 2.62502833, -0.71182606, 1.49128153, 4.07595571,  3.84903291,  1.23479208, -1.84971481, -2.48423784, -2.89135785, 1.66872929, 2.9749277, 0.13388845, -6.34083074, 2.45295962, 4.37867807, -2.04605488, 5.83143133, 8.31634908, -1.15218811, -1.67941218, -2.21080881, 0.73068735, -1.38228386, -3.62022672, -0.93999936, 0.93052909, -3.83909343, -4.79640119, 3.39088663, 1.88639523, -2.10136095, -5.79122022, -0.39145108, -3.16506474, -0.99953878, -4.03241107, 2.27154235, 1.29838529, 1.65980538, 3.56765327, 0.2334956, -1.50648655, 3.43253185, 1.96654523, -0.52561201, 2.16708779, 3.43269409, -0.89938682, -4.57967624, -0.54859459, -0.02998314, 0.57531064, -3.22645606, -2.01130185, -0.92961716, 2.72892938, 1.70440582, -0.74359544, -1.75646309])
    hidden_to_output._setParameters([5.23726518, 4.23002634, -10.17060945, 0.24449631, -0.40430924, -5.10170453, -1.97718355, -3.29225334, 9.76797256, -0.31218471, -2.26134518, 8.19478967, 4.64314258, -5.10091146, -6.16245232, 5.12979712, -6.02720454, 0.98557444, -7.00566727, 5.45351844, -6.94985525, -5.76529963, 10.43833636, -0.35966021, -0.53525741])

    network.addConnection(input_to_hidden)
    network.addConnection(hidden_to_output)
    network.sortModules()               # initialize    
    return network
    
if __name__ == '__main__':
    inFile = sys.argv[1]
    bitmap = toBitmap(inFile)
    vector = vectorize(bitmap, N)
    network = buildNetwork()
    output = network.activate(vector)
    print(CLASSES[np.argmax(output)])
