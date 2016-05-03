# This file builds, trains, and test neural networks
# Jody LeSage
# ID: HX85135
# CMSC471
# Spring 2016

# program constants
CLASSES = ['Smile','Hat','Hash','Heart','Dollar']   # possible outputs
TRAINING_FOLDER = 'Data/TrainingSet/'               # parent folder of training set
TEST_FOLDER = 'Data/TestSet/'                       # parent folder of test set
CLASS_FOLDERS = ['01/','02/','03/','04/','05/']     # class folder names correspond with indicies in CLASSES
MAX_EPOCHS = 10000                                  # max number of epochs when training to convergence
#N = 5                                              # bitmap hashing to n-by-n regions
#INPUT_DIMENSION = N*N                              # dimesionality of input vector

# stuff I wrote
from ConvertData                    import toBitmap
from HashFunctions                  import vectorize

# library stuff
import os

# pybrain stuff
from pybrain.datasets               import ClassificationDataSet
from pybrain.utilities              import percentError
from pybrain.tools.shortcuts        import buildNetwork
from pybrain.supervised.trainers    import BackpropTrainer
from pybrain.structure              import FeedForwardNetwork, LinearLayer, SigmoidLayer, SoftmaxLayer, FullConnection

def generateDataSet(directory, n, verbose=False):
    ds = ClassificationDataSet(n*n, class_labels=CLASSES)
    if(verbose):
        print('Building data set...')
    for folder in CLASS_FOLDERS:
        path = directory + folder
        if(verbose):
            print('    Now entering + ' + path)
        for filename in os.listdir(path):
            bitmap = toBitmap(path + filename, verbose)
            vector = vectorize(bitmap, n, verbose)
            ds.appendLinked(vector , [CLASS_FOLDERS.index(folder)])
    ds._convertToOneOfMany(bounds=[0,1])
    return ds
    
def generateNetwork(hiddenNodes, inputDimension):
    return buildNetwork(inputDimension, hiddenNodes, len(CLASSES), outclass=SoftmaxLayer, bias=False) # using softmax for outputs because it's appropriate for classification problems
    
def trainNetwork(network, trainingSet, testSet, verbose=False):
    trainer = BackpropTrainer(network, dataset=trainingSet, verbose=verbose)
    if(verbose):
        print('Number of training samples:', len(trainingSet))
        print('Number of inputs:', trainingSet.indim)
        print('Number of classes:', trainingSet.outdim)
        print('Sample Input:', trainingSet['input'][0], trainingSet['target'][0], trainingSet['class'][0])
    trainer.trainUntilConvergence(maxEpochs=MAX_EPOCHS)
    
    trainingSetError = percentError(trainer.testOnClassData(), trainingSet['class'])
    testSetError = percentError(trainer.testOnClassData(dataset=testSet), testSet['class'])
    print('Total epochs:', trainer.totalepochs)
    print('Training Set Error:', trainingSetError)
    print('Test Set Error:', testSetError)
    return testSetError

def examineNetwork(network):
    print(network)
    print(network.params)

def findBestParameters():
    results = []
    for n in range(2,8):
        trainingSet = generateDataSet(TRAINING_FOLDER, n)
        testSet = generateDataSet(TEST_FOLDER, n)
        for hiddenNodes in range(1,9):
            network = generateNetwork(hiddenNodes, n*n)
            r = trainNetwork(network, trainingSet, testSet)
            results.append((r, n, hiddenNodes))
    results.sort()
    for i in results:
        print(i)

def buildByHand():
    N = 4
    hiddenNodes = 5
    
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
    
    network.addConnection(input_to_hidden)
    network.addConnection(hidden_to_output)
    network.sortModules()               # initialize
    
    # build training and test sets
    trainingSet = generateDataSet(TRAINING_FOLDER, N)
    testSet = generateDataSet(TEST_FOLDER, N)
    
    # train the network
    trainNetwork(network, trainingSet, testSet, verbose=True)
    
    # examine and record weights
    print(input_to_hidden.params)
    print(hidden_to_output.params)
    
    
    
if __name__ == '__main__':
    buildByHand()
