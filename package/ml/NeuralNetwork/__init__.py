from ml.NeuralNetwork import  cgd
from ml.NeuralNetwork import lstm
__CATEGORY__ = ["Cgd","Lstm"]

def getNeuralNetworkByName(name):

    if name == "Cgd":
        return cgd.Cgd()
    elif name == "Lstm":
    	return lstm.Lstm()

    return None

def getNeuralNetworkModels():
    return __CATEGORY__
