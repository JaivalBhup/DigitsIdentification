import numpy
import scipy.special
import matplotlib.pyplot

class NeuralNetwork:

    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        self.inodes = inputNodes
        self.hnodes = hiddenNodes
        self.onodes = outputNodes
        self.lr = learningRate
        self.W_ih = numpy.random.normal(0.0, pow(self.hnodes,-0.5), (self.hnodes, self.inodes))
        self.W_ho = numpy.random.normal(0.0, pow(self.onodes,-0.5), (self.onodes, self.hnodes))
        self.activationFunction = lambda x: scipy.special.expit(x)

    def train(self, input_list, target_list):
        # feedforwording
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T
        hidden_input = numpy.dot(self.W_ih,inputs)
        hidden_output = self.activationFunction(hidden_input)
        final_input = numpy.dot(self.W_ho, hidden_output)
        final_output = self.activationFunction(final_input)
        #backpropogation
        output_error = targets - final_output
        hidden_error = numpy.dot(self.W_ho.T, output_error)
        #gradientdecent
        self.W_ho += self.lr * numpy.dot(output_error*final_output*(1-final_output), numpy.transpose(hidden_output))
        self.W_ih += self.lr * numpy.dot(hidden_error*hidden_output*(1-hidden_output), numpy.transpose(inputs))

    def query(self, input_list):
        inputs = numpy.array(input_list, ndmin=2).T
        hidden_input = numpy.dot(self.W_ih,inputs)
        hidden_output = self.activationFunction(hidden_input)
        final_input = numpy.dot(self.W_ho, hidden_output)
        final_output = self.activationFunction(final_input)
        return final_output

# creating neural network
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
lr = 0.3
n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, lr)

#getting training data
datafile = open("mnistdataset/mnist_train.csv", 'r')
dataList = datafile.readlines()
datafile.close()
#training
for record in dataList:
    all_set = record.split(",")
    #input
    scaled_input = (numpy.asfarray(all_set[1:])/255 * 0.99) + 0.01
    #target
    target = numpy.zeros(output_nodes) + 0.01
    target[int(all_set[0])] = 0.99
    n.train(scaled_input, target)

#getting test data
test_datafile = open("mnistdataset/mnist_test_10.csv",'r')
test_dataList = test_datafile.readlines()
test_datafile.close()
#testing
scoreCard = []
for record in test_dataList:
    all_set = record.split(",")
    testAnswer = int(record[0])
    image = numpy.asfarray(all_set[1:]).reshape((28, 28))
    # matplotlib.pyplot.imshow(image, cmap='Greys', interpolation='None')
    # matplotlib.pyplot.show()
    output = n.query((numpy.asfarray(all_set[1:])/255 * 0.99)+0.01)
    testResult = numpy.argmax(output)
    print("The printed digit is ", testAnswer)
    print("The computer got ", testResult)
    if testAnswer == testResult:
        print("success")
        scoreCard.append(1)
    else:
        print("Failed")
        scoreCard.append(0)

    print("_____________________________")

scoreCard_array = numpy.asarray(scoreCard)
print("performance  :", scoreCard_array.sum()/scoreCard_array.size)
