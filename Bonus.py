import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


############################## math functions
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    ds = s * (1 - s)
    return s, ds


def tanh(x):
    t = np.tanh(x)
    dt = 1 - t ** 2
    return t, dt


def applyApplicationFunction(x, option):
    if option == "Sigmoid":
        return sigmoid(x)
    else:
        return tanh(x)


##############################

num_features = 784
num_out_classes = 10

class_encoding = {}

def encodeClasses():
    for num in range(0,10,1):
        encode_list=[]
        for i in range(0,10,1):
            if i==num:
                encode_list.append(1)
            else :
                encode_list.append(0)
        class_encoding[num]=encode_list

def calculateError(current,target):
    if(len(current)!= len(target)):
        return np.inf
    cur_mx=max(current)
    for i in range(len(current)):
        if current[i]==cur_mx:
            current[i]=1
        else :
            current[i]=0
    if current==target:
        return 1
    else :
        return -1


def splitDataset():
    mnist_train=pd.read_csv('mnist_train.csv')
    mnist_train = mnist_train.sample(frac=1).reset_index(drop=True)
    mnist_train.to_numpy()
    mnist_train=mnist_train.to_numpy()
    mnist_test=pd.read_csv('mnist_test.csv')
    mnist_test = mnist_test.sample(frac=1).reset_index(drop=True)
    mnist_test=mnist_test.to_numpy()
    # print(mnist_train[213][0])
    return mnist_train,mnist_test



def forwardPropagation(cur_sample,weight_matrix,num_neurons,num_layers,bias,application_function):

    all_neurons_lists=[]

    cur_neurons_list=list(cur_sample[1:])
    next_neurons_list=[]
    for layer in range(num_layers):

        if bias:
            cur_neurons_list.append(bias)

        next_neurons_list=[0]*num_neurons[layer] # initialze neurons list of the given size with value 0
        # print(cur_neurons_list)
        for next_neuron in range(len(next_neurons_list)): #   i know it can be done using vector multiplication
            for cur_neuron in range(len(cur_neurons_list)):

                next_neurons_list[next_neuron]+=cur_neurons_list[cur_neuron]*weight_matrix[layer][cur_neuron][next_neuron]

            f, df = applyApplicationFunction(next_neurons_list[next_neuron], application_function) #find application function for Fnet and it's derivative
            next_neurons_list[next_neuron] = f

        all_neurons_lists.append(next_neurons_list)
        cur_neurons_list=next_neurons_list


    output_neurons_list=[0]*num_out_classes

    if bias:
        cur_neurons_list.append(bias)

    for out_neuron in range(num_out_classes):
        for crnt_neuron in range(len(cur_neurons_list)):
            output_neurons_list[out_neuron]+=cur_neurons_list[crnt_neuron]*weight_matrix[num_layers][crnt_neuron][out_neuron]

        f,df=applyApplicationFunction(output_neurons_list[out_neuron],application_function) #find application function for Fnet and it's derivative
        output_neurons_list[out_neuron]=f


    all_neurons_lists.append(output_neurons_list)

    return all_neurons_lists #doesn't contain input neurons


def backwardPropagation(neurons_list,cur_sample,weight_matrix,num_neurons,num_layers,bias,application_function):

    gradiant_list=neurons_list # initialize gradiant list with the size of neurons list
    target=class_encoding[cur_sample[0]]

    # if bias :  # remove bias .... we don't need gradiant for it
    #     for i in range(num_layers):
    #         gradiant_list[i].pop()

    for i in range(num_out_classes):
        fnet=neurons_list[len(neurons_list)-1][i] # current output neuron
        f, df = applyApplicationFunction(fnet, application_function)#find application function for Fnet and it's derivative
        gradiant_list[len(gradiant_list)-1][i]=(target[i]-fnet)*df # gradiant at current output neuron

    # print("teest test -------------------------------------------")
    # for i in range(num_layers):
    #     print(i, "  len weight : ",len(weight_matrix[i+1]))
    #     print(i,"   len gradiant : ", len(gradiant_list[i]))
    # print("teest test -------------------------------------------")
    for layer in reversed(range(num_layers)):
        for cur in range(len(gradiant_list[layer])): # for each hidden unit
            gradiant_list[layer][cur]=0
            for next in range(len(gradiant_list[layer+1])):
                cur_weight=weight_matrix[layer+1][cur][next]
                gradiant_list[layer][cur]+=(cur_weight*gradiant_list[layer+1][next])
            fnet = neurons_list[layer][cur]
            f, df = applyApplicationFunction(fnet,application_function)  # find application function for Fnet and it's derivative
            gradiant_list[layer][cur]*=df

    return gradiant_list

def updateWeightBackPropagation(neuron_list,gradiant_list,cur_sample,weight_matrix,num_layers,learning_rate,bias):

    cur_neurons_list=list(cur_sample[1:])
    if bias:
        cur_neurons_list.append(bias)

    for in_neuron in range(len(cur_neurons_list)):
        for next_neuron in range(len(gradiant_list[0])):
            weight_matrix[0][in_neuron][next_neuron]+=(cur_neurons_list[in_neuron]*learning_rate*gradiant_list[0][next_neuron])

    for layer in range(num_layers):
        for in_neuron in range(len(neuron_list[layer])):
            for next_neuron in range(len(gradiant_list[layer+1])):
                weight_matrix[layer+1][in_neuron][next_neuron]+=(neuron_list[layer][in_neuron]*learning_rate*gradiant_list[layer+1][next_neuron])

    return weight_matrix


def trainBackPropagation(train_dataset,weight_matrix,num_neurons,num_layers,num_epochs,learing_rate,bias,application_function):

    for epoch in range(num_epochs):
        for sample in train_dataset:
            neurons_list=forwardPropagation(sample,weight_matrix,num_neurons,num_layers,bias,application_function)
            gradiant_list=backwardPropagation(neurons_list,sample,weight_matrix,num_neurons,num_layers,bias,application_function)
            weight_matrix=updateWeightBackPropagation(neurons_list,gradiant_list,sample,weight_matrix,num_layers,learing_rate,bias)
                                                    #(neuron_list,gradiant_list,cur_sample,weight_matrix,num_layers,learning_rate,bias)
            error=calculateError(neurons_list[len(neurons_list)-1],class_encoding[sample[0]])
            bias+=(learing_rate*error)
    return weight_matrix,bias

def testBackProp(test_dataset,weight_matrix,num_neurons,num_layers,num_epochs,learing_rate,bias,application_function):
    correct=0
    wrong=0
    confusion_matrix=[]
    for sample_num in range(len(test_dataset)):
        if sample_num == (len(test_dataset)/2):
            confusion_matrix.append([correct,wrong])
            correct=0
            wrong=0
        sample=test_dataset[sample_num]
        neurons_list=forwardPropagation(sample, weight_matrix, num_neurons, num_layers, bias, application_function)
        error=calculateError(neurons_list[len(neurons_list)-1],class_encoding[sample[0]])
        if error==1:
            correct+=1
        else :
            wrong+=1

    confusion_matrix.append([wrong,correct])
    accuracy=(confusion_matrix[0][0]+confusion_matrix[1][1])/(len(test_dataset))
    # print("Accuracy is : ",accuracy)
    return confusion_matrix,accuracy

def mainBackPropagation(num_neurons,num_layers,num_epochs,learing_rate,bias,application_function):
    encodeClasses()
    train_dataset,test_dataset=splitDataset()
    weight_matrix=[]
    frnum=int(num_features+bias+1)
    scnum=int(num_neurons[0]+1)
    weight_matrix.append(np.random.rand(frnum,scnum)) # create the first matrix between features and first hidden layer

    for i in range(num_layers):
        if i==num_layers-1 :
            frnum=int(num_neurons[num_layers-1]+1)
            scnum=int(num_out_classes+1)
            weight_matrix.append(np.random.rand(frnum,scnum))# create the last matrix between the last hidden layer and the output classes
        else :
            frnum=int(num_neurons[i]+ bias+1)
            scnum=int(num_neurons[i+1]+1)
            weight_matrix.append(np.random.rand(frnum,scnum))


    weight_matrix,bias=trainBackPropagation(train_dataset,weight_matrix,num_neurons,num_layers,
                     num_epochs,learing_rate,bias,application_function)    #return the new weight matrix , bias  and the neurons


    return testBackProp(test_dataset, weight_matrix, num_neurons, num_layers,
                 num_epochs, learing_rate, bias,application_function)

# print(mainBackPropagation([20,20,20,20,20],5,5,0.001,1,"Sigmoid"))#num_neurons,num_layers,num_epochs,learing_rate,bias,application_function
# splitDataset()