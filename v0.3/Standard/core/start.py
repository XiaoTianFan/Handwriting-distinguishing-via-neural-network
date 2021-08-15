# compute by numpy and cupy
import numpy               
import cupy     
# imprt fundamental functions from file
import network_Definition_CPU
import network_Definition_GPU
# imprt data&image functions from file
import data_Initialisation
# library for plotting arrays
import matplotlib.pyplot
import time
import random
import os


def networkTrainingProcess(Netname, epochs, training_list):
# epochs is the number of times the training data set is used for training   
    
    # record the training start time
    start_time = time.time()
    # record the total rounds
    total_rounds = 0

    for e in range(epochs):
        # record the training time in the current log loop
        current_round = 0

        # record the training start time of the current log loop
        logLoop_start_time = time.time()

        # go through all records in the trainging set LIST
        for i in training_list:
            #append total rounds
            total_rounds += 1

            
            # give the Network two item numpy.ndarray(data & target) in each round of the training
            # get the current errors while the Network is training
            output_errors = Netname.train(i[0], i[1])
            
            '''# output the training speed and precision (in the current log loop)
            if current_round == 50:
                # record the training end time of the current log loop
                logLoop_end_time = time.time()
                
                # calculate the average error
                loss = output_errors.sum() / Netname.onodes
                # make sure the error is readable
                if loss < 0:
                    loss *= -1
                else:
                    pass

                # calculate the time consumption of each train
                speed = (logLoop_start_time - logLoop_end_time) / 50

                #output loss, speed and total rounds
                print('Loss: ', str(loss)[:10], 'Speed: ', str(speed)[:10]+' s/train', 'Total rounds: ', str(total_rounds))

                # reset the training time in the current log loop
                current_round = 0
                # reset the training start time of the current log loop
                logLoop_start_time = time.time()
                
                pass

            else:
                current_round += 1
                
                pass'''

            pass
        pass

    # record the training end time
    end_time = time.time()
    # output the total running time
    print('Total Running time: %s Seconds'%(end_time - start_time))


def NetworkTesting(Netname, testing_list, computing_mode='CPU'):
    # scorecard for how well the network performs, initially empty
    scorecard = []
    # go through all records in the testing set
    # give the Network two items(data & target) each round of the training
    for i in testing_list:
        # correct answer is second value of each item couple
        correct_label = i[1]
        # query the network
        outputs = Netname.query(i[0])
        # the index of the highest value corresponds to the label
        label = numpy.argmax(outputs)
       
        # append correct or incorrect to list
        if (label == numpy.argmax(correct_label)):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)
            pass
       
        # output match detail
        #print(label, numpy.argmax(correct_label))
    
    pass

    # calculate the performance score, the fraction of correct answers
    scorecard_array = numpy.asarray(scorecard)
    print ("performance = ", scorecard_array.sum() / scorecard_array.size)


# run the network backwards, given a label, see what image it produces
def backwardImage(Netname, label, save=True, display=True, computing_mode='CPU'):
    # set the Matplotlib
    # 'Agg': no image display 
    # 'TkAgg': display via Tkinter
    if display == True:
        matplotlib.use('TkAgg')
    else:
        matplotlib.use('Agg')
    
    # create the output signals for this label
    targets = numpy.zeros(Netname.onodes) + 0.01
    # all_values[0] is the target label for this record
    targets[label] = 0.99
    
    # re-calculate image data
    image_data = Netname.backquery(targets)
    # reshape image data in to array
    image_array = image_data.reshape(28,28)
    #create output image 
    if computing_method == 'GPU':
        matplotlib.pyplot.imshow(cupy.ndarray.get(image_array), cmap='Greys', interpolation='None')
    else:
        matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')

    #save output image as standard setting
    if save == True:
        image_rout_name = '../Netmindscane '+str(label)+' '+str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))+'.jpg'
        matplotlib.pyplot.savefig(image_rout_name)
    else:
        pass

    #dispaly the image
    if display == True:
        matplotlib.pyplot.show()
    else:
        pass



#intro
input('''welcome to this neural network engine (v0.3)- by Tim Fan
this engine allows you to create, train, and test your image categorising network via this interactive system

in this version, only 3 layers of nodes are available, includes input, hidden, and output
the output nodes should correspond to the category number of the data set

during your operation, both CPU computing and GPU computing are available, 
decide which mode you're going to utilize based on your data size and predictable training duration
however, GPU computing mode requires NVIDIA-CUDA supporting and it will only be activated when training/testing/backquery process is precceding
in this version, CUDA-v11.4.X should perfectly fit in this programme

although the default setting is designed for categorizing hand-writed numbers, 
any other kind of image that according to the requiring format is applicable
lookup the data format in ../data/format.txt

the following guide will help you start your journey

---> PRESS ANYKEY to continue
''')

#start the main loop
while True:
    
    
    #input computing method
    computing_method = input('''
-------------------------------------------------------------------------------------
    above all, pleas choose the computing method: GPU or CPU

time consumption comparison on NVIDIA GTX1070 for refer:
CC10 in:784 hid:300 out:10 lr:0.1 e:10
GPU 1.46s
CPU 2.78s

mnist in:784 hid:300 out:10 lr:0.1 e:5
GPU 186s
CPU 560s

generally speaking, GPU computing mode will perform better when dealing with bulky data

''')
    #record computing method
    if computing_method == 'GPU':
        computing_method = 'GPU'
        print('CONFIRMED')
    elif computing_method == 'CPU':
        computing_method = 'CPU'
        print('CONFIRMED')

    # asking whether want to create a new data set or not
    if input('''
-------------------------------------------------------------------------------------    
do you want to create your own data set or start training with former data set?: create/former

''') == 'create':
        new_data_set_name = input('''
-------------------------------------------------------------------------------------    
this function haven't installed yet, you can read the following introduction first
-------------------------------------------------------------------------------------    
to create a new data set, please make sure your data images have been organised in given format:

1. create a fold in ../data/image, name it as your new data set's name
    e.g. ../data/image/mynewdataset
2. create folds that named as int sequence
    e.g. ./mynewdataset/(0)&(1)&(2)
    make sure you can remember the int represent which category in your raw data set
    the reason of this int-only format is the int fold name will be used as target value(label) of each category directly
    and it have to be int to fit in the calculation process
3. make sure all your images are shaped in square 
    in other words, the pixels number of different sides should conform to each other. 

now, please input your new data set's name:
(again, you should create a fold in ../data/image before)

''')
        new_data_set_imgsize = int(input('''
------------------------------------------------------------------------------------- 
in this case, 'total pixels number' means the result of image_side_pixels^2
    e.g. the default data sets contain data of images formed in 28p*28p, so the total pixels number will be 784

input the total pixels number in a single image of your new data set:

'''))
        print('CONFIRMED, check your new data set in root dir')
        #create a new data set from '../data/image/(category)/(label).png'
        try:
            os.mkdir('../tempo')
        except FileExistsError:
            pass

        for image_fold_name in os.listdir('../data/image/%s/' % new_data_set_name):
            new_image_data = data_Initialisation.imageArray('../data/image/' + new_data_set_name + '/%s' % image_fold_name, image_size = new_data_set_imgsize)
            # save iamge datas of each categories as a single file in the root path
            numpy.savetxt("../tempo/category_" + image_fold_name + ".csv", new_image_data, delimiter = ',', fmt = '%1.0i')

        # mix all the datas randomly and save as one file
        record = []
        for j in os.listdir("../tempo/"):
            record.append(numpy.loadtxt("../tempo/"+j, delimiter = ',', skiprows=0))
        random.shuffle(record)
        numpy.savetxt("../"+new_data_set_name+".csv", record, delimiter = ',', fmt = '%1.0i')

        
        for r in os.listdir('../tempo'):
            os.remove('../tempo/'+r)
        os.rmdir('../tempo')
        

    else:
        pass
    
    # initialise the neural network 
    # set default parameters
    input_nodes = int(input('''
-------------------------------------------------------------------------------------
please start creating a neural network instance by inputing its parameters
(whether you are going to load a previous network or not, you need to create your network first
if you intend to load a previous network, it doesn't matter what parameter you inputed)

default parameters for refering:
input_nodes = 784
hidden_nodes = 300
output_nodes = 10
learning_rate = 0.1

PROMPT: the output_nodes should correspond to the category number of your data set,
        the input_nodes is the square of image side length(pixels),
        in this case, its the product of 28^2,
        as a result, you'd better don't change them, unless you are going to use your own data set

input input__nodes: '''))
    hidden_nodes = int(input('\ninput hidden__nodes:'))
    output_nodes = int(input('\ninput output__nodes:'))
    learning_rate = float(input('\ninput learning_rate: '))

    # create network instance 
    # Network denominating pattern e.g.            creating time      training times  performance
    #                                  'CC10 Network 2021_03_04_23_12_20 30k 0.9')
    if input('''
-------------------------------------------------------------------------------------    
do you want to create a new network or load a previous one?: new/previous

''') == 'new': 
        if computing_method == 'GPU':
            Net = network_Definition_GPU.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
        else:
            Net = network_Definition_CPU.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    else:
        if computing_method == 'GPU':
            Net = network_Definition_GPU.neuralNetwork(reload_net = input('''
-------------------------------------------------------------------------------------
your network fold should be saved in ../save

the parameters of your network should correspond to the parameters you have just inputed

please input your network fold name:

'''))
        else:
            Net = network_Definition_CPU.neuralNetwork(reload_net = input('''your network fold should be saved in ../save
the parameters of your network should correspond to the parameters you have just inputed

please input your network fold name:

'''))
        
    # prepare datas
    # there are 4 data sets of mnist in the fold(../data), 2 of minimised and 2 of full sized
    # there is also 2 data sets of 10 chinese charactors 人 心 永 天 火 寸 古 工 口 女, named as CC10_train(test).csv
    # the charactors order of CC10 is just above
    # default file names: '../data/mnist_train.csv'; '../data/mnist_test.csv'; '../data/CC10_train.csv'; '../data/CC10_test.csv'
    if input('''
-------------------------------------------------------------------------------------
do you want to import default data sets?: T/F
PROMPT: if you have chosen a previous network in anterior process, make sure you will select the data set that accord to it

''') == 'T':
        defaul_data_set_selection = input('''
-------------------------------------------------------------------------------------
which set do you prefer?: mnist/CC10/mnist_10010

CAUTION: "mnist" data set need to be download in https://pjreddie.com/projects/mnist-in-csv/, but not included in github because of its size
         if you didn't have the full version data set, use mnist10010

''')
        training_data_list = data_Initialisation.dataArray('../data/%s_train.csv' % defaul_data_set_selection, Net.onodes)
        testing_data_list = data_Initialisation.dataArray('../data/%s_test.csv' % defaul_data_set_selection, Net.onodes)
    else:
        training_data_file = input('''
-------------------------------------------------------------------------------------
ALL your data set should be saved in ../data

The total categories of your data should correspond to the output nodes number you have just set

please input your training_date_set file name:

''')
        training_data_list = data_Initialisation.dataArray('../data/' + training_data_file, Net.onodes)
        print('training data set loaded')
        testing_data_file = input('''please input your testing _data_set file name:
''')
        testing_data_list = data_Initialisation.dataArray('../data/' + testing_data_file, Net.onodes)
    
    # traning & testing        
    if input('''
-------------------------------------------------------------------------------------    
do you want to train the network or test it?: train/test

''') == 'train':
        training_epoch = int(input('''
-------------------------------------------------------------------------------------
in this case, 'training_epochs' means how many times that your data set will be thoroughly trained 
        
prompt: when using default data set, an optimum epoch number should start from 5-10, or it may cause time suqandering

please input the training epochs:

'''))
        networkTrainingProcess(Net, training_epoch, training_data_list)
        if input('''
-------------------------------------------------------------------------------------
will you feel like test your network via testing_data_set you have loaded before immediately?: T/F

''') == 'T':
            NetworkTesting(Net, testing_data_list)
        else:
            pass
    else:
        if input('''
-------------------------------------------------------------------------------------
TEST function is only applicable when you loaded a previous network or trained a new network just now,
if you haven't loaded one, you may get a superising outcome

proceed?: T/F

''') == 'T':
            NetworkTesting(Net, testing_data_list)
        else:
            pass
    
    # save network
    if input('''
-------------------------------------------------------------------------------------
save network?: T/F

''') == "T":
        Net.saveNetwork()
    
    # display & save the Backward images
    if input('''
-------------------------------------------------------------------------------------
do you want to back activate the network and save the outcome images?: T/F

''') == 'T':
        print('creating '+str(Net.onodes)+'images in the root dir')
        for i in range(0, Net.onodes):
            backwardImage(Net, i, save=True, display=False)
            print('DONE')
    else:
        pass
    
    input('''
-------------------------------------------------------------------------------------
current procedure due, back to start
    
''')

    



