# compute by numpy when using CPU computing mode
import numpy                
import scipy.misc
import imageio
import os


def dataArray(file_name, output_node_num):
    # load the original data CSV file into a list
    with open(file_name, 'r') as original_data_file:
        original_data_list = original_data_file.readlines()
    # creat returning list
    modified_data_list = []
    # go through all records in the data set
    for record in original_data_list:
        # split the record by the ',' commas
        current_values = record.split(',')
        # scale and shift the inputs
        input_array = (numpy.asfarray(current_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        target_array = numpy.zeros(output_node_num) + 0.01
        # current_values[0] is the target label for this record
        target_array[int(current_values[0])] = 0.99    
        # save the modified datas
        modified_data_list.append((input_array,target_array))

        pass
    
    #this is a LIST
    return modified_data_list


def imageArray(image_file_path, image_group_number=0, image_size=784):
    # create new image list
    image_data_list = []

    # obtain image data from path 
    for image_file_name in os.listdir(image_file_path):
        
        # re-organise image instituion
        img_array = imageio.imread(image_file_path+'/'+image_file_name, as_gray = True)
        img_data = 255.0 - img_array.reshape(image_size)

        # record image label
        label = int(image_file_path.split('/')[-1])

        # append label and image data to test data set
        record = numpy.append(label, img_data.astype(int))
        image_data_list.append(record) 

        pass
        
    # each item of this list is consist of 785 numbers
    # the first is label and the rest are the pixels' values
    return image_data_list
    