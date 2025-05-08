# Cong Tran
# 1002046419

import numpy as np

# tr_data: the training inputs. 
#   This is a 2D numpy array, where each row is a training input vector.
# tr_labels: a numpy column vector. 
#   That means it is a 2D numpy array, with a single column. 
#   tr_labels[i,0] is the class label for the vector stored at tr_data[i].
# test_data: the test inputs. 
#   This is a 2D numpy array, where each row is a test input vector.
# test_labels: a numpy column vector. 
#   That means it is a 2D numpy array, with a single column. 
#   test_labels[i,0] is the class label for the vector stored at test_data[i].
# training_rounds: An integer greater than or equal to 1, 
#   specifying the number of training rounds that you should use. 
#   Each training round consists of using the whole training set exactly once 
#   (i.e., using each training example exactly once to update the weights).
def perceptron_train_and_test(tr_data: any, tr_labels: any, test_data: any, test_labels: any, training_rounds: int) -> None: 

    # the steps from slides 45: 

    # get absolute max value
    absolute_max_tr_data = get_absoluate_maximum_value(tr_data)

    #normalize the data
    normalized_tr_data = tr_data / absolute_max_tr_data
    normalized_test_data = test_data / absolute_max_tr_data 

    # length - # of vectors in array --  # of value in each vector
    length, size = normalized_tr_data.shape 
    
    #initalize the weights
    weights = np.random.uniform(-0.5, 0.5, size = (size, 1))
    biases = np.random.uniform(-0.5, 0.5, size = (1, 1))

    # find the last loss E(b, w) - sum over all traing examples
    initial_loss = loss_function(biases, weights, normalized_tr_data, tr_labels, length)
    # print(f' this is the current loss of the function {initial_loss}')

    

    #repeat this n amount of times
    for round in range(1, training_rounds + 1): 
        # learning rate update after eery round
        learning_rate  = 0.98 ** round

        for i in range(length): 
            
            #initalize the values 
            input_vector = normalized_tr_data[i].reshape(-1, 1)
            output_label = tr_labels[i, 0]

            #calculate the partial dervivative
            dldw = partial_derivative_w(w=weights, b=biases, tn = output_label, xn = input_vector)
            dldb = partial_derivative_b(w=weights, b=biases, tn = output_label, xn = input_vector)

            # the update on weights
            weights -= learning_rate * dldw
            biases -= learning_rate * dldb
        

        # curr_loss = loss_function(biases, weights, normalized_tr_data, tr_labels, length)
        # print(f"training Round {round}, loss: {curr_loss}")

    correct = 0
    for i in range(length): 

        #initalize the values 
        input_test_vector = normalized_test_data[i].reshape(-1, 1)
        output_test_label = test_labels[i, 0]

        #calculate the values
        a = pre_activation_function(w=weights, b=biases, x=input_test_vector)
        z = activation_function(a)

        predicted = test_output(z)
        accuracy = 1 if predicted == output_test_label else 0 #idk why its asking for an int
        correct += accuracy

        print_test_object(object_id=i, predicted_class=predicted, true_class=output_test_label, accuracy=accuracy)

    classification_accuracy = correct / length
    print_classification_accuracy(classification_accuracy=classification_accuracy)

            


#partial derivative of error function with respect to w
def partial_derivative_w(w: float, b: float, tn: any, xn: any) -> float:
    a = pre_activation_function( w=w, b=b, x=xn)
    z = activation_function(a)
    # (z(n) - tn) * (1 - z(xn)) * z(xn) * xn
    return (z - tn) * (1 - z) * z * xn

# partial derivative of error function with respect to b
def partial_derivative_b(w: float, b: float, tn: any, xn: any) -> float:
    a = pre_activation_function(w=w, b=b, x=xn)
    z = activation_function(a)
    # (z(xn) - tn) * z(xn) * (1 - z(xn))
    return (z - tn) * z * (1 - z)


# calculating the loss (sum of squared differences)
def loss_function(b: any, w: any, training_data: any, training_label: any, length: int) -> float:

    loss = 0

    for i in range(length): 
        a = pre_activation_function(w, b, training_data[i])
        z = activation_function(a)
        square_differences = 0.5 * ((z - training_label[i]) ** 2)
        # print(f'this is a: {a}\nthis is z: {z}\nthis is ssd {square_differences}')
        loss += square_differences

    return loss[0] # it is returned in an array lol


# pre-activation aka calculating a
def pre_activation_function(w: any, b: any, x: any) -> any: 
    w_T = np.transpose(w)
    return np.dot(w_T, x) + b

# this is basically e^(-b - wT @ xn)
def e_function(a: float): 
    return np.exp(- a)

def activation_function(a: float): 
    return 1 / (1 + e_function(a))

# get the absoluate maximum value of the entire dataset
def get_absoluate_maximum_value(dataset: any) -> any: 
   return np.max(np.abs(dataset))

# print test objects
def print_test_object(object_id: int, predicted_class: any, true_class: any, accuracy: float): 
    print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f\n' % 
           (object_id, str(predicted_class), str(true_class), accuracy))
    
# print classification accuracy
def print_classification_accuracy(classification_accuracy: float) -> None: 
    print('classification accuracy=%6.4f\n' % (classification_accuracy))


def test_output(perceptron_value: float) -> int: 
#     If the perceptron outputs a value less than .5 for some test input, 
#       and the true class label for that test input is 0, 
#       then we consider that the output is correct.
    if perceptron_value < 0.5: 
        return 0
#     If the perceptron outputs a value greater than or equal to .5 for some test input, 
#       and the true class label for that test input is 1, 
#       then we consider that the output is correct.
    elif perceptron_value >= 0.5: 
        return 1
#     Otherwise, 
#       we consider that the output of the perceptron is incorrect.
    else: 
        print(f'the output of the perceptron is incorrect: {perceptron_value}')
        return None