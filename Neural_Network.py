# Rutger Introduction To Artificial Intelligence (CS 520) Final project
# Neural Network.py
# Date: 04/09/2017
# Created by Kaixiang Huang, Yehan Wang

import numpy as np
from samples import timecounter


relu = lambda x: x * (x > 0)


class NeuralNetworkClassifier:

    def __init__(self, data_type):
        # Some more separated functions can help us get better thinking beyond this neural network
        self.init = 0
        self.learning_para_store = {}
        self.data_type = data_type
        # -------------------- load trained model part --------------------------------
        # if data_type == "digit":
        #     try:
        #         model = np.load("final_parameter_digit.npy").item()
        #         self.learning_para_store = model
        #
        #     except IOError or ValueError:
        #         print("There is no stored weights and bias! We will initialize them!")
        # else:
        #     try:
        #         model = np.load("final_parameter_face.npy").item()
        #         self.learning_para_store = model
        #
        #     except IOError or ValueError:
        #         print("There is no stored weights and bias!")
        # -------------------- load trained model part --------------------------------

    def model_generate(self, input_size, output_size, hidden_size):
        # the initialize of Neural Network Classifier
        # W1 is the first layers' weights and its shape is (input_size * hidden_layer_size)
        # W2 is the second layers' weights and its shape is (hidden_layer_size * output_size)
        # B1, B2 is the first and second layers' bias
        # input_data should be (train_data number, input_data.flatten)

        para_name = ["W1", "B1", "W2", "B2"]
        model = {}
        # -------------------- load trained model part --------------------------------
        # if self.data_type == "digit":
        #     try:
        #         model = np.load("final_parameter_digit.npy").item()
        #         self.learning_para_store = model
        #         return model
        #     except IOError or ValueError:
        #         print("There is no stored weights and bias! We will initialize them!")
        # else:
        #     try:
        #         model = np.load("final_parameter_face.npy").item()
        #         self.learning_para_store = model
        #         return model
        #     except IOError or ValueError:
        #         print("There is no stored weights and bias! We will initialize them!")
        # -------------------- load trained model part --------------------------------
        model['W1'] = 0.001 * np.random.randn(input_size, hidden_size)
        model['B1'] = np.zeros(hidden_size)
        model['W2'] = 0.001 * np.random.randn(hidden_size, output_size)
        model['B2'] = np.zeros(output_size)

        return model

    def loss_cal(self, input_data,  model, input_labels=None, reg_strength=0.001):

        input_data_number = input_data.shape[0]
        # -----------------------Feed forward part-------------------------------------
        # compute each layers weights and output based on relu activation function
        hidden_layer1 = input_data.dot(model['W1']) + model['B1']
        hidden_layer1_output = np.maximum(0, hidden_layer1)

        hidden_layer2 = np.dot(hidden_layer1_output, model['W2']) + model['B2']

        # if labels is none, means we are doing some prediction that we can just use hidden_layer2 as our final scores to
        # do prediction

        if input_labels is None:
            return hidden_layer2

        # this out put is designed for softmax classifier
        hidden_layer2_output = np.sum(-hidden_layer2[range(input_data_number), input_labels] +
                                      np.log(np.sum(np.exp(hidden_layer2), axis=1)))/input_data_number

        # -----------------------back propagation part-------------------------------------
        # compute forward pass loss based on soft max loss function
        loss = hidden_layer2_output + 0.5 * reg_strength * (np.sum(model['W1']*model['W1'])
                                                                + np.sum(model['W2']*model['W2']))

        back_gradient = {}

        # calculate the back propagation gradient by chain rule

        gradient_layer2 = (np.exp(hidden_layer2).T / np.sum(np.exp(hidden_layer2), axis=1)).T
        back_y = np.zeros(gradient_layer2.shape)

        back_y[range(input_data_number), input_labels] = 1

        gradient_layer2 = (gradient_layer2 - back_y)/input_data_number
        gradient_layer1 = gradient_layer2.dot(model['W2'].T)*(hidden_layer1 >= 0)

        gradient_weight1 = input_data.T.dot(gradient_layer1)
        gradient_weight2 = hidden_layer1_output.T.dot(gradient_layer2)
        gradient_bias1 = np.sum(gradient_layer1, axis=0)
        gradient_bias2 = np.sum(gradient_layer2, axis=0)
        # Gradient regularization part
        gradient_weight1 += reg_strength * model['W1']
        gradient_weight2 += reg_strength * model['W2']

        back_gradient['W1'] = gradient_weight1
        back_gradient['B1'] = gradient_bias1
        back_gradient['W2'] = gradient_weight2
        back_gradient['B2'] = gradient_bias2
        return loss, back_gradient

    @timecounter
    def train(self, input_data, input_labels, validation_test_data,
              validation_test_label, output_size=10,  hidden_size=50, learning_rate=0.05, epoch=5000, validation_iters=1000):
        if self.data_type == "face":
            output_size = 2
        input_size = input_data.shape[1]
        model = self.model_generate(input_size, output_size, hidden_size)
        final_parameter = {}
        loss_data_history_table = []
        train_acc_history_table = []
        val_acc_history_table = []
        best_acc = 0.0

        for iters in range(epoch):
            loss, back_gradient = self.loss_cal(input_data,  model, input_labels)
            loss_data_history_table.append(loss)

            # update parameters
            for para in model:
                learning_rate_decay = 1

                if para not in self.learning_para_store:
                    self.learning_para_store[para] = np.zeros(back_gradient[para].shape)

                # ----Another way of updating parameters( Momentum update ) but does not work.....due to some bug

                # self.learning_para_store[para] = learning_rate_decay * self.learning_para_store[para] + \
                #                                  (1 - learning_rate_decay) * back_gradient[para]**2
                #
                # update = -learning_rate * back_gradient[para] / (np.sqrt(self.learning_para_store[para]) + 1e-8)
                # momentum = 0.9
                # self.learning_para_store[para] = momentum * self.learning_para_store[para] - learning_rate *back_gradient[para]
                # update = self.learning_para_store[para]
                # ----Another way of updating parameters( Momentum update ) but does not work.....due to some bug

                # -----------------Vanilla update-------------------------
                update = -learning_rate * back_gradient[para]
                model[para] += update

            learning_rate *= learning_rate_decay
            iters += 1

            if (iters % validation_iters) == 0 or iters == epoch:

                train_accuracy_test_data = input_data
                train_accuracy_test_label = input_labels

                train_score = self.loss_cal(train_accuracy_test_data, model)

                train_prediction = np.argmax(train_score, axis=1)

                train_accuracy = np.mean(train_prediction == train_accuracy_test_label)
                train_acc_history_table.append(train_accuracy)

                validation_score = self.loss_cal(validation_test_data, model)

                validation_prediction = np.argmax(validation_score, axis=1)
                validation_accuracy = np.mean(validation_prediction == validation_test_label)
                val_acc_history_table.append(validation_accuracy)

                if validation_accuracy >= best_acc:
                    best_acc = validation_accuracy
                    for parameter in model:
                        final_parameter[parameter] = model[parameter].copy()

                print('Finished epoch {} / {}: Train_Loss {}, train_accuracy: {}, validation_accuracy: {}, '
                      'leaning rate {}'
                      .format(iters, epoch, loss, train_accuracy, validation_accuracy, learning_rate))

                print('Finished optimization. best validation accuracy: {}'.format(best_acc))

        # Save the best model
        if self.data_type == "digit":
            np.save('final_parameter_digit.npy', final_parameter)

        else:
            np.save('final_parameter_face.npy', final_parameter)

        self.learning_para_store = final_parameter

        return final_parameter, loss_data_history_table, train_acc_history_table, val_acc_history_table

    def prediction(self, input_data):
        train_accuracy_test_data = input_data
        train_score = self.loss_cal(train_accuracy_test_data, self.learning_para_store)
        train_prediction = np.argmax(train_score, axis=1)
        return train_prediction
