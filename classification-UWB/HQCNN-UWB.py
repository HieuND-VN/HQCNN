import os
from numpy import vstack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM, Bidirectional
import tensorflow as tf

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np
import autograd.numpy as anp
from pennylane.templates.embeddings import AmplitudeEmbedding, AngleEmbedding
from pennylane.templates.state_preparations import MottonenStatePreparation
from pennylane.templates import RandomLayers
from sklearn.preprocessing import normalize
import torch
from torchsummary import summary as summary

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def import_from_files():

    rootdir = './dataset/'
    output_arr = []
    val_arr = []
    first = 1
    for dirpath, dirnames, filenames in os.walk(rootdir):
        for file in filenames:
            filename = os.path.join(dirpath, file)
            print(filename)
            output_data = []
            df = pd.read_csv(filename, sep=',', header=0)
            input_data = df.values
            for item in input_data:
                item[15:] = item[15:] / item[2]
            randomize = np.arange(len(input_data))
            np.random.shuffle(randomize)
            random_data = input_data[randomize]
            if first > 0:
                first = 0
                output_arr = random_data
            else:
                output_arr = vstack((output_arr, random_data))

    return output_arr
data = import_from_files()



test_data = data[37800:, :]
trainval_data = data[:37800, :]
#---------------------------------------------------------------#
x_train = trainval_data[:33600, 15:]
x_train = x_train
y_train = trainval_data[:33600, 0]
#---------------------------------------------------------------#
x_test = test_data[:, 15:]
x_test = x_test
y_test = test_data[:, 0]
#---------------------------------------------------------------#
x_val = trainval_data[33600:, 15:]
y_val = trainval_data[33600:, 0]
#---------------------------------------------------------------#
x_train_filter_01 = np.where((y_train == 0) | (y_train == 1))
x_val_filter_01 = np.where((y_val == 0) | (y_val == 1))
x_test_filter_01 = np.where((y_test == 0) | (y_test == 1))
#---------------------------------------------------------------#
X_train, X_val, X_test = x_train[x_train_filter_01], x_val[x_val_filter_01], x_test[x_test_filter_01]
Y_train, Y_val, Y_test = y_train[x_train_filter_01], y_val[x_val_filter_01], y_test[x_test_filter_01]
#---------------------------------------------------------------#
Y_train_1 = [1 if y == 1 else 0 for y in Y_train]
Y_val = [1 if y == 1 else 0 for y in Y_val]
Y_test = [1 if y == 1 else 0 for y in Y_test]
#---------------------------------------------------------------#
Y_train_ = torch.unsqueeze(torch.tensor(Y_train_1), 1)
Y_val_ = torch.unsqueeze(torch.tensor(Y_val), 1)
Y_test_ = torch.unsqueeze(torch.tensor(Y_test), 1)
#---------------------------------------------------------------#
Y_train_hot = torch.scatter(torch.zeros((len(X_train), 2)), 1, Y_train_, 1)
Y_val_hot = torch.scatter(torch.zeros((len(X_val), 2)), 1, Y_val_, 1)
Y_test_hot = torch.scatter(torch.zeros((len(X_test), 2)), 1, Y_test_, 1)
#---------------------------------------------------------------#
X_train_1 = torch.tensor(X_train).float()
X_val_1 = torch.tensor(X_val).float()
X_test_1 = torch.tensor(X_test).float()
#---------------------------------------------------------------#
data_loader = torch.utils.data.DataLoader(list(zip(X_train_1, Y_train_hot)), batch_size=64, shuffle=True, drop_last=True)
val_data_loder = torch.utils.data.DataLoader(list(zip(X_val_1, Y_val_hot)), batch_size=64, shuffle=True, drop_last=True)
#---------------------------------------------------------------#

print(f'<Total dataset> TYPE: {type(data)} / SHAPE: {data.shape}')
print(f'<Total dataset[0]> TYPE: {type(data[0])} / SHAPE: {data[0].shape}')
print('___________________________________________________________________________________________')
print(f'\t\t Training dataset......')
print(f'x_train: TYPE-{type(x_train)} / SHAPE-{np.shape(x_train)} / VALUE-{x_train[0][0:20]}')
print(f'x_train_filter: TYPE-{type(x_train_filter_01)} / SHAPE-{np.shape({type(x_train_filter_01)})} / VALUE-{x_train_filter_01}')
print(f'X_train: TYPE-{type(X_train)} / SHAPE-{np.shape(X_train)} / VALUE-{X_train[0][0:20]}')
print(f'X_train_1: TYPE-{type(X_train_1)} / SHAPE-{np.shape(X_train_1)} / VALUE-{X_train_1[0][0:20]}')
print(f'\n')
print(f'y_train: TYPE-{type(y_train)} / SHAPE-{np.shape(y_train)} / VALUE-{y_train[0:10]}')
print(f'Y_train: TYPE-{type(Y_train)} / SHAPE-{np.shape(Y_train)} / VALUE-{Y_train[0:10]}')
print(f'Y_train_1: TYPE-{type(Y_train_1)} / SHAPE-{np.shape(Y_train_1)}/ VALUE-{Y_train_1[0:10]}')
print(f'Y_train_: TYPE-{type(Y_train_)} / SHAPE-{np.shape(Y_train_)}/ VALUE-{Y_train_[0:10]}')
print(f'Y_train_hot: TYPE-{type(Y_train_hot)} / SHAPE-{np.shape(Y_train_hot)}/ VALUE-{Y_train_hot[0:10]}')
print('___________________________________________________________________________________________')


'''
import pennylane as qml


def U_4(params, wires):  # 3 params 2 qubit
    qml.RZ(-np.pi / 2, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])  # (source, target)
    qml.RZ(params[0], wires=wires[0])
    qml.RY(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RZ(params[2], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(np.pi / 2, wires=wires[0])


def conv_layer1(U, params, Uname):
    if Uname == 'U_4':  # parameter 3
        U(params[0:3], wires=[0, 1])
        U(params[3:6], wires=[2, 3])
        U(params[6:9], wires=[4, 5])
        U(params[9:12], wires=[6, 7])
        U(params[12:15], wires=[8, 9])

        U(params[15:18], wires=[1, 2])
        U(params[18:21], wires=[3, 4])
        U(params[21:24], wires=[5, 6])
        U(params[24:27], wires=[7, 8])
        U(params[27:30], wires=[9, 0])


def U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6,
          wires):  # 15 params, Convolutional Circuit 10
    qml.U3(*weights_0, wires=wires[0])
    qml.U3(*weights_1, wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(weights_2, wires=wires[0])
    qml.RZ(weights_3, wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(weights_4, wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(*weights_5, wires=wires[0])
    qml.U3(*weights_6, wires=wires[1])


# Unitary Ansatz for Pooling Layer
def Pooling_ansatz1(weights_0, weights_1, wires):  # 2 params
    qml.CRZ(weights_0, wires=[wires[0], wires[1]])
    qml.PauliX(wires=wires[0])
    qml.CRX(weights_1, wires=[wires[0], wires[1]])


n_qubits = 8
dev = qml.device("default.qubit", wires=n_qubits)
Pooling_out = [1, 3, 5, 7]


@qml.qnode(dev)
def qnode(inputs, weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7,
          weights_8):  # , weights_9, weights_10, weights_11, weights_12, weights_13, weights_14, weights_15, weights_16, weights_17
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    # qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
    # qml.BasicEntanglerLayers(weights, wires=range(n_qubits))

    # QCNN
    # --------------------------------------------------------- Convolutional Layer1 ---------------------------------------------------------#
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[0, 1])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[2, 3])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[4, 5])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[6, 7])
    # U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[8, 9])

    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[1, 2])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[3, 4])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[5, 6])
    U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[7, 0])
    # U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[7, 8])
    # U_SU4(weights_0, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, wires=[9, 0])

    # --------------------------------------------------------- Pooling Layer1 ---------------------------------------------------------#
    ## Pooling Circuit  Block 2 weights_7, weights_8
    Pooling_ansatz1(weights_7, weights_8, wires=[0, 1])
    Pooling_ansatz1(weights_7, weights_8, wires=[2, 3])
    Pooling_ansatz1(weights_7, weights_8, wires=[4, 5])
    Pooling_ansatz1(weights_7, weights_8, wires=[6, 7])
    # Pooling_ansatz1(weights_7, weights_8, wires=[8,9])

    # --------------------------------------------------------- Convolutional Layer2 ---------------------------------------------------------#
    # U_SU4(weights_9, weights_10, weights_11, weights_12, weights_13, weights_14, weights_15, wires=[1, 3])
    # U_SU4(weights_9, weights_10, weights_11, weights_12, weights_13, weights_14, weights_15, wires=[5, 7])
    # U_SU4(weights_9, weights_10, weights_11, weights_12, weights_13, weights_14, weights_15, wires=[3, 5])
    # U_SU4(weights_9, weights_10, weights_11, weights_12, weights_13, weights_14, weights_15, wires=[7, 9])
    # U_SU4(weights_9, weights_10, weights_11, weights_12, weights_13, weights_14, weights_15, wires=[9, 1])

    ##--------------------------------------------------------- Pooling Layer2 ---------------------------------------------------------#
    ### Pooling Circuit  Block 2 weights_7, weights_8
    # Pooling_ansatz1(weights_16, weights_17, wires=[1,3])
    # Pooling_ansatz1(weights_16, weights_17, wires=[3,5])
    # Pooling_ansatz1(weights_16, weights_17, wires=[5,7])
    # Pooling_ansatz1(weights_16, weights_17, wires=[7,9])

    # conv_layer1(U, params9, U2)
    # pooling_layer1(Pooling_ansatz1, params5, V)
    result = [qml.expval(qml.PauliZ(wires=i)) for i in Pooling_out]
    return result

weight_shapes = {
    "weights_0": 3,
    "weights_1": 3,
    "weights_2": 1,
    "weights_3": 1,
    "weights_4": 1,
    "weights_5": 3,
    "weights_6": 3,
    "weights_7": 1,
    "weights_8": 1,
}

clayer_1 = torch.nn.Linear(1016, 8)
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
clayer_2 = torch.nn.Linear(4, 2)
#clayer_3 = torch.nn.Linear(128, 2)
softmax = torch.nn.Softmax(dim=1) #  torch.nn.sigmoid()
layers = [clayer_1, qlayer, clayer_2, softmax] # clayer_1,
model = torch.nn.Sequential(*layers)#.to(device)

opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss = torch.nn.CrossEntropyLoss()

summary(model, (1016,))
batch_size = 64  # 5
batches = 33600 // batch_size  # 200 // batch_size

data_loader = torch.utils.data.DataLoader(list(zip(X_train, Y_train_hot)), batch_size=64, shuffle=True, drop_last=True)
val_data_loder = torch.utils.data.DataLoader(list(zip(X_val, Y_val_hot)), batch_size=64, shuffle=True, drop_last=True)
epochs = 10
count = 0
for epoch in range(epochs):
    running_loss = 0
    running_loss_val = 0
    model.train()
    for xs, ys in data_loader:
        count += 1
        if count % 10 == 0:
            print(count)
        opt.zero_grad()

        loss_evaluated = loss(model(xs), ys)
        loss_evaluated.backward()
        opt.step()

        # running_loss += loss_evaluated
        running_loss += loss_evaluated.item()
        # print("loss_evaluated:", loss_evaluated)
    avg_loss = running_loss / batches
    print(f"[{epoch}]/[{epochs}] Loss_evaluated: {avg_loss}")
    
    model.eval()
    with torch.no_grad():
        y_pred_tr = model(X_train)
        predictions_tr = torch.argmax(y_pred_tr, axis=1).detach().numpy()
        correct_tr = [1 if p == p_true else 0 for p, p_true in zip(predictions_tr, Y_train)]
        accuracy_tr = sum(correct_tr) / len(correct_tr)
        print("training_Accuracy : ", accuracy_tr)
        for xv, yv in val_data_loder:
            loss_evaluated_val = loss(model(xv), yv)
            running_loss_val += loss_evaluated_val.item()
        avg_loss_val = running_loss_val / batches * 8
        y_pred_val = model(X_val)
        predictions_val = torch.argmax(y_pred_val, axis=1).detach().numpy()
        correct_val = [1 if p == p_true else 0 for p, p_true in zip(predictions_val, Y_val)]
        accuracy_val = sum(correct_val) / len(correct_val)
        print("Validation_Accuracy : ", accuracy_val)
    f = open("test10.txt", 'a')
    f.write("Epoch : %f\n" % epoch)
    f.write("Avg_loss_traing: %f\n" % avg_loss)
    f.write("Acc_training: %f\n" % accuracy_tr)
    f.write("Avg_loss_validation: %f\n" % avg_loss_val)
    f.write("Acc_validation: %f \n" % accuracy_val)
    f.close()
    print("Average loss over epoch {}: {:.4f}".format(epoch + 1, avg_loss))

model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    predictions = torch.argmax(y_pred, axis=1).detach().numpy()
correct = [1 if p == p_true else 0 for p, p_true in zip(predictions, Y_test)]
accuracy = sum(correct) / len(correct)
print(f"Accuracy: {accuracy * 100}%")
'''
