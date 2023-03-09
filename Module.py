from lib import *

class Layer:
    def __init__(self) -> None:
        self.name = self.__class__.__name__

    def feedforward(self, features):
        features = np.array(features, dtype=np.float64)

class linear(Layer):
    def __init__(self, features_in, features_out) -> None:
        super().__init__()
        self.features_out = features_out
        self.features_in = features_in
        self.weight = np.random.uniform(-1,1,[features_out, features_in])
        self.bias = np.zeros(features_out, dtype=np.float64)
        self.Z = np.zeros(features_out, dtype=np.float64)
        self.input = np.zeros(features_in, dtype=np.float64)
        self.dW = np.zeros([features_out, features_in], dtype=np.float64)
        self.dB = np.zeros([features_out], dtype=np.float64)

    def feedforward(self, features):
        super().feedforward(features)
        self.input = np.zeros(self.features_in, dtype=np.float64)
        self.input = features
        self.Z = np.dot(self.weight, features) + self.bias
        return self.Z
    
class relu(Layer):
    def __init__(self, features_in) -> None:
        super().__init__()
        self.A = np.zeros(features_in, dtype=np.float64)

    def __sigmoid(self, features):
        return 1 / (1+np.exp(-features))

    def feedforward(self, features):
        super().feedforward(features)
        self.A = self.__sigmoid(features=features)
        return self.A

class sequence:
    def __init__(self) -> None:
        self.layers = []
        self.loss: float = ...
        self.linear = []
        self.relu = []
        self.learning_rate = 0.1
        self.seed = 1
        np.random.seed(self.seed)

    def add(self, layer):
        self.layers.append(layer)
        if layer.name == "linear":
            self.linear.append(layer)
        elif layer.name == "relu":
            self.relu.append(layer)
        else:
            pass
    
    def __cost(self, features, label):
        return -(np.sum(label * np.log(features) + (1-label) * np.log(1-features))) / label.__len__()
    
    def __dCost(self, pred, label):
        return - ((label/pred) - ((1-label)/(1-pred)))
    
    def __dSigmoid(self, a):
        return a * (1-a)

    def forward_prop(self, features, label):
        label = np.array(label, dtype=np.uint8)
        self.features = features
        for i in range(self.layers.__len__()):
            self.features = self.layers[i].feedforward(self.features)

        self.loss = self.__cost(features=self.features, label=label)

    def backward_prop(self, label):
        label = np.array(label, dtype=np.uint8)
        for lab_i in range(label.size):
            for i in reversed(range(self.linear.__len__())):
                for j in range(self.linear[i].features_out):
                    for k in range(self.linear[i].features_in):
                        self.linear[i].dW[j,k] = self.__cal_dW(label=label, lab_i=lab_i, i=i,j=j,k=k)
                    self.linear[i].dB[j] = self.__cal_dB(label=label, lab_i=lab_i, i=i, j=j)

    def gradient_descend(self):
        for i in range(self.linear.__len__()):
            self.linear[i].weight -= self.learning_rate * self.linear[i].dW
            self.linear[i].bias -= self.learning_rate * self.linear[i].dB

    def __cal_dW(self, label, lab_i, i, j, k) ->np.ndarray:
        updated = 0
        input_features = 0
        sum_of_dW = 0

        if (i-1) < 0:
            input_features = self.linear[i].input[k]
        else:
            input_features = self.relu[i-1].A[k]

        if (i+1) >= self.linear.__len__():
            sum_of_dW = 1
        else:
            for index in range(self.linear[i+1].features_out):
                sum_of_dW += self.linear[i+1].dW[index][j] * self.linear[i+1].weight[index][j]
            sum_of_dW = sum_of_dW / (self.relu[i].A[j])
        updated = self.__dCost(self.features[lab_i], label[lab_i]) * self.__dSigmoid(self.relu[i].A[j]) * input_features * sum_of_dW
        return updated
    
    def __cal_dB(self, label, lab_i, i, j) ->np.ndarray:
        updated = 0
        sum_of_dB = 0

        if (i+1) >= self.linear.__len__():
            sum_of_dB = 1
        else:
            for index in range(self.linear[i+1].features_out):
                sum_of_dB += self.linear[i+1].dW[index][j] * self.linear[i+1].weight[index][j]
        updated = self.__dCost(self.features[lab_i], label[lab_i]) * self.__dSigmoid(self.relu[i].A[j]) * sum_of_dB
        return updated
