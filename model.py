from Module import linear, relu, sequence
from lib import pickle
import random as rand

class SNN:
    def __init__(self) -> None:
        self.model = sequence()
        self.model.seed = 1234
        self.model.learning_rate = 0.1
        self.model.add(linear(features_in=2, features_out=2))
        self.model.add(relu(features_in=2))
        
        self.model.add(linear(features_in=2, features_out=1))
        self.model.add(relu(features_in=1))

def make_label(arr):
    if arr[0] == 1 and arr[1] == 1:
        return [0]
    elif arr[0] == 0 and arr[1] == 0:
        return [0]
    elif arr[0] == 1 and arr[1] == 0:
        return [1]
    else:
        return [1]

if __name__ == "__main__":
    model = SNN()
    for i in range(100000):
        arr = [rand.randint(0,1), rand.randint(0,1)]
        model.model.forward_prop(features=arr, label=make_label(arr))
        model.model.backward_prop(label=make_label(arr))
        model.model.gradient_descend()
        if (i%10000) == 0:
            print('label is : ', arr)
            print('loss is : ', model.model.loss)
    model.model.forward_prop(features=[1,1], label=make_label([0,1]))
    print()
    print(model.model.loss)
    print(model.model.features)

    model.model.forward_prop(features=[0,1], label=make_label([0,1]))
    print()
    print(model.model.loss)
    print(model.model.features)
    

