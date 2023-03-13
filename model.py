from Module import linear, relu, sequence
from lib import *

class XOR:
    def __init__(self) -> None:
        self.model = sequence(seed=1234)
        self.model.learning_rate = 0.1

        self.model.add(linear(features_in=2, features_out=4))
        self.model.add(relu(features_in=4))

        self.model.add(linear(features_in=4, features_out=2))
        self.model.add(relu(features_in=2))


def make_label(arr):
    if arr[0] == 0 and arr[1] == 0:
        return [0,0]
    elif arr[0] == 1 and arr[1] == 1:
        return [0,1]
    elif arr[0] == 0 and arr[1] == 1:
        return [1,0]
    else:
        return [1,1]
    
if __name__ == "__main__":
    model = XOR()

    for i in range(100000):
        arr = [rand.randint(0,1), rand.randint(0,1)]
        model.model.forward_prop(arr, make_label(arr))
        model.model.backward_prop(make_label(arr))
        model.model.gradient_descend()
        if (i%10000) == 0:
            print("LABEL IS : ", arr)
            print("loss is : ", model.model.loss)

    print()
    model.model.forward_prop([1,1], make_label([1,1]))
    print("PRED IS : ", model.model.features, " loss is : ", model.model.loss)
    print()

    model.model.forward_prop([0,0], make_label([0,0]))
    print("PRED IS : ", model.model.features, " loss is : ", model.model.loss)
    print()
    
    model.model.forward_prop([1,0], make_label([1,0]))
    print("PRED IS : ", model.model.features, " loss is : ", model.model.loss)
    print()
    