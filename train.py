from Module import linear, relu, sequence, Brain
from lib import *

class SNN:
    def __init__(self) -> None:
        rand.seed(1234)
        self.brain = Brain()
        self.model = sequence(
            fc= [
                linear(features_in=2, features_out=2),
                relu(),

                linear(features_in=2, features_out=1),
                relu()
            ]
        )

    def save(self, path):
        self.brain.weight_linear = []
        self.brain.bias_linear = []

        for i in reversed(range(self.model.linear.__len__())):
            self.brain.weight_linear.append(self.model.linear[i].weight)
            self.brain.bias_linear.append(self.model.linear[i].bias)
        pickle.dump(self.brain, open(path, "wb"))
        
def atttach_lable(arr) -> np.float64:
    if arr[0] == 0 and arr[1] == 1:
        return [1]
    elif arr[0] == 0 and arr[1] == 0:
        return [0]
    elif arr[0] == 1 and arr[1] == 1:
        return [0]
    else:
        return [1]

def none_minus(input) -> np.float64:
        if input < 0:
            return -input
        return input

def xor_fit():
    model = SNN()
    root_path = os.path.abspath(os.path.dirname(__file__))
    root_brain = root_path + '\\xornn.brain'
    
    for i in range(10):
        arr = [np.float64(rand.randint(0,1)), np.float64(rand.randint(0,1))]
        model.model.forward_prop(features=arr, label=atttach_lable(arr=arr))
        model.model.backward_prop(1, atttach_lable(arr=arr))

    model.save(root_brain)
    predict = pickle.load(open(root_brain, "rb"))
    print(predict.prediction_value([0,0], NeuralNetwork=model, label = atttach_lable([0,0])))

if __name__ == "__main__":
    xor_fit()