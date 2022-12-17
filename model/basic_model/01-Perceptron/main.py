import torch

from Perceptron import Perceptron, PerceptronLiHangBook

def basic():
    w = torch.tensor([0., 0.], requires_grad=False)
    b = torch.tensor(0.0, requires_grad=False)
    model = Perceptron(w, b)
    x = torch.tensor([[1., 2.], [2.2, -0.5], [-10.3, 8.], [34.2, -8.5]], requires_grad=False)
    y = torch.tensor([1., -1., 1., -1.], requires_grad=False)
    lr = 0.1

    flag = True
    while flag:
        flag = False
        for i in range(len(x)):
            pre_y = model(x[i])
            if pre_y != y[i]:
                model.w += lr * y[i] * x[i]
                model.b += lr * y[i]
                flag = True
            print("--------" + str(i) + "--------")
            print(model.w)
            print(model.b)


def book():
    x = torch.tensor([[3., 3.], [4., 3.], [1., 1.]])
    y = torch.tensor([1., 1., -1.])
    w = torch.tensor([0. for _ in range(len(x[0]))], requires_grad=False)
    b = torch.tensor(0., requires_grad=False)
    model = PerceptronLiHangBook(w, b)
    lr = 1.

    flag = True
    while flag:
        flag = False
        for i in range(len(x)):
            if model(x[i]) * y[i] <= 0:
                model.w += lr*y[i]*x[i]
                model.b += lr*y[i]
                flag = True
                print("-------------" + str(i) + "-------------")
                print(model.w)
                print(model.b)

def main():
    # basic()
    book()

if __name__ == "__main__":
    main()