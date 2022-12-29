from mnist import MNIST

mnist_root = "D:\linux\opt\pjs\machine_learning\data\my_mnist"
mndata = MNIST(mnist_root)
print(type(mndata))
images, labels = mndata.load_training()
print(type(images))