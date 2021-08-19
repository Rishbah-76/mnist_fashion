# mnist_fashion
Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. ]

Here's an example of how the data looks (each class takes three-rows):
![image](https://user-images.githubusercontent.com/74089340/130066769-78f51f92-cd67-4fb0-88e8-1fedc4fb1c5f.png)

# **What I'm doing?**
Here I'm doing multi-classification, this dataset conssit of many images of 10 different Items named as Labels
Each training and test example is assigned to one of the following labels:
Label	Description

```
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot
```

# **Our Result as well as Accuracy?**
## **Accuracy: 91%**
## **loss: 0.23133501410484314**
![image](https://user-images.githubusercontent.com/74089340/130069199-e0188adc-e4d4-44cd-b478-3e067a6e2113.png)


# **Appraoch**
For This use case we are using the CNN Network consisting the Following
3-Convolution Layers
2-Maxpooling layer
3-BachNormalization Layers
Finally the Fully Connected Network

# **How to use our code?**
1. First Download the Data for this [Link](https://github.com/zalandoresearch/fashion-mnist.git) :
   - Clone the Repo and Extract the downloaded file
   - Go to Data
   - Inside data Copy all the 'file.gz' 
2. Now add the Location in the prediction_app.ipynb and you are good to go.
   
   
# **How to load the Data as it is in Ubyte**
We can use two appraches, here are both but in code I have used the Later one(Struct one)
->> Here path must be where your Data is Present
```
import gzip
filePath_train_set = '<Your path>/train-images-idx3-ubyte.gz'
filePath_train_label = '<Your path>/data/train-labels-idx1-ubyte.gz'

filePath_test_set = '<Your path>/t10k-images-idx3-ubyte.gz'
filePath_test_label = '<Your path>/data/t10k-labels-idx1-ubyte.gz'

with gzip.open(filePath_train_label, 'rb') as trainLbpath:
    trainLabel = np.frombuffer(trainLbpath.read(), dtype=np.uint8,offset=8)
with gzip.open(filePath_train_set, 'rb') as trainSetpath:
    trainSet = np.frombuffer(trainSetpath.read(), dtype=np.uint8,offset=16).reshape(len(trainLabel), 784)

with gzip.open(filePath_test_label, 'rb') as testLbpath:
    testLabel = np.frombuffer(testLbpath.read(), dtype=np.uint8,offset=8)

with gzip.open(filePath_test_set, 'rb') as testSetpath:
    testSet = np.frombuffer(testSetpath.read(), dtype=np.uint8,offset=16).reshape(len(testLabel), 784)

X_train, X_test, y_train, y_test = trainSet, testSet, trainLabel, testLabel

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
```
<<-
  
Second Methos is to use Struct
```
->>
import gzip
import struct 
import numpy as np

def read_idx(filename):
    with gzip.open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
        #np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
  
x_train = read_idx("<Your Path>/data/train-images-idx3-ubyte.gz")
y_train = read_idx("<Your Path>/data/train-labels-idx1-ubyte.gz")
x_test = read_idx("<Your Path>/data/t10k-images-idx3-ubyte.gz")
y_test = read_idx("<Your Path>/data/t10k-labels-idx1-ubyte.gz")
```
 <<-
