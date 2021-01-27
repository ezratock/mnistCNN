import torch
import torch.nn
import torch.nn.functional

batch_size_train = 64
batch_size_test = 1000

train_images_size = 60000 #max 60,000
test_images_size = 10000 #max 10,000

NUM_EPOCHS = 5
BATCHES_PER_EPOCH = 1000

learning_rate = 0.0001
beta1 = 0.5    # ???
beta2 = 0.999  # ???

train_images = open('/Users/ezra/Downloads/train-images-idx3-ubyte','rb').read()[16:]
train_labels = open('/Users/ezra/Downloads/train-labels-idx1-ubyte','rb').read()[8:]
print('Received training data')
test_images = open('/Users/ezra/Downloads/t10k-images-idx3-ubyte','rb').read()[16:]
test_labels = open('/Users/ezra/Downloads/t10k-labels-idx1-ubyte','rb').read()[8:]
print('Received test data')

def create_image_tensor(images,size): #So pytorch just automatcially converts from binary string?
    images_list = []
    for image in range(size):
        images_list.append([])
        for col in range(28):
            images_list[image].append([])
            for row in range(28):
                images_list[image][col].append(images[image*784+col*28+row])
    return torch.tensor(images_list).view(size,1,28,28)

def create_label_tensor(labels,size): #but apparently NOT bs->tensor when you just do torch.tensor(train_labels).... wtf??
    labels_list = []                  #made this pretty useless seeming function to fix this ^
    for label in range(size):
        labels_list.append(labels[label])
    return torch.tensor(labels_list).view(size)

train_images_tensor = create_image_tensor(train_images, train_images_size)
print('Constructed training image tensor')
train_labels_tensor = create_label_tensor(train_labels, train_images_size)
print('Constructed training label tensor')
test_images_tensor = create_image_tensor(test_images, test_images_size)
print('Constructed test image tensor')
test_labels_tensor = create_label_tensor(test_labels, test_images_size)
print('Constructed test label tensor')

class MNIST(torch.nn.Module):
    def __init__(self):
        super(MNIST,self).__init__()
        self.layer1=torch.nn.Linear(784,32)
        self.layer2=torch.nn.Linear(32,32)
        self.layer3=torch.nn.Linear(32,10)
    def forward(self,input):
        input = torch.nn.functional.relu(self.layer1(input))
        input = torch.nn.functional.relu(self.layer2(input))
        input = self.layer3(input)
        return torch.nn.functional.log_softmax(input, dim=1)

def train_model(train_data, train_labels):
    model=MNIST()
    model=model.float()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) #Not sure what 'betas=(beta1, beta2)' parameter does ??
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        for batch in range(BATCHES_PER_EPOCH):
            optimizer.zero_grad() #sets the gradient to zero before the loss calculation
            BATCH_SIZE = int(train_images_size/BATCHES_PER_EPOCH)
            batch_input = train_data[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE] #list of images from batch beginning to end position in train_data
            batch_labels = train_labels[batch*BATCH_SIZE:(batch+1)*BATCH_SIZE]
            output = model(batch_input.view(BATCH_SIZE,784)) #expects first dimension to be batches dimension
            loss_criterion = torch.nn.CrossEntropyLoss()
            loss = loss_criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print("Epoch %s loss: %s" % (epoch, epoch_loss / BATCHES_PER_EPOCH))
    return model

def test_model(test_data, test_labels, model):
    with torch.no_grad():
        output = model(test_data.view(-1,784)) #-1 because there are no batches for testing.  output = list of size 10 1D tensors
        correct = 0
        total = 0
        for image_num, guess in enumerate(output):
            if torch.argmax(guess) == test_labels[image_num]:
                correct += 1
            total += 1
    print("Accuracy: %s%%" % (round(float(correct)*100/float(total),5)))

print('\nTraining the model')
model = train_model(train_images_tensor.float(), train_labels_tensor) #train_labels should NOT be .float()
print("Training complete\n\nTesting Model")
test_model(test_images_tensor.float(), test_labels_tensor, model)
