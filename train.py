import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from Dataset import *
from Model import LeNet
import time

#1.train_set and test_set
train_set_size = len(train_set)
test_set_size = len(test_set)

print("The length of train_set = {}".format(train_set_size))
print("The length of test_set = {}".format(test_set_size))

#2.loader
train_loader = DataLoader(train_set,batch_size=64)
test_loader = DataLoader(test_set,batch_size=64)

#3.init Model,try to use GPU
LeNet_Model = LeNet()
if torch.cuda.is_available():
    LeNet_Model = LeNet_Model.cuda()

#4.Loss_function
Loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    Loss_fn = Loss_fn.cuda()

#5.Optimizer
learning_rate = 1e-3
optimizer = torch.optim.Adam(params=LeNet_Model.parameters(),lr=learning_rate)

#6.init Global variables

total_train_step = 0
total_test_step = 0

#7.add tensorboard
writer = SummaryWriter(log_dir='LeNet-Fashion_MNIST')

epochs = 20

#record train time
start_time = time.time()

for epoch in range(epochs):
    print("-----epoch[{}/{}] begin-----".format(epoch + 1, epochs))

    # ---------- train ----------
    LeNet_Model.train()
    for data in train_loader:
        images, targets = data
        if torch.cuda.is_available():
            images = images.cuda()
            targets = targets.cuda()

        outputs = LeNet_Model(images)
        loss = Loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step += 1
        writer.add_scalar('train_loss',loss,total_train_step)

        if total_train_step % 100 ==0:
            print("Training Step={}, Loss={:.4f}".format(total_train_step, loss.item()))

    LeNet_Model.eval()
    test_loss_total = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, targets = data
            if torch.cuda.is_available():
                images = images.cuda()
                targets = targets.cuda()

            outputs = LeNet_Model(images)
            loss = Loss_fn(outputs, targets)
            test_loss_total += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    avg_test_loss = test_loss_total / len(test_loader)

    total_test_step += 1
    writer.add_scalar('test_loss', avg_test_loss, total_test_step)
    writer.add_scalar('test_accuracy', accuracy, total_test_step)

    print(f"Epoch [{epoch + 1}/{epochs}]  Test Accuracy: {accuracy:.2f}%  |  Avg Loss: {avg_test_loss:.4f}")

writer.close()
torch.save(LeNet_Model.state_dict(), 'LeNet_Fashion_MNIST.pth')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training complete in {elapsed_time:.2f} seconds.")
print(f"The model has been saved successfully : LeNet_Fashion_MNIST.pth")