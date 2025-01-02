# import required libraries
import numpy as np
import tarfile
import os

import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torchvision.models import resnet18

torch.manual_seed(100)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def training_step(model, batch):
    images, labels = batch
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    out = model(images)                  
    loss = F.cross_entropy(out, labels) 
    return loss

def validation_step(model, batch):
    images, labels = batch
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    out = model(images)                    
    loss = F.cross_entropy(out, labels)   
    acc = accuracy(out, labels)
    return {'Loss': loss.detach(), 'Acc': acc}

def validation_epoch_end(model, outputs):
    batch_losses = [x['Loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   
    batch_accs = [x['Acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()      
    return {'Loss': epoch_loss.item(), 'Acc': epoch_acc.item()}

def epoch_end(model, epoch, result):
    print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
    epoch, result['lrs'][-1], result['train_loss'], result['Loss'], result['Acc']))
    
def distance(model,model0):
    distance=0
    normalization=0
    for (k, p), (k0, p0) in zip(model.named_parameters(), model0.named_parameters()):
        space='  ' if 'bias' in k else ''
        current_dist=(p.data0-p0.data0).pow(2).sum().item()
        current_norm=p.data0.pow(2).sum().item()
        distance+=current_dist
        normalization+=current_norm
        print(f'Distance: {np.sqrt(distance)}')
        print(f'Normalized Distance: {1.0*np.sqrt(distance/normalization)}')
    return 1.0*np.sqrt(distance/normalization)

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [validation_step(model, batch) for batch in val_loader]
    return validation_epoch_end(model, outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    for epoch in range(epochs): 
        model.train()
        train_losses = []
        lrs = []
        for batch in train_loader:
            loss = training_step(model, batch)
            train_losses.append(loss)
            loss.backward()
            
            if grad_clip: 
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
            
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        epoch_end(model, epoch, result)
        history.append(result)
        sched.step(result['Loss'])
    return history

# defining the noise structure
class NoiseGenerator(nn.Module):
    """
    A neural network module for generating noise patterns
    through a series of fully connected layers.
    """

    def __init__(
            self, 
            dim_out: list,
            dim_hidden: list = [1000],
            dim_start: int = 100,
            ):
        """
        Initialize the NoiseGenerator.

        Parameters:
        dim_out (list): The output dimensions for the generated noise.
        dim_hidden (list): The dimensions of hidden layers, defaults to [1000].
        dim_start (int): The initial dimension of random noise, defaults to 100.
        """
        super().__init__()
        self.dim = dim_out
        self.start_dims = dim_start  # Initial dimension of random noise

        # Define fully connected layers
        self.layers = {}
        self.layers["l1"] = nn.Linear(self.start_dims, dim_hidden[0])
        last = dim_hidden[0]
        for idx in range(len(dim_hidden)-1):
            self.layers[f"l{idx+2}"] = nn.Linear(dim_hidden[idx], dim_hidden[idx+1])
            last = dim_hidden[idx+1]

        # Define output layer
        self.f_out = nn.Linear(last, math.prod(self.dim))        

    def forward(self):
        """
        Forward pass to transform random noise into structured output.

        Returns:
        torch.Tensor: The reshaped tensor with specified output dimensions.
        """
        # Generate random starting noise
        x = torch.randn(self.start_dims)
        x = x.flatten()

        # Transform noise into learnable patterns
        for layer in self.layers.keys():
            x = self.layers[layer](x)
            x = torch.relu(x)

        # Apply output layer
        x = self.f_out(x)

        # Reshape tensor to the specified dimensions
        reshaped_tensor = x.view(self.dim)
        return reshaped_tensor

def main(
    t_Epochs: int,
    t_Steps: int,
    t_Learning_Rate: float,
    t_Batch_Size: int,
    t_Number_of_Noise_Batches: int,
    t_Regularization_term: float,
    t_Layers: list,
    t_Noise_Dim: int,
    new_baseline: bool = False,
    logs: bool = False,
):

    # Dowload the dataset
    dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
    download_url(dataset_url, '.')

    # Extract from archive
    with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
        tar.extractall(path='./data')

    # Look into the data directory
    data_dir = './data/cifar10'
    if logs:
        print(os.listdir(data_dir))
    classes = os.listdir(data_dir + "/train")
    if logs:
        print(classes)

    transform_train = tt.Compose([
        tt.ToTensor(),
        tt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = tt.Compose([
        tt.ToTensor(),
        tt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_ds = ImageFolder(data_dir+'/train', transform_train)
    valid_ds = ImageFolder(data_dir+'/test', transform_test)

    batch_size = t_Batch_Size # Changed
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = resnet18(num_classes = 10).to(DEVICE = DEVICE)

    if new_baseline or not os.path.exists("ResNET18_CIFAR10_ALL_CLASSES.pt"):
        epochs = 40
        max_lr = 0.01
        grad_clip = 0.1
        weight_decay = 1e-4
        opt_func = torch.optim.Adam

        history = fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl, 
                                    grad_clip=grad_clip, 
                                    weight_decay=weight_decay, 
                                    opt_func=opt_func)

        torch.save(model.state_dict(), "ResNET18_CIFAR10_ALL_CLASSES.pt")

    model.load_state_dict(torch.load("ResNET18_CIFAR10_ALL_CLASSES.pt"))
    history = [evaluate(model, valid_dl)]
    
    if logs:
        print(history)

    # list of all classes
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # classes which are required to un-learn
    classes_to_forget = [0, 2]

    # classwise list of samples
    num_classes = 10
    classwise_train = {}
    for i in range(num_classes):
        classwise_train[i] = []

    for img, label in train_ds:
        classwise_train[label].append((img, label))
        
    classwise_test = {}
    for i in range(num_classes):
        classwise_test[i] = []

    for img, label in valid_ds:
        classwise_test[label].append((img, label))

        # getting some samples from retain classes
    num_samples_per_class = 1000

    retain_samples = []
    for i in range(len(classes)):
        if classes[i] not in classes_to_forget:
            retain_samples += classwise_train[i][:num_samples_per_class]

            # retain validation set
    retain_valid = []
    for cls in range(num_classes):
        if cls not in classes_to_forget:
            for img, label in classwise_test[cls]:
                retain_valid.append((img, label))
                
    # forget validation set
    forget_valid = []
    for cls in range(num_classes):
        if cls in classes_to_forget:
            for img, label in classwise_test[cls]:
                forget_valid.append((img, label))
                
    forget_valid_dl = DataLoader(forget_valid, batch_size, num_workers=3, pin_memory=True)
    retain_valid_dl = DataLoader(retain_valid, batch_size*2, num_workers=3, pin_memory=True)


    # loading the model
    model = resnet18(num_classes = 10).to(DEVICE = DEVICE)
    model.load_state_dict(torch.load("ResNET18_CIFAR10_ALL_CLASSES.pt"))

    noises = {}
    for cls in classes_to_forget:
        if logs:
            print("Optiming loss for class {}".format(cls))
        # Here is the place of change
        noises[cls] = NoiseGenerator(
            dim_out = [batch_size, 3, 32, 32],
            dim_hidden=t_Layers,
            dim_start=t_Noise_Dim,
            )
        opt = torch.optim.Adam(noises[cls].parameters(), lr = t_Learning_Rate)

        num_epochs = t_Epochs # Changed
        num_steps = t_Steps # Changed
        class_label = cls
        for epoch in range(num_epochs):
            total_loss = []
            for batch in range(num_steps):
                inputs = noises[cls]()
                labels = torch.zeros(batch_size)+class_label
                outputs = model(inputs)
                loss = -F.cross_entropy(outputs, labels.long()) + t_Regularization_term*torch.mean(torch.sum(torch.square(inputs), [1, 2, 3])) # Changed
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss.append(loss.cpu().detach().numpy())
            if logs:
                print("Loss: {}".format(np.mean(total_loss)))

    batch_size = t_Batch_Size # Changed
    noisy_data = []
    num_batches = t_Number_of_Noise_Batches # Changed
    class_num = 0

    for cls in classes_to_forget:
        for i in range(num_batches):
            batch = noises[cls]().cpu().detach()
            for i in range(batch[0].size(0)):
                noisy_data.append((batch[i], torch.tensor(class_num)))

    other_samples = []
    for i in range(len(retain_samples)):
        other_samples.append((retain_samples[i][0].cpu(), torch.tensor(retain_samples[i][1])))
    noisy_data += other_samples
    noisy_loader = torch.utils.data.DataLoader(noisy_data, batch_size=t_Batch_Size, shuffle = True)


    optimizer = torch.optim.Adam(model.parameters(), lr = 0.02)

    for epoch in range(1):  
        model.train(True)
        running_loss = 0.0
        running_acc = 0
        for i, data in enumerate(noisy_loader):
            inputs, labels = data
            inputs, labels = inputs,torch.tensor(labels)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            # if logs:
            #   print statistics
            running_loss += loss.item() * inputs.size(0)
            out = torch.argmax(outputs.detach(),dim=1)
            assert out.shape==labels.shape
            running_acc += (labels==out).sum().item()
        if logs:
            print(f"Train loss {epoch+1}: {running_loss/len(train_ds)},Train Acc:{running_acc*100/len(train_ds)}%")

    if logs:
        print("Performance of Standard Forget Model on Forget Class")
    history = [evaluate(model, forget_valid_dl)]
    if logs:
        print("Accuracy: {}".format(history[0]["Acc"]*100))
    if logs:
        print("Loss: {}".format(history[0]["Loss"]))

    if logs:
        print("Performance of Standard Forget Model on Retain Class")
    history = [evaluate(model, retain_valid_dl)]
    if logs:
        print("Accuracy: {}".format(history[0]["Acc"]*100))
    if logs:
        print("Loss: {}".format(history[0]["Loss"]))

    heal_loader = torch.utils.data.DataLoader(other_samples, batch_size=t_Batch_Size, shuffle = True)

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)


    for epoch in range(1):  
        model.train(True)
        running_loss = 0.0
        running_acc = 0
        for i, data in enumerate(heal_loader):
            inputs, labels = data
            inputs, labels = inputs,torch.tensor(labels)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()

            # if logs:
            #   print statistics
            running_loss += loss.item() * inputs.size(0)
            out = torch.argmax(outputs.detach(),dim=1)
            assert out.shape==labels.shape
            running_acc += (labels==out).sum().item()
        if logs:
            print(f"Train loss {epoch+1}: {running_loss/len(train_ds)},Train Acc:{running_acc*100/len(train_ds)}%")

    if logs:
        print("Performance of Standard Forget Model on Forget Class")
    history = [evaluate(model, forget_valid_dl)]
    if logs:
        print("Accuracy: {}".format(history[0]["Acc"]*100))
    if logs:
        print("Loss: {}".format(history[0]["Loss"]))

    if logs:
        print("Performance of Standard Forget Model on Retain Class")
    history = [evaluate(model, retain_valid_dl)]
    if logs:
        print("Accuracy: {}".format(history[0]["Acc"]*100))
    if logs:
        print("Loss: {}".format(history[0]["Loss"]))

