# import required libraries
import numpy as np
import tarfile
import os

import random
from typing import Tuple, Dict, List, Union
import math
import torch
from torch import nn
from tqdm import tqdm
from PIL import Image

import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as tt
from torchvision.models import resnet18

# Set the seed for reproducible results
# random.seed(100)
# torch.manual_seed(100)
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
        for batch in tqdm(train_loader, desc="Training...", unit="batch", leave=True):
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
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.device = "cpu"
        # Define fully connected layers
        self.layers = {}
        self.layers["l1"] = nn.Linear(self.start_dims, dim_hidden[0]).to(self.device)
        last = dim_hidden[0]
        for idx in range(len(dim_hidden)-1):
            self.layers[f"l{idx+2}"] = nn.Linear(dim_hidden[idx], dim_hidden[idx+1]).to(self.device)
            last = dim_hidden[idx+1]

        # Define output layer
        self.f_out = nn.Linear(last, math.prod(self.dim)).to(self.device)    

    def forward(self):
        """
        Forward pass to transform random noise into structured output.

        Returns:
        torch.Tensor: The reshaped tensor with specified output dimensions.
        """
        # Generate random starting noise
        x = torch.randn(self.start_dims).to(self.device)
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

class NoiseDataset(Dataset):

    def __init__(self, noise_generator: NoiseGenerator, noise_labels: torch.Tensor, number_of_noise: int = 100,):

        self.noise_generator = noise_generator
        self.noise_labels  = noise_labels
        self.number_of_noise = number_of_noise if isinstance(self.noise_labels, torch.Tensor) else len(self.noise_labels)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __len__(self) -> int:
        return self.number_of_noise
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:

        return self.noise_generator().to(self.device), self.noise_labels.to(self.device)

class SubData(Dataset):
    def __init__(self, data, transform):
        self.data = data
        self.transform = transform
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        path, label = self.data[idx]

        img = Image.open(f"{path}").convert('RGB')
        img = self.transform(img)
        return img, label

class FeatureMU_Loader(Dataset):
    """
    This class creates a dataset that combines retained data with noise-generated samples.
    
    The noise is generated using a list of dictionaries, each containing a noise generator, a label for the noise, and the number of noise samples to generate.
    The class provides a unified dataset interface with a length method and an item retrieval method that distinguishes between retained data and noise-generated samples.
    """

    def __init__(self, noisies: List[Dict[str, Union[NoiseGenerator, torch.Tensor, int]]], retain_data: Dataset, transform=None):
        """
        Initialize the FeatureMU_Loader.

        Args:
            noisies (List[Dict[str, Union[NoiseGenerator, torch.Tensor, int]]]): A list of dictionaries containing information about the noise to generate.
                Each dictionary should contain the following keys:
                    - "gen": The noise generator.
                    - "label": The label of the noise.
                    - "n": The number of noise samples to generate.
            retain_data (Dataset): The dataset to retain.
            transform (Optional[Callable]): The transform to apply to the images. If None, no transform is applied.
        """
        super().__init__()
        self.noisies = noisies
        self.retain_data = retain_data
        self.transform = transform

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.retain_data) + sum([n["n"] for n in self.noisies])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return an element of the dataset.

        If the index is in the range of the retained dataset, return an element from the retained dataset.
        If the index is after the retained dataset, generate a noise sample using the noise generator and return it.
        Args:
            idx (int): The index of the element to return.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The element of the dataset, which is a tuple containing the image and the label.
        """
        if idx < len(self.retain_data):
            # If the index is in the range of the retained dataset, return an element from the retained dataset
            path, label = self.retain_data[idx]

            img = Image.open(f"{path}").convert('RGB')
            if self.transform is not None:
                # If a transform is specified, apply it to the image
                img = self.transform(img)
            return img.to(self.device), label.to(self.device)
        else:
            # If the index is after the retained dataset, generate a noise sample using the noise generator and return it
            c = len(self.retain_data)
            for i in self.noisies:
                # For each noise generator, check if the index is in the range of the generated noise
                c += i["n"]
                if idx < c:
                    # If the index is in the range, generate a noise sample and return it
                    return i["gen"]().to(self.device), i["label"].to(self.device)

def main(
    t_Epochs: int = 10,
    t_Steps: int = 15,
    t_Learning_Rate: float = 0.18,
    t_Batch_Size: int = 370,
    t_Number_of_Noise_Batches: int = 1,
    t_Regularization_term: float = 0.15,
    t_Layers: list = [1024, 1024, 1024, 1024, 1024, 1024],
    t_Noise_Dim: int = 70,
    new_baseline: bool = True,
    logs: bool = False,
    model_eval_logs: bool = True,
    idx: int = -1,
    ):

    # Checking if the dataset needs to be downloaded
    if not os.path.exists(f'data{os.sep}cifar10'):
        if logs:
            print("Dataset not found. Downloading...")
        # Dowload the dataset
        dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
        download_url(dataset_url, '.')

        # Extract from archive
        with tarfile.open('cifar10.tgz', 'r:gz') as tar:
            tar.extractall(path='data')

    # Look into the data directory
    data_dir = f'data{os.sep}cifar10'
    if logs:
        print(os.listdir(data_dir))
    classes = os.listdir(data_dir + f"{os.sep}train")
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

    torch.cuda.empty_cache()

    train_ds = ImageFolder(data_dir+f'{os.sep}train', transform_train)
    valid_ds = ImageFolder(data_dir+f'{os.sep}test', transform_test)

    batch_size = t_Batch_Size # Changed
    valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    # list of all classes
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # classes which are required to un-learn
    classes_to_forget = [0, 2]

    if not os.path.exists("data/all/models"):
        os.makedirs("data/all/models")
    n = len(os.listdir("data/all/models"))
    if new_baseline:
        print("---Training new ResNet18---")
        model = resnet18(num_classes = 10).to(DEVICE)
        epochs = 40
        max_lr = 0.01
        grad_clip = 0.1
        weight_decay = 1e-4
        opt_func = torch.optim.Adam

        t_tr = {}
        for t, l in train_ds.imgs:
            t_tr[len(t_tr)] = (t, l)
        t_vl = {}
        for t, l in valid_ds.imgs:
            t_vl[len(t_vl)] = (t, l)

        t_tr = SubData(t_tr, transform_train)
        t_vl = SubData(t_vl, transform_test)

        t_tr_dl = DataLoader(t_tr, 256, shuffle=True)
        t_vl_dl = DataLoader(t_vl, 256*2)

        history = fit_one_cycle(epochs, max_lr, model, t_tr_dl, t_vl_dl, 
                                    grad_clip=grad_clip, 
                                    weight_decay=weight_decay, 
                                    opt_func=opt_func)
        
        torch.save(model.state_dict(), f"data/all/models/ResNET18_CIFAR10_ALL_CLASSES_{n}.pt")

    # same for the exact unlearned model
    if new_baseline:
        print("---Training new Exact Unlearned ResNet18---")
        model = resnet18(num_classes = 10).to(DEVICE)
        epochs = 40
        max_lr = 0.01
        grad_clip = 0.1
        weight_decay = 1e-4
        opt_func = torch.optim.Adam

        # Define train and validation dataloaders for retain data classes
        rt_tr = {}
        for t, l in train_ds.imgs:
            if l not in classes_to_forget:
                rt_tr[len(rt_tr)] = (t, l)
        rt_vl = {}
        for t, l in valid_ds.imgs:
            if l not in classes_to_forget:
                rt_vl[len(rt_vl)] = (t, l)

        rt_tr = SubData(rt_tr, transform_train)
        rt_vl = SubData(rt_vl, transform_test)

        rt_tr_dl = DataLoader(rt_tr, 256, shuffle=True)
        rt_vl_dl = DataLoader(rt_vl, 256*2)

        history = fit_one_cycle(epochs, max_lr, model, rt_tr_dl, rt_vl_dl, 
                                    grad_clip=grad_clip, 
                                    weight_decay=weight_decay, 
                                    opt_func=opt_func)

        if not os.path.exists("data/retrain/models"):
            os.makedirs("data/retrain/models")
        torch.save(model.state_dict(), f"data/retrain/models/ResNET18_CIFAR10_RETRAIN_CLASSES_{n}.pt")
    if n == 0:
        raise Exception("No model found")
    if not new_baseline:
        if idx == -1:
            n = random.randint(0, len(os.listdir("data/all/models"))-1)
        else:
            n = idx
    model = resnet18(num_classes = 10).to(DEVICE)
    model.load_state_dict(torch.load(f"data/all/models/ResNET18_CIFAR10_ALL_CLASSES_{n}.pt", map_location=DEVICE, weights_only=True))

    if model_eval_logs:
        history = [evaluate(model, valid_dl)]
        print(history)

    # classwise list of samples
    num_classes = 10

    classwise_train = {}
    for i in range(num_classes):
        classwise_train[i] = []

    for img_path, label in train_ds.imgs:
        classwise_train[label].append((img_path, torch.tensor(label).to(DEVICE)))

    # getting some samples from retain classes
    num_samples_per_class = 1000 # This one could also be finetuned

    retain_samples = {}
    for i in range(len(classes)):
        if classes[i] not in classes_to_forget:
            retain_samples[len(retain_samples)] = classwise_train[i][:num_samples_per_class]

    # Define train and validation dataloaders for retain data classes
    # retain validation set
    forget_valid = {}
    for t, l in valid_ds.imgs:
        if l in classes_to_forget:
            forget_valid[len(forget_valid)] = (t, l)
    # forget validation set
    retain_valid = {}
    for t, l in valid_ds.imgs:
        if l not in classes_to_forget:
            retain_valid[len(retain_valid)] = (t, l)

    forget_valid_ds = SubData(forget_valid, transform_train)
    retain_valid_ds = SubData(retain_valid, transform_test)
    
    forget_valid_dl = DataLoader(forget_valid_ds, 256, shuffle=True)
    retain_valid_dl = DataLoader(retain_valid_ds, 256*2)

    # loading the model
    model = resnet18(num_classes = 10).to(DEVICE)
    model.load_state_dict(torch.load("ResNET18_CIFAR10_ALL_CLASSES.pt", map_location=DEVICE, weights_only=True))

    if logs:
        print("---Optimizing noise generator---")
    noises = {}
    for cls in classes_to_forget:
        if logs:
            print("Optiming loss for class {}".format(cls))
        # Here is the place of change
        noises[cls] = NoiseGenerator(
            dim_out = [3, 32, 32],
            dim_hidden=t_Layers,
            dim_start=t_Noise_Dim,
            ).to(DEVICE)
        opt = torch.optim.Adam(noises[cls].parameters(), lr = t_Learning_Rate)

        num_epochs = t_Epochs # Changed
        num_steps = t_Steps # Changed
        class_label = cls
        for epoch in range(num_epochs):
            total_loss = []
            for batch in range(num_steps):
                inputs = noises[cls]().to(DEVICE).unsqueeze(0)
                for i in range(batch_size-1):
                    new = noises[cls]().to(DEVICE).unsqueeze(0)
                    inputs = torch.cat((inputs, new), 0)
                labels = torch.zeros(batch_size).to(DEVICE)+class_label
                outputs = model(inputs).to(DEVICE)
                loss = -F.cross_entropy(outputs, labels.long()) + t_Regularization_term*torch.mean(torch.sum(torch.square(inputs), [1, 2, 3])) # Changed
                opt.zero_grad()
                loss.backward()
                opt.step()
                total_loss.append(loss.cpu().detach().numpy())
            if logs:
                print("Loss: {}".format(np.mean(total_loss)))

    batch_size = t_Batch_Size # Changed
    num_batches = t_Number_of_Noise_Batches # Changed

    other_samples = {}
    for i in range(len(retain_samples)):
        for j in range(len(retain_samples[i])):
            other_samples[len(other_samples)] = retain_samples[i][j]

    noisy_data = FeatureMU_Loader(
        noisies=[
            {"gen": noises[cls], "label": torch.tensor(cls), "n": num_batches*t_Batch_Size} for cls in classes_to_forget
        ],
        retain_data=other_samples,
        transform=transform_train,
    )
    noisy_loader = torch.utils.data.DataLoader(noisy_data, batch_size=t_Batch_Size, shuffle = True)

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.02)
    if logs:
        print("---Impairing Phase---")
    for epoch in range(1):  
        model.train(True)
        running_loss = 0.0
        running_acc = 0
        for i, data in enumerate(noisy_loader):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.clone().detach().to(DEVICE)

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

    if model_eval_logs:
        print("Performance of Standard Forget Model on Forget Class")
        history = [evaluate(model, forget_valid_dl)]
        print("Accuracy: {}".format(history[0]["Acc"]*100))
        print("Loss: {}".format(history[0]["Loss"]))

        print("Performance of Standard Forget Model on Retain Class")
        history = [evaluate(model, retain_valid_dl)]
        print("Accuracy: {}".format(history[0]["Acc"]*100))
        print("Loss: {}".format(history[0]["Loss"]))


    other_samples = SubData(other_samples, transform_train)
    heal_loader = torch.utils.data.DataLoader(other_samples, batch_size=t_Batch_Size, shuffle = True)

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

    if logs:
        print("---Repairing Phase---")
    for epoch in range(1):  
        model.train(True)
        running_loss = 0.0
        running_acc = 0
        for i, data in enumerate(heal_loader):
            inputs, labels = data
            inputs, labels = inputs.to(DEVICE), labels.clone().detach().to(DEVICE)

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

    if model_eval_logs:
        print("Performance of Standard Forget Model on Forget Class")
        history = [evaluate(model, forget_valid_dl)]
        print("Accuracy: {}".format(history[0]["Acc"]*100))
        print("Loss: {}".format(history[0]["Loss"]))

        print("Performance of Standard Forget Model on Retain Class")
        history = [evaluate(model, retain_valid_dl)]
        print("Accuracy: {}".format(history[0]["Acc"]*100))
        print("Loss: {}".format(history[0]["Loss"]))

    return model