import os
import numpy as np 
import random
import zstandard
from tqdm import tqdm
import torch
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from timeit import default_timer as timer
import torchvision
from torchvision import datasets, models, transforms

from gamutrf.sample_reader import get_reader
from gamutrf.utils import parse_filename 

from gamutrf_dataset import * 
from gamutrf_model import *

def class_counts(dataset, idx_to_class): 
    class_counts = {v:0 for v in idx_to_class.values()}
    for i in range(len(dataset)): 
        class_counts[idx_to_class[dataset[i][1].item()]] += 1
    return class_counts

# PARAMETERS
label_dirs= {
    'train': {
        #'drone': ['data/gamutrf-birdseye-field-days/leesburg_field_day_2022_06_15/worker1/','data/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/worker1/gamutrf/'], 
        'mavic3': ['data/office/mavic3/'],
        'mini2': ['data/office/mini2/'],
        'skydio2': ['data/office/skydio2/'], 
        'skydiox2': ['data/office/skydiox2/'],
        #'wifi_2_4': ['data/gamutrf-pdx/07_21_2022/wifi_2_4/'], 
        'wifi_5': ['data/gamutrf-pdx/07_21_2022/wifi_5/'],
    }, 
    'validation': {
        'mini2': ['data/gamutrf-birdseye-field-days/leesburg_field_day_2022_06_15/worker1/']
    }, 
    'test': {
        #'mini2': ['data/gamutrf-birdseye-field-days/pdx_field_day_2022_05_26/worker1/gamutrf/']
    },
}
assert 'train' in label_dirs
experiment_name = '2_9_23'
sample_secs = 0.02
nfft = 512
batch_size = 8
num_workers = 19
num_epochs = 4
train_val_test_split = [0.75, 0.05, 0.20]
assert sum(train_val_test_split) == 1
#save_iter = 200
eval_iter = 5000
leesburg_split = False
###

# DATASET
print(f"\n\nTraining Dataset")
dataset = GamutRFDataset(label_dirs['train'], sample_secs=sample_secs, nfft=nfft)
train_dataset = dataset

train_idx = list(range(len(train_dataset)))
random.shuffle(train_idx)

if label_dirs['validation']: 
    print(f"\n\nValidation Dataset from config")
    validation_dataset = GamutRFDataset(label_dirs['validation'], sample_secs=sample_secs, nfft=nfft, idx_to_class=dataset.idx_to_class)
    validation_dataset_counts = validation_dataset.class_counts()
else: 
    print(f"\n\Validation Dataset split from training")
    #train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, (1-train_val_test_split[1], train_val_test_split[1]))
    
    split = round(len(train_idx) * (1-train_val_test_split[1]))
    train_idx, validation_idx = train_idx[:split], train_idx[split:]
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    validation_dataset = torch.utils.data.Subset(dataset, validation_idx)
    validation_dataset_counts = dataset.class_counts(validation_idx)

if label_dirs['test']: 
    print(f"\n\nTest Dataset from config")
    test_dataset = GamutRFDataset(label_dirs['test'], sample_secs=sample_secs, nfft=nfft, idx_to_class=dataset.idx_to_class)
    test_dataset_counts = test_dataset.class_counts()
else: 
    print(f"\n\nTest Dataset split from training")
    #train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, (1-train_val_test_split[2], train_val_test_split[2]))

    split = round(len(train_idx) * (1-train_val_test_split[2]))
    train_idx, test_idx = train_idx[:split], train_idx[split:]
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    test_dataset_counts = dataset.class_counts(test_idx)

train_dataset_counts = dataset.class_counts(train_idx)
train_class_weights = {k:(1/v) for k,v in dataset.class_counts(train_idx).items()}
train_sampler = torch.utils.data.WeightedRandomSampler([train_class_weights[dataset.idx[c,0]] for c in train_idx], len(train_dataset))

print(f"\n\nDataset mapping: {dataset.idx_to_class}")

# n_train = int(np.floor(train_val_test_split[0]*len(dataset)))
# n_validation = int(np.floor(train_val_test_split[1]*len(dataset)))
# n_test = len(dataset) - n_train - n_validation
# train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, (n_train, n_validation, n_test))

# if leesburg_split: 
#     train_val_test_split = [0.77, 0.03, 0.20]
#     all_except_leesburg = [i for (i, idx) in enumerate(dataset.idx) if not('leesburg' in idx[1] and 'field' in idx[1])] 
#     dataset_sub = torch.utils.data.Subset(dataset, all_except_leesburg)
#     train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset_sub, (int(np.ceil(train_val_test_split[0]*len(dataset_sub))), int(np.ceil(train_val_test_split[1]*len(dataset_sub))), int(train_val_test_split[2]*len(dataset_sub))))
#     just_leesburg = [i for (i, idx) in enumerate(dataset.idx) if 'leesburg' in idx[1]]
#     leesburg_subset = torch.utils.data.Subset(dataset, just_leesburg)
#     validation_dataset = torch.utils.data.ConcatDataset((validation_dataset,leesburg_subset))
print(f"\n\nDataset statistics:")
print(f"total = {len(train_dataset)+len(validation_dataset)+len(test_dataset)=}")   
print(f"\n{len(train_dataset)=}, train/total = {100*len(train_dataset)/(len(train_dataset)+len(validation_dataset)+len(test_dataset)):.1f} %")
print(f"{train_dataset_counts}")
print(f"{train_class_weights}")
print(f"\n{len(validation_dataset)=}, validation/total = {100*len(validation_dataset)/(len(train_dataset)+len(validation_dataset)+len(test_dataset)):.1f} %")
print(f"{validation_dataset_counts}")
print(f"\n{len(test_dataset)=}, test/total = {100*len(test_dataset)/(len(train_dataset)+len(validation_dataset)+len(test_dataset)):.1f} %")
print(f"{test_dataset_counts}")
print("\n\n")

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, num_workers=num_workers)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

count_testing = {l:0 for l in dataset.idx_to_class.values()}
for i, (inputs, labels) in enumerate(tqdm(train_dataloader)): 
    labels = labels.detach().cpu().numpy()
    for l in labels: 
        count_testing[dataset.idx_to_class[l]] += 1
    if i%100 == 0: 
        print(count_testing)
quit()
# x,y = next(iter(dataloader))
# for x,y in zip(x,y): 
#     plt.imshow(np.moveaxis(x.cpu().numpy(), 0, -1), aspect='auto', origin='lower', cmap=plt.get_cmap('jet'))
#     plt.colorbar()
#     plt.title(f"{dataset.idx_to_class[y.item()]}")
#     plt.show()
###

# MODEL
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GamutRFModel(
    experiment_name=experiment_name, 
    sample_secs=sample_secs, 
    nfft=nfft, 
    label_dirs=label_dirs, 
    dataset_idx_to_class=dataset.idx_to_class,
)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
###

# TRAIN
def train(model, criterion, optimizer, train_dataloader, validation_dataloader, num_epochs, device):
    min_validation_loss = np.Inf
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0
        start = timer()
        for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
            #print(f"epoch {epoch}/{num_epochs}, iter {i}/{len(train_dataloader)}")
            #print(f"load training data = {timer()-start} seconds")
        
            start = timer() 
            inputs = inputs.to(device)
            labels = labels.to(device)
            #print(f".to(device) = {timer()-start} seconds")
            optimizer.zero_grad()

            start = timer() 
            outputs = model(inputs)
            #print(f"inference = {timer()-start} seconds")
            start = timer() 
            loss = criterion(outputs, labels)
            #print(f"loss = {timer() - start} seconds")
            _, preds = torch.max(outputs, 1)
            correct = torch.sum(preds == labels.data)
            #print(f"loss={loss.item()}, accuracy={correct/len(preds)}")

            start = timer() 
            loss.backward()
            optimizer.step()
            #print(f"backward and step = {timer()-start} seconds")

            # statistics
            running_loss += loss.item() 
            running_corrects += torch.sum(preds == labels.data)
            start = timer() 
            
            # if (i+1)%save_iter == 0:
            #     checkpoint_path = f"resnet18_{experiment_name}_{sample_secs}_{epoch}_current.pt"
            #     model.save_checkpoint(checkpoint_path)

            if (i+1)%eval_iter == 0: 
                validation_loss = validate(model, validation_dataloader, device, dataset.class_to_idx, experiment_name, epoch, i)
                if validation_loss < min_validation_loss: 
                    print(f"\nSaving model checkpoint.")
                    checkpoint_path = f"resnet18_{experiment_name}_{sample_secs}_{epoch}.pt"
                    model.save_checkpoint(checkpoint_path)

        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = running_corrects.double() / (len(train_dataloader)*batch_size)

        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        
    
def validate(model, validation_dataloader, device, class_to_idx, experiment_name, epoch, iteration): 
    model.eval()
    validation_predictions = []
    validation_labels = []
    validation_loss = 0
    with torch.no_grad():
        for j,(data,label) in enumerate(tqdm(validation_dataloader)): 
            #print(f"validating {j}/{len(validation_dataloader)}")

            data = data.to(device)
            label = label.to(device)
            out = model(data)
            loss = criterion(out, label)
            validation_loss += loss.item()
            _, predictions = torch.max(out, 1)
            predictions = predictions.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            validation_predictions.extend(predictions)
            validation_labels.extend(label)
            n_correct = sum(predictions == label)
            #print(validation_predictions)
            #print(validation_labels)
    validation_loss /= len(validation_dataloader)
    disp = ConfusionMatrixDisplay.from_predictions(validation_labels, validation_predictions, labels=list(class_to_idx.values()), display_labels=list(class_to_idx.keys()), normalize='true')
    disp.figure_.savefig(f"confusion_matrix_validation_{experiment_name}_{epoch}_{iteration}.png")
    model.train()

    return validation_loss




#validate(model, validation_dataloader, device, dataset.class_to_idx, "test_exp", 0, 0)

train(model, criterion, optimizer, train_dataloader, validation_dataloader, num_epochs, device)


# Visualize predictions 
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
# model.eval()
# x,y = next(iter(dataloader))
# print(dataset.idx_to_class)
# for x,y in zip(x,y): 
    
#     test_x = x.unsqueeze(0).to(device)
#     test_y = y.unsqueeze(0).to(device)
#     out = model(test_x)

#     _, preds = torch.max(out, 1)
#     correct = preds == test_y.data
#     print(f"out={out}")
#     print(f"label={test_y.item()}, prediction={preds.item()}")
#     print(f"correct={correct.item()}")
    
#     plt.imshow(np.moveaxis(x.cpu().numpy(), 0, -1), aspect='auto', origin='lower', cmap=plt.get_cmap('jet'))
#     plt.colorbar()
#     plt.title(f"{dataset.idx_to_class[y.item()]}")
#     plt.show()
