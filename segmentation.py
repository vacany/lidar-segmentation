import torch
import argparse
import os
import glob
import numpy as np

from datatools import loader, datafuns, stats
from utils import seg
import detectors



parser = argparse.ArgumentParser(description="set hyperparameters for learning")
### Specify training
parser.add_argument('--epochs', "-e", default=1, type=int, help='set number of epochs')
parser.add_argument('--num_classes', '-cls', default=4, type=int, help='Choose number of label categories')
parser.add_argument('--mode', '-m', default='segment', type=str, help='describe procedure and save to mode dir')
parser.add_argument('--target', '-t', default='ps', type=str, help='choose ground truth for learning')
parser.add_argument('--process', '-p', default='trn', type=str, help='Choose between trn (training) and tst(testing)')
### Model hyperparameters
parser.add_argument('--net', default='UNet', type=str, help='Choose model to train or infer')
parser.add_argument('--learning_rate', "-lr", default=0.001, type=float, help='set learning rate')
parser.add_argument('--batch_size', '-bs', default=1, type=int, help='batch_size value')
parser.add_argument('--load', '-l', default=False, type=str, help='adress to model state dict')


args = parser.parse_args()

batch_size = 1
num_classes = 4

data_folder = 'dataset/processed/'
data_file = data_folder + 'example.npz'


dataset = loader.Seg_Dataset(data_file)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
model = detectors.squeezesegv2.SqueezeSegV2(dataset.input_data.shape[1], num_classes)
model = detectors.unet.UNet(4, num_classes)


mode = args.mode     # define folders, describe procedure
target = args.target    # define label
bs = args.batch_size
epochs = args.epochs
num_classes = args.num_classes
device = datafuns.get_device(0)

trn_files = glob.glob(f'dataset/trn/*.npz')
val_files = glob.glob(f'dataset/val/*.npz')
tst_files = glob.glob(f'dataset/tst/*.npz')

os.makedirs(f'output/models/{mode}', exist_ok=True)
os.makedirs(f'output/img/{mode}', exist_ok=True)
img_save = f'output/img/{mode}'
model_save = f'output/models/{mode}'

model.weight_init()

if args.load:
    model_load = os.path.expanduser("~") + f'/datasets/models/{args.load}'
    model.load_state_dict(torch.load(model_load))

# Show training parameters
print(args)

if args.process == 'trn':
    for e in range(epochs):
        for i, trn in enumerate(trn_files):
            val = val_files[np.random.randint(0, len(val_files))]
            ### datasets and optimizers init
            trn_dataset = loader.Seg_Dataset(trn, device=device)
            val_dataset = loader.Seg_Dataset(val, device=device)
            cls, loss_weight = datafuns.class_weights(trn_dataset.label_data)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(loss_weight, dtype=torch.float), reduction='mean').to(device)
            metric = stats.IOU_Metric(num_classes)

            ### Trainer and loader
            trainer = seg.Training(model=model, criterion=criterion, optimizer=optimizer, metric=metric, model_save=model_save, img_save=img_save)
            trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=bs, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=False)
            ### Run training procedure
            trainer.run(e, trn_loader, val_loader, extra=f'_{i:02d}')

### eval mode
if args.process == 'tst':
    for i, tst in enumerate(tst_files):
        ### datasets and optimizers init
        tst_dataset = loader.Seg_Dataset(tst, device=device)
        cls, loss_weight = datafuns.class_weights(tst_dataset.label_data)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(loss_weight, dtype=torch.float), reduction='mean').to(device)
        metric = stats.IOU_Metric(num_classes)

        ### Trainer and loader
        trainer = seg.Training(model=model, criterion=criterion, optimizer=optimizer, metric=metric, model_save=model_save, img_save=img_save)
        tst_loader = torch.utils.data.DataLoader(tst_dataset, batch_size=bs, shuffle=False)

        ### Run testing procedure
        trainer.one_epoch(0, tst_loader, extra=f'_finished', validate=True)
