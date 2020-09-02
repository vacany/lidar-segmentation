import torch

from tools import datafuns, stats, visuals
import dataloaders
from models.unet import Lil_UNet, UNet
from models import seg
import os
import argparse
import glob
import sys

parser = argparse.ArgumentParser(description="set hyperparameters for learning")
parser.add_argument('--learning_rate', "-lr", default=0.001, type=float, help='set learning rate')
parser.add_argument('--source', '-s', default='valeo', type=str, help='choose source data for training')
parser.add_argument('--mode', '-m', default='seg_ps_nips', type=str, help='describe procedure and save to mode dir')
parser.add_argument('--target', '-t', default='ps', type=str, help='choose ground truth for learning')
parser.add_argument('--batch_size', '-bs', default=16, type=int, help='batch_size value')
parser.add_argument('--load', '-l', default=False, type=str, help='adress to model state dict')
parser.add_argument('--process', '-p', default='trn', type=str, help='Choose purpose of running script')
args = parser.parse_args()

source = args.source
mode = args.mode     # define folders, describe procedure
target = args.target    # define label
bs = args.batch_size

epochs = 1
device = datafuns.get_device(0)

trn_files = glob.glob(os.path.expanduser("~") + f'/datasets/{source}/data/trn/*.npz')
val_files = glob.glob(os.path.expanduser("~") + f'/datasets/{source}/data/val/*.npz')

drawer = visuals.Bev_drawer(img_resize=(3,3))


os.makedirs(os.path.expanduser("~") + f'/datasets/{source}/models/{mode}', exist_ok=True)
os.makedirs(os.path.expanduser("~") + f'/datasets/{source}/img/{mode}', exist_ok=True)
img_save = os.path.expanduser("~") + f'/datasets/{source}/img/{mode}'
model_save = os.path.expanduser("~") + f'/datasets/{source}/models/{mode}'

#validation = True

# Output log
sys.stdout = open(f'{model_save}/out.log', 'w') # OUTPUT, KEEP
sys.stderr = sys.stdout

if target == 'ps':
    in_ch = 4
    out_ch = 4
    model = Lil_UNet(in_ch, out_ch).to(device)

if source == 'cadc':
    in_ch = 3
    out_ch = 2
    model = UNet(in_ch, out_ch).to(device)

model.weight_init()

if args.load:
    model_load = os.path.expanduser("~") + f'/datasets/{source}/models/{args.load}'
    model.load_state_dict(torch.load(model_load))


print(args)
# Init
if args.process == 'trn':
    for e in range(epochs):
        for i, (trn, val) in enumerate(zip(trn_files, val_files)):
            # datasets and utils
            print(trn)
            trn_dataset = dataloaders.Seg_Dataset(trn, source=source, device=device)
            val_dataset = dataloaders.Seg_Dataset(val, source=source, device=device)
            cls, loss_weight = datafuns.class_weights(trn_dataset.label_data)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(loss_weight, dtype=torch.float), reduction='mean').to(device)
            metric = stats.IOU_Metric(out_ch)

            # Trainer and loader
            trainer = seg.Training(model=model, criterion=criterion, optimizer=optimizer, metric=metric, model_save=model_save, img_save=img_save)
            trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=bs, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=False)

            trainer.run(e, trn_loader, val_loader, extra=f'_{i:02d}')

# eval mode
if args.process == 'val':
    for i, (trn, val) in enumerate(zip(trn_files, val_files)):
        val_dataset = dataloaders.Seg_Dataset(val, source=source, device=device)
        cls, loss_weight = datafuns.class_weights(val_dataset.label_data)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(loss_weight, dtype=torch.float), reduction='mean').to(device)
        metric = stats.IOU_Metric(out_ch)

        # Trainer and loader
        trainer = seg.Training(model=model, criterion=criterion, optimizer=optimizer, metric=metric, model_save=model_save, img_save=img_save)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=False)

        trainer.one_epoch(0, val_loader, extra=f'{os.path.basename(val_dataset.data_file)}_',validate=True)
