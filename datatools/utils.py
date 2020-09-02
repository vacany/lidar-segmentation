import os
import glob
import subprocess
import numpy as np
import torch
from tools import stats

def bash(bashCommand):
    subprocess.Popen(bashCommand.split())
    #output, error = process.communicate()


def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available), memory_available

def print_metric():
    for name in sorted(os.listdir()):
        a = torch.load(name)
        print(f"Seg: {a['metric_seg'].calculate()} \t PS: {a['metric_ps'].calculate()} ")

def print_loss(verbose=False):
    best_trn = np.inf
    best_val = np.inf
    for name in sorted(os.listdir()):
        if 'info' in name:
            a = torch.load(name)
            if not verbose:
                print(f"name: {name} \t trn_loss: {a['trn_loss']:.5f} \t val_loss: {a['val_loss']:.5f}")
            if best_trn > a['trn_loss']:
                best_trn = a['trn_loss']

            if best_val > a['val_loss']:
                best_val = a['val_loss']
                same_trn = a['trn_loss']
                best_name = name

    print(f"\nBest epoch name: {best_name} \t trn_loss: {same_trn:.5f} \t val_loss: {best_val:.5f} \t all_epoch_trn : {best_trn:.5f}")

def print_iou():
    for name in sorted(os.listdir()):
        if 'info' in name:
            a = torch.load(name)
            print(f"\n {name} --- {a['iou']}")

def find_best_iou():
    li = [name for name in sorted(os.listdir()) if 'FINETUNING' not in name]
    li2 = [name for name in sorted(os.listdir()) if 'FINETUNING' in name]
    for folder in (li + li2):
        best_iou = 0
        best_name = 0
        for name in sorted(os.listdir(folder)):
            if 'info' in name:
                a = torch.load(folder + '/' + name, map_location='cpu')
                if a['iou'][1] > best_iou:
                    best_iou = a['iou'][1]
                    best_name = name
        print(f"Intensity mod: {folder} \t Best epoch: {best_name} \t best car IOU: {100*best_iou:.2f}")

def return_best_model_by_iou(folder):
    best_iou = 0
    for name in sorted(os.listdir(folder)):
        if 'info' in name:
            a = torch.load(folder + '/' + name, map_location='cpu')
            if a['iou'][1] > best_iou:
                best_iou = a['iou'][1]
                best_name = name

    model = torch.load(f'{folder}/{best_name[:3]}.pt', map_location='cpu')
    print(f"Intensity mod: {folder} \t Best epoch: {best_name} \t best car IOU: {best_iou} \t model Loaded!")
    return model

def find_best_model(path):
    best_val = np.inf
    best_name = 'none'

    for name in sorted(os.listdir(path)):
        if 'info' in name:
            a = torch.load(path + '/' + name)
            if best_val > a['val_loss']:
                best_val = a['val_loss']
                best_name = name[:3] + '.pt'

    return best_name


def print_minmax(array, dim=0):
    for i in range(array.shape[dim]):
        print(f'Min: \t {np.take(array, i, axis=dim).min()} \t Max: \t {np.take(array, i, axis=dim).max()}')

def count_gt():
    for name in sorted(os.listdir()):
        a = np.load(name, allow_pickle=True)
        print(name, 'gt', np.unique(a['gt'], return_counts=True)[1], '\t ps', np.unique(a['ps'], return_counts=True)[1])

def comparsion():
    dirs = glob.glob('*.pt')
    if len(dirs) > 0:
        num = 0
        for name in sorted(os.listdir()):
            a = torch.load(name)
            if a['metric_seg'].calculate()[1] > a['metric_ps'].calculate()[1]:
                print(f"Seg: {a['metric_seg'].calculate()} \t PS: {a['metric_ps'].calculate()} ")
                num += 1
        print(f'number of better samples: {num}')

    else:
        for dir in sorted(os.listdir()):
            num = 0
            print(dir)
            for name in sorted(os.listdir(dir + '/data')):

                a = torch.load(f'{dir}/data/{name}', map_location='cpu')
                if a['metric_seg'].calculate()[1] > a['metric_ps'].calculate()[1]:
                    print(f"Seg: {a['metric_seg'].calculate()} \t PS: {a['metric_ps'].calculate()} ")
                    num += 1
            print(f'number of better samples: {num}')



def results():
    for dir in sorted(os.listdir()):
            num = 0
            print(dir)
            met_seg = stats.Valeo_IoU(4)
            met_ps = stats.Valeo_IoU(4)

            for name in sorted(os.listdir(dir + '/data')):
                if name.startswith('001'):
                    a = torch.load(f'{dir}/data/{name}', map_location='cpu')
                    met_seg = stats.merge_metrics(met_seg, a['metric_seg'])
                    met_ps = stats.merge_metrics(met_ps, a['metric_ps'])

            print(dir)
            print(stats.results(met_seg, met_ps))
                # if a['metric_seg'].calculate()[1] > a['metric_ps'].calculate()[1]:
                #     print(f"Seg: {a['metric_seg'].calculate()} \t PS: {a['metric_ps'].calculate()} ")
                #     num += 1


            #print(f'number of better samples: {num}')



