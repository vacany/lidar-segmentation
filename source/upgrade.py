import torch
import itertools
from tools import datafuns, stats, visuals
import dataloaders
from models.unet import Lil_UNet, UNet
from models import cost
import os
import argparse
import glob
from shutil import rmtree

parser = argparse.ArgumentParser(description="set hyperparameters for learning")
parser.add_argument('--learning_rate', "-lr", default=0.001, type=float, help='set learning rate')
parser.add_argument('--source', '-s', default='cadc', type=str, help='choose source data for training')
parser.add_argument('--mode', '-m', default='cost', type=str, help='describe procedure and save to mode dir')
parser.add_argument('--target', '-t', default='cost', type=str, help='choose ground truth for learning')
parser.add_argument('--batch_size', '-bs', default=1, type=int, help='batch_size value')
parser.add_argument('--load_seg', '-ls', default=False, type=str, help='adress to model state dict')
parser.add_argument('--load_cost', '-lc', default=False, type=str, help='adress to model state dict')
parser.add_argument('--train_file', '-tf', default='*.npz', type=str, help='choose sample for training')
parser.add_argument('--weights', '-w', nargs="+", default=(0,0,0,1,-1), type=float, help='choose weights for training loss')
parser.add_argument('--loss_mode', '-lm', default='cost', type=str, help='describe loss tree logic')
parser.add_argument('--network', '-nn', default='seg', type=str, help='Which to train')
args = parser.parse_args()

source = args.source
mode = args.mode     # define folders, describe procedure
target = args.target    # define label
weights = args.weights
loss_mode = args.loss_mode
network = args.network

cf = 80
bs = args.batch_size
epochs = 150
device = datafuns.get_device(0)
move_cost = 0.1

trn_files = glob.glob(os.path.expanduser("~") + f'/datasets/{source}/data/sample/*.npz')
img_save = os.path.expanduser("~") + f'/datasets/{source}/img/{mode}'
model_save = os.path.expanduser("~") + f'/datasets/{source}/models/{mode}'
drawer = visuals.Bev_drawer()

os.makedirs(model_save, exist_ok=True)
os.makedirs(img_save, exist_ok=True)
rmtree(img_save)
os.makedirs(img_save, exist_ok=True)

# Init
gauss = (0, 0.1)
num_traj = 5
cm_model = UNet(1,1).to(device)
seg_model = Lil_UNet(4,3).to(device)



if args.load_seg:
    model_load = os.path.expanduser("~") + f'/datasets/{source}/models/{args.load_seg}'
    seg_model.load_state_dict(torch.load(model_load))

if args.load_seg:
    model_load = os.path.expanduser("~") + f'/datasets/{source}/models/{args.load_cost}'
    cm_model.load_state_dict(torch.load(model_load))

tst_dataset = dataloaders.NIPS_Dataset(trn_files[0], NUM=cf, target=target, every_th=4)
tst_loader = torch.utils.data.DataLoader(tst_dataset, batch_size=bs)

print(args)

# Init
running_loss = 0



from models.cost import Cat_outputs
from models.seg import Softargmax
cat_layer = Cat_outputs()

if source == 'valeo':
    softargmax = Softargmax(3).to(device)
    metric = stats.IOU_Metric(3)

if source == 'cadc':
    softargmax = Softargmax(2).to(device)
    metric = stats.IOU_Metric(2)


if network == 'seg':
    seg_model.train()
    cm_model.eval()
    optimizer = torch.optim.SGD(seg_model.parameters(), lr=args.learning_rate)

if network == 'cost':
    seg_model.eval()
    cm_model.train()
    optimizer = torch.optim.SGD(cm_model.parameters(), lr=args.learning_rate)

if network == 'both':
    seg_model.train()
    cm_model.train()
    #optimizer = torch.optim.Adam(itertools.chain(seg_model.parameters(), cm_model.parameters()), lr=args.learning_rate)

    optimizer_seg = torch.optim.Adam(seg_model.parameters(), lr=args.learning_rate)
    optimizer_cm = torch.optim.Adam(cm_model.parameters(), lr=args.learning_rate)


criterion = cost.Loss(mode= loss_mode, move_cost=move_cost, num_traj=num_traj, gauss=gauss, weights=weights, device=device)

count = 0
for e in range(350):
    for it, batch in enumerate(tst_loader):
        cm = []
        # all as list
        batch['traj'], batch['odo'], batch['future'], gt = tst_loader.dataset.get_trajectories(batch['frame'])

        #for i in range(len(batch['future'])):
        pred_map = seg_model(batch['future'][0])    # 0 is b

        pred_map = softargmax(pred_map)


        if source == 'valeo':
            metric.build(pred_map[:1], gt[0][0])
        if source == 'cadc':
            metric.build(pred_map[:1], batch['gt'][0] * batch['in'][0, -1])

        res = metric.calculate()
        metric.reset()


        if source == 'valeo':
            cost_map = cm_model(pred_map)  # swap occupancy for prediction
        if source == 'cadc':
            cost_map = cm_model(pred_map)  # swap occupancy for prediction


        # if source == 'valeo':
        #     cost_map = cm_model(torch.cat((batch['future'][0][:,:-2], pred_map), dim=1))  # swap occupancy for prediction
        # if source == 'cadc':
        #     cost_map = cm_model(torch.cat((batch['future'][0][:,:-1], pred_map), dim=1))  # swap occupancy for prediction


        cost_map = torch.nn.functional.softplus(cost_map)  # saturate

        if str(count)[-1] in ['0', '1', '2', '3', '4']:
            criterion = cost.Loss(mode=loss_mode, move_cost=move_cost, num_traj=num_traj, gauss=gauss,
                                  weights=(0, 0, 1, 1, -1), device=device)
        else:
            criterion = cost.Loss(mode=loss_mode, move_cost=move_cost, num_traj=num_traj, gauss=gauss,
                                  weights=(0, 1, 0, 1, -1), device=device)

        cm = cat_layer(batch, cost_map.squeeze(1).unsqueeze(0))

        if network == 'cost':
            l2_loss = (((cost_map[:,0][pred_map[:, 0] != 0] - pred_map[:, 0][pred_map[:, 0] != 0]) ** 2).mean())
            loss = criterion(batch, cm) + l2_loss

        if network == 'seg':
            loss = criterion(batch, cm)

        if network == 'both':
            #l2_loss = (((cost_map[:, 0][pred_map[:, 0] != 0] - pred_map[:, 0][pred_map[:, 0] != 0]) ** 2).mean())
            loss = criterion(batch, cm)# + l2_loss

        running_loss += loss.item()


        loss.backward()

        if network == 'both':
            if str(count)[-1] in ['0', '1', '2','3','4']:

                optimizer_seg.step()
                optimizer_seg.zero_grad()

            if str(count)[-1] in ['5','6','7','8','9']:

                optimizer_cm.step()
                optimizer_cm.zero_grad()

        metric.calculate()

        print(f'Result: {res} \t Loss: {loss:.3f} \t exp: {criterion.exp_loss:.3f}, opt: {criterion.gen_loss:.3f},'
              f' best: {criterion.best_loss:.3f}, sim: {criterion.sim_loss:.3f}')
        #diff = torch.tensor(pred_map[0][0].cpu() == (batch['gt'][0] * batch['in'][0,-1]).cpu(), dtype=torch.float)
        drawer.update(cm[0].detach().cpu().numpy(), expert=batch['traj'][0], optimal=criterion.best_trajs[0].detach().cpu())
        drawer.add_label((pred_map[0].squeeze(0)).detach().cpu().numpy())
        # split valeo and cadc
        img = torch.cat(((batch['gt'][0] * batch['in'][0, -2]).cpu(), batch['in'][0, -2].cpu()), dim=1)
        drawer.add_label(img.detach().cpu().numpy(), dim=0)
        drawer.out(f'{img_save}/{e:03d}.png')
        break

    count += 1
    print(f'epoch: {e:03d}')

    if network == 'cost':
        torch.save(cm_model.state_dict(),f'{model_save}/{e:03d}.pt')

    if network == 'seg':
        torch.save(seg_model.state_dict(),f'{model_save}/{e:03d}.pt')





    #drawer.update(cm[0].detach().cpu().numpy(), expert=batch['traj'][0], optimal=criterion.best_trajs[0])

    #drawer.out(f'{img_save}/{epoch_num:03d}_{it:02d}.png')


    #print(f'Epoch {mode}: {epoch_num:03d} ended')

