import torch
import torch.nn as nn
from tools.visuals import Bev_drawer

class Training():
    def __init__(self, model=None, criterion=None, optimizer=None, metric=None, model_save=None, img_save=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.metric = metric
        self.model_save = model_save
        self.img_save = img_save
        self.softargmax = Softargmax(4)
        self.drawer = Bev_drawer()

    def one_epoch(self, epoch_num, dataloader, extra='', validate=False, break_it=False):
        running_loss = 0
        mode = 'eval' if validate else 'train'
        getattr(self.model, mode)()

        for it, batch in enumerate(dataloader):

            pred = self.model(batch['in'].squeeze(1))

            loss = self.criterion(pred, batch['ps'].squeeze(1))
            running_loss += loss.item()

            self.metric.build(pred, batch['ps'].squeeze(1))
            res = self.metric.calculate()

            if not validate:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            print(f'Mode: {mode} \t Iter batch: {(it) * dataloader.batch_size} / {len(dataloader.dataset.label_data)} \t'
                  f'Running_loss: {running_loss:.3f} \t Loss: {loss:.3f} \t results: {res}', flush=True)

            if validate:
                cls = self.softargmax(pred)

                # for i in range(len(pred)):
                #     cls[i] = cls[i]
                #     self.drawer.labels(cls[i].squeeze(0).detach().cpu().numpy())
                #     self.drawer.add_label((batch['gt'][i,0]).detach().cpu().numpy())
                #     self.drawer.out(f'{self.img_save}/{extra}{it*dataloader.batch_size+i:04d}.png')

            if break_it:
                break

        print(f'Epoch {mode}: {epoch_num:03d}{extra} ended, results: {self.metric.calculate(overall=True)}', flush=True)
        self.metric.reset()
        torch.save(self.model.state_dict(), f'{self.model_save}/{epoch_num:03d}{extra}.pt')

    def run(self, epochs, trn_loader, val_loader=False, extra=''):

        self.one_epoch(epochs, trn_loader, extra=extra, validate=False)

        if val_loader:
            self.one_epoch(epochs, val_loader, extra=extra, validate=True)


class Softargmax(nn.Module):

    def __init__(self, num_cls):
        super().__init__()
        self.num_cls = num_cls

    def forward(self, logits, dim=1):
        y_soft = logits.softmax(dim)
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft

        if self.num_cls == 2:
            gum = ret[:,0] * 0 + ret[:,1] * 1

        if self.num_cls == 3:
            gum = ret[:,0] * 0 + ret[:,1] * 1 + ret[:,2] * 2

        if self.num_cls == 4:
            gum = ret[:,0] * 0 + ret[:,1] * 1 + ret[:,2] * 2 + ret[:,3] * 3

        return gum.unsqueeze(1)
