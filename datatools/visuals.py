
import numpy as np
import yaml
import matplotlib.pyplot as plt
from PIL import Image
import torch
import os

class Bev_drawer():

    def __init__(self, config=False, img_resize=(3,3), trans = (1, 0, 2)):
        self.img = np.array(())
        self.trajs = np.array(())
        self.img_resize = img_resize
        self.trans = trans

        self.expert = np.array(())
        self.generated = np.array(())
        self.best = np.array(())
        if config:
            self.config = config
        if not config:
            with open(f'{os.path.expanduser("~")}/datasets/configs/colors.yml', 'r') as f:
                self.config = yaml.load(f, yaml.Loader)

    def update(self, array=None, expert=None, generated=None, best=None):

        if array is not None:
            if type(array) == torch.Tensor:
                array = array.detach().cpu().numpy()
            if array.shape[-1] == 1 or len(array.shape) == 2:
                self.img = (plt.cm.viridis((array - array.min()) / (array.max() - array.min()))[..., :3] * 255).astype('u1')
            if array.shape[-1] == 3:
                self.img = array

        if expert is not None:
            if type(expert) == torch.Tensor:
                expert = expert.detach().cpu().numpy()
            self.expert = np.array(expert)
            self.img[self.expert[:, 0], self.expert[:, 1]] = self.config['EXPERT']
        else:
            self.expert = np.array(())

        if generated is not None:
            if type(generated) == torch.Tensor:
                generated = generated.detach().cpu().numpy()
            self.generated = np.array(generated)
            self.img[self.generated[:, 0], self.generated[:, 1]] = self.config['GEN']
        else:
            self.generated = np.array(())

        if best is not None:
            if type(best) == torch.Tensor:
                best = best.detach().cpu().numpy()
            self.best = np.array(best)
            self.img[self.best[:, 0], self.best[:, 1]] = self.config['BEST']
        else:
            self.best = np.array(())


    def retype(self):
        self.img = self.img.detach().cpu().numpy()
        self.expert = self.expert.detach().cpu().numpy()
        self.generated = self.generated.detach().cpu().numpy()

    def transpose(self):
        self.img = np.transpose(self.img, self.trans)
        self.img = self.img[::-1]

    def resize(self):
        h, w = self.img.shape[:2]
        self.img = Image.fromarray(self.img)
        self.img = self.img.resize((w * self.img_resize[0], h * self.img_resize[1]), Image.NEAREST)    # This swaps axes
        self.img = np.array(self.img)

    def show(self):
        self.resize()
        Image.fromarray(self.img).show()

    def out(self, dest=False, img_return=False, resize=True):
        if resize:
            self.resize()
        if dest:
            Image.fromarray(self.img).save(dest)
        if img_return:
            return self.img

    def labels(self, cls_array):
        self.img = np.zeros((cls_array.shape[0], cls_array.shape[1], 3), dtype=np.uint8)

        for cls, col in enumerate(self.config['LABEL']):

            self.img[cls_array == cls] = col

    def add_image(self, array, dim=1):
        vis_img = (plt.cm.viridis((array - array.min()) / (array.max() - array.min()))[..., :3] * 255).astype('u1')
        self.img = np.concatenate((self.img, vis_img), axis=dim)

    def add_label(self, cls_array, dim=1):
        if type(cls_array) == torch.Tensor:
            cls_array = cls_array.detach().cpu().numpy()

        vis_img = np.zeros((cls_array.shape[0], cls_array.shape[1], 3), dtype=np.uint8)
        for cls, col in enumerate(self.config['LABEL']):
            vis_img[cls_array == cls] = col
        self.img = np.concatenate((self.img, vis_img), axis=dim)

    def four_window(self, st, nd, rd, th):

        return self.img


class Index_GUI():
    def __init__(self, X):

        self.fig, self.ax = plt.subplots(1, 1)

        self.ax.set_title('Use arrow to navigate')
        self.X = X
        if X.shape[1] == 3:
            self.slices, ch, rows, cols = X.shape
            self.X = np.moveaxis(X, 1, 3)
        else:
            self.slices, rows, cols = X.shape

        self.ind = 0

        self.im = self.ax.imshow(self.X[self.ind])
        self.update()

        self.fig.canvas.mpl_connect('key_press_event', self.click)
        plt.show()

    def click(self, event):

        if event.key == 'right':
            self.ind = (self.ind + 1) % self.slices
        elif event.key == 'left':
            self.ind = (self.ind - 1) % self.slices
        elif event.key == 'up':
            self.ind = (self.ind + 50) % self.slices
        elif event.key == 'down':
            self.ind = (self.ind - 50) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind])
        self.ax.set_ylabel(f'{self.ind} /  {self.slices}')
        self.im.axes.figure.canvas.draw()


def look_at_all(key, files=None):
    name_list = []
    drawer = Bev_drawer()
    for name in os.listdir():
        if files is None:
            pass
        elif not name.startswith(files):
            continue

        data_npz = np.load(name, allow_pickle=True)
        data = data_npz[key]

        print(name)
        if key in ['gt', 'ps']:
            new = []
            for i in data:
                drawer.labels(i)
                new.append(drawer.img)

            new = np.stack(new).transpose(0,3,1,2)
            print(new.shape)
            Index_GUI(new)


        else:
            Index_GUI(data)

        var = input('go next?')
        if var.startswith('n'):
            break

        if var.startswith('y'):
            continue

        if var.startswith('s'):
            name_list.append(name)
            continue

    return name_list
