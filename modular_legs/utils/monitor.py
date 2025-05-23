import pdb
import time
import matplotlib
from matplotlib import pyplot as plt
import numpy as np


class Monitor():

    def __init__(self, titles, ranges=None, window=100, shape=None):

        matplotlib.use('TkAgg')
        n_plot = len(titles)
        self.n_plot = n_plot
        self.window = window

        self.x = np.linspace(0,window, num=window)
        self.ys_list = []
        if shape is None:
            shape = (1, n_plot)
        self.fig, axs = plt.subplots(shape[0], shape[1], figsize=(5*shape[1], 4*shape[0]))
        self.axs = axs.flatten()
        self.axs = [self.axs] if n_plot == 1 else self.axs
        self.lines = []
        if ranges is None:
            ranges = [[-1.1, 1.1]]*n_plot
        
        for ax in self.axs:
            line, = ax.plot([], lw=3)
            self.lines.append(line)
        # self.text = self.ax.text(0.8,0.5, "")

        for i in range(n_plot):
            self.axs[i].set_xlim(self.x.min(), self.x.max())
            self.axs[i].set_ylim([ranges[i][0], ranges[i][1]])
            self.axs[i].set_title(titles[i])

        self.fig.canvas.draw()   # note that the first draw comes before setting data 

        # cache the background
        self.axbackgrounds = []
        for ax in self.axs:
            self.axbackgrounds.append(self.fig.canvas.copy_from_bbox(ax.bbox))

        plt.show(block=False)


        self.t_start = time.time()
        self.i = 0

    def plot(self, ys: list):
        self.ys_list.append(ys)
        if len(self.ys_list) > 100:
            self.ys_list.pop(0)

        ys_array = np.array(self.ys_list)
        ys_array = np.pad(ys_array, ((self.window-ys_array.shape[0], 0), (0, 0)), mode='constant')

        # pdb.set_trace()
        # ys_array = (np.zeros((ys_array.shape[0], self.window-ys_array.shape[1])))

        for i in range(self.n_plot):
            self.lines[i].set_data(self.x, ys_array[:,i])
        tx = 'Mean Frame Rate:\n {fps:.3f}FPS'.format(fps= ((self.i+1) / (time.time() - self.t_start)) ) 
        # self.text.set_text(tx)
        #print tx

        # restore background
        for axbackground in self.axbackgrounds:
            self.fig.canvas.restore_region(axbackground)

        # redraw just the points
        for ax, line in zip(self.axs, self.lines):
            ax.draw_artist(line)
        # self.ax.draw_artist(self.text)

        # fill in the axes rectangle
        for ax in self.axs:
            self.fig.canvas.blit(ax.bbox)

        # in this post http://bastibe.de/2013-05-30-speeding-up-matplotlib.html
        # it is mentionned that blit causes strong memory leakage. 
        # however, I did not observe that.


        self.fig.canvas.flush_events()
        #alternatively you could use
        #plt.pause(0.000000000001) 
        # however plt.pause calls canvas.draw(), as can be read here:
        #http://bastibe.de/2013-05-30-speeding-up-matplotlib.html
        self.i += 1


# matplotlib.use('TkAgg')

# # live_update_demo(True)   # 175 fps
# live_update_demo(False) # 28 fps


if __name__ == "__main__":
    m = Monitor(["1", "2"])

    for _ in range(1000000):
        m.plot([np.random.random(), np.random.random()])
        time.sleep(0.01)