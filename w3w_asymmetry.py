import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.colors import LinearSegmentedColormap

# christians ugly ass colormap
cdict = {'red': ((0., 1, 1),
                 (0.05, 1, 1),
                 (0.11, 0, 0),
                 (0.66, 1, 1),
                 (0.89, 1, 1),
                 (1, 0.5, 0.5)),
         'green': ((0., 1, 1),
                   (0.05, 1, 1),
                   (0.11, 0, 0),
                   (0.375, 1, 1),
                   (0.64, 1, 1),
                   (0.91, 0, 0),
                   (1, 0, 0)),
         'blue': ((0., 1, 1),
                  (0.05, 1, 1),
                  (0.11, 1, 1),
                  (0.34, 1, 1),
                  (0.65, 0, 0),
                  (1, 0, 0))}

my_cmap = LinearSegmentedColormap('my_colormap',cdict,256)

with h5py.File('//home/wfrisch/Documents/SharedCode/SimpleMansModel/Results/w3w/20170315Neon100pbins2.h5', 'r') as f:
    # read the data
    data = f['asymmetry'].get('Distribution')[...]
    # phi = f['variables'].get('phases')[...]
    p = f['variables'].get('momentum bins')[...]

    # shift by pi/2
    data = np.roll(data,25, axis=0)
    # cut out middle
    data = data[:,10:90]
    # make it 6pi length
    data1 = np.append(data,data,axis=0)
    data1 = np.append(data1,data,axis=0)
    # data1 = np.fliplr(data1)

    # calculate the "asymmetry"
    A = np.sum(data1[:,28:52], axis=1)
    B = np.sum(data1[:,6:23], axis=1)

    # pretty plot
    ax1 = plt.subplot2grid((3,23),(0,0),rowspan=2,colspan=22)
    im = ax1.imshow(data1.T, cmap=my_cmap, aspect='auto', interpolation='none')
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.set_yticks([0,10,20,30,40,50,60,70,80])
    ax1.set_yticklabels([p[i] for i in [10,20,30,40,50,60,70,80,90]])
    ax1.set_ylabel('momentum/au')
    ax1.axhline(y=28,ls='--',c='b')
    ax1.axhline(y=52,ls='--',c='b')
    ax1.axhline(y=6,ls='--',c='r')
    ax1.axhline(y=23,ls='--',c='r')
    plt.title('Neon')

    ax2 = plt.subplot2grid((3,23),(2,0), colspan=22)
    ax2.plot(A)
    ax2.set_xlim((0,150))
    ax2.set_xticks([0,25,50,75,100,125,150])
    ax2.set_xticklabels([0,1,2,3,4,5,6])
    ax2.set_xlabel('relative phase/pi')
    ax2.set_ylabel('yield', color='b')
    ax2.set_yticklabels(ax2.get_yticks(), color='b')

    ax3 = plt.subplot2grid((3,23),(0,22), rowspan=2)
    plt.colorbar(im, ax3, label='yield')

    ax4 = plt.twinx(ax2)
    ax4.plot(B, c='r')
    ax4.set_xlim((0,150))
    ax4.set_ylabel('yield', color='r')
    ax4.set_yticklabels(ax4.get_yticks(), color='r')

    plt.show()
