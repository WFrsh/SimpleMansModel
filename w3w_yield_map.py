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

    # calculate the asymmetry map
    yield_map = np.zeros((50,50))
    middle = 50
    for i in range(middle): # loop over momenta
        N = data[:,middle+i] + data[:,middle-i-1] # KER resolved total yield
        N_avg = np.mean(N) # yield averaged over all phases
        P = (N - N_avg)/N_avg # probability modulation
        yield_map[:,i] = P

    # make it 4pi length
    yield_map = np.append(yield_map,yield_map,axis=0)
    yield_map = yield_map[:,:40]
    yield_map = np.fliplr(yield_map)

    # pretty plot
    ax1 = plt.subplot2grid((3,23),(0,0),rowspan=3,colspan=22)
    im = ax1.imshow(yield_map.T, cmap=my_cmap, aspect='auto', interpolation='none')
    ax1.set_xticks([])
    ax1.set_xticklabels([])
    ax1.set_yticks([0,10,20,30,40])
    ax1.set_yticklabels([p[i] for i in [90,80,70,60,50]])
    ax1.set_ylabel('momentum/au')
    ax1.set_xticks([0,25,50,75,100])
    ax1.set_xticklabels([0,1,2,3,4])
    ax1.set_xlabel('relative phase/pi')
    plt.title('Probability Modulation Map for Neon')

    # ax2 = plt.subplot2grid((3,23),(2,0), colspan=22)
    # ax2.plot(A)
    # ax2.set_xticks([0,25,50,75,100])
    # ax2.set_xticklabels([0,1,2,3,4])
    # ax2.set_xlabel('relative phase/pi')
    # ax2.set_ylabel('asymmetry')

    ax3 = plt.subplot2grid((3,23),(0,22), rowspan=3)
    plt.colorbar(im, ax3, label='asymmetry')

    plt.show()
