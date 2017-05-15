import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.colors import LinearSegmentedColormap
# colormap
from matplotlib import cm
# font
from matplotlib.font_manager import FontProperties


# make a colormap from inferno starting from white
cdict = {'red': (),
         'green': (),
         'blue': ()}

for i in np.linspace(0,1,5):
    if i == .5: # white in the middle
        c = [1,1,1] # white
    elif i > .55:
        c = cm.inferno(i-.1)
    else:
        c = cm.inferno(i)
    cdict['red'] = cdict['red'] + ((i, c[0], c[0]),)
    cdict['green'] = cdict['green'] + ((i, c[1], c[1]),)
    cdict['blue'] = cdict['blue'] + ((i, c[2], c[2]),)

my_cmap2 = LinearSegmentedColormap('my_colormap2',cdict,256)

# font
font = FontProperties()
font.set_family('serif')
font.set_size(10)
font.set_weight('light')


with h5py.File('//home/wfrisch/Documents/SharedCode/SimpleMansModel/Results/w2w/20170511Neon.h5', 'r') as f:#20170315Neon100pbins6.h5
    # read the data
    data = f['asymmetry'].get('Distribution')[...]
    # phi = f['variables'].get('phases')[...]
    p = f['variables'].get('momentum bins')[...]

    # shift
    data = np.roll(data,3, axis=0)
    # make it 4pi length

    # calculate the asymmetry map
    asym_map = np.zeros((50,50))
    middle = 52 # trace not perfectly in the middle
    for i in range(middle-4): # loop over momenta
        left = data[:,middle-i-1]
        right = data[:,middle+i]
        a = (left-right)/(left+right+0.05*np.max(data))
        asym_map[:,i] = a

    # calculate the asymmetry
    A = np.zeros(50)
    for i in range(50):
        left = np.sum(data[i,:middle])
        right = np.sum(data[i,middle:])
        A[i] = (left-right)/(left+right+0.05*np.max(data))

    # make it 4pi length
    asym_map = np.append(asym_map,asym_map,axis=0)
    # asym_map = asym_map[:,:40]
    asym_map = np.fliplr(asym_map)
    A = np.append(A,A)


    # pretty plot
    fig=plt.figure(figsize=(0.95*5.59,0.95*5.59*3/4)) # 5.59in = \textwidth of latex, 3/4 ratio
    ax1 = plt.subplot2grid((12,10000),(0,1),rowspan=8,colspan=10000)
    plt.imshow(asym_map.T, cmap=my_cmap2, aspect='auto', interpolation='none',vmin=-.9,vmax=.9)
    ax1.set_xticks([11.5,24.5,36.5,49.5,61.5,74.5,86.5,99.5])
    ax1.set_xticklabels([])
    ax1.set_ylim(50,10)
    ax1.set_yticks([i/0.06 for i in [0.5,1.0,1.5,2.0,2.5]]) # one index is 0.06 a.u. of p
    ax1.set_yticklabels([2.5,2.0,1.5,1.0,0.5], fontproperties=font)
    ax1.set_ylabel('Momentum/a.u.',fontproperties=font, labelpad=0)
    cbar = plt.colorbar()#ticks=[0,5,10,15,20,25,30,35,40])
    # cbar.ax.tick_params(labelsize=10)
    for l in cbar.ax.yaxis.get_ticklabels():
        l.set_family("serif")
        l.set_fontsize(10)
    cbar.set_label('Asymmetry',rotation=90,labelpad=4,fontproperties=font)
    cbar.outline.set_linewidth(.5)
    ax1.spines['top'].set_linewidth(.5)
    ax1.spines['bottom'].set_linewidth(.5)
    ax1.spines['left'].set_linewidth(.5)
    ax1.spines['right'].set_linewidth(.5)

    ax2 = plt.subplot2grid((12,10000),(8,1),rowspan=4,colspan=8000)
    ax2.plot(A-np.mean(A),c='k',lw=2)
    plt.xticks([0,12,25,37,50,62,75,87,100],[0,0.5,1,1.5,2,2.5,3,3.5,4],fontproperties=font)
    ax2.set_xlabel('Relative Phase $\phi$/$\pi$', fontproperties=font, labelpad=0)
    ax2.set_ylabel('Asymmetry', fontproperties=font,labelpad=0)
    plt.yticks(fontproperties=font)
    ax2.set_ylim(-1.2,1.2)
    ax2.spines['top'].set_linewidth(.5)
    ax2.spines['bottom'].set_linewidth(.5)
    ax2.spines['left'].set_linewidth(.5)
    ax2.spines['right'].set_linewidth(.5)

    fig.subplots_adjust(right=0.98,hspace=0)

    plt.show()
