import numpy as np
import matplotlib.pyplot as plt

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Sompy
from sompy.sompy import (SOM, SOMFactory)
from sompy.visualization.mapview import View2D
from sompy.visualization.hitmap import HitMapView
from sompy.visualization.umatrix import UMatrixView

# Minisom
from minisom import MiniSom

# Somber
from somber import Som
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Colors
example1 = True

if example1:
    colors = [[0., 0., 0.],
          [0., 0., 1.],
          [0., 0., 0.5],
          [0.125, 0.529, 1.0],
          [0.33, 0.4, 0.67],
          [0.6, 0.5, 1.0],
          [0., 1., 0.],
          [1., 0., 0.],
          [0., 1., 1.],
          [1., 0., 1.],
          [1., 1., 0.],
          [1., 1., 1.],
          [.33, .33, .33],
          [.5, .5, .5],
          [.66, .66, .66]]

    nb = 1000 # lines
    trois = False # 3cols

    if trois: # data.shape = (nb, 3)
        dat1 = np.linspace(0,1,nb) # 0 to 1
        dat2 = np.linspace(1,0,nb) # 1 to 0
        dat3 = (np.linspace(0,1,nb)+0.5)%1 # 0.5 to 1 and 0 to 0.5
        data = np.vstack((dat1,dat2,dat3)).T # 3 tab
    if not trois: # data.shape = (nb, 5)
        dat1 = np.linspace(0,1,nb) # 0 to 1
        dat2 = np.linspace(1,0,nb) # 1 to 0
        dat3 = (np.linspace(0,1,nb)+0.5)%1 # 0.5 to 1 and 0 to 0.5
        dat4 = (np.linspace(0,1,nb)+0.25)%1 #
        dat5 = (np.linspace(0,1,nb)+0.75)%1 #
        data = np.vstack((dat1,
                dat2,
                dat3,
                dat4,
                dat5
                )).T # 5 tabs

    som_x, som_y = [30,30] # SOMs shape

    if False: # MiniSom
        som = MiniSom(som_x, som_y, data.shape[1],
                sigma=0.5, learning_rate=1)
        som.train(data, 20000, verbose=True)

        if True:
            win_tab = np.zeros(som_shape)
            for i in range(nb):
                win_tab[som.winner(data[i])] +=1

            plt.figure(0)
            plt.imshow(win_tab, cmap='gist_rainbow')
            plt.colorbar()

        plt.figure(1)
        plt.imshow(som.distance_map(), cmap='gist_rainbow')
        plt.colorbar()
        plt.show()

    if False: # Somber
        #names = [str(i) for i in range(nb)]

        som = Som((som_x, som_y), learning_rate=0.3)

        # train
        # 10 updates with 10 epochs = 100 updates to the parameters.
        som.fit(data, num_epochs=10, updates_epoch=10)

        mapped = som.map_weights()

        plt.imshow(mapped)
        plt.show()

    if True: # Sompy SOM
        sm = SOMFactory.build(data, mapsize=[som_x, som_y], initialization='pca')
        sm.train(n_job = 1, shared_memory = 'no', verbose='info')

        # Visualization class
        view2D  = View2D(30,30,"Data Map",text_size=14)
        hitmap  = HitMapView(30,30,"Cluster Hit Map",text_size=14)
        umat  = UMatrixView(30,30,"Unified Distance Matrix", text_size=14)

        view2D.show(sm, col_sz=3, denormalize=True)
        hitmap.show(sm)
        umat.show(sm)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Clustering
def example2():
    datas1 = np.random.sample((20,2))
    datas2 = np.random.sample((20,2))+2
    datas3 = np.random.sample((20,2))+4
    datas = np.concatenate([datas1, datas2, datas3])


    som = MiniSom(1, 3, len(datas[0]),
                sigma=.5,
                learning_rate=.5,
                neighborhood_function='gaussian')

    som.train_random(datas, 100, verbose=True)


    winner_coordinates = np.array([som.winner(x) for x in datas]).T
    cluster_index = np.ravel_multi_index(winner_coordinates, (1,3))

    plt.figure(figsize=(8, 8))
    for i in np.unique(cluster_index):
        plt.scatter(datas[cluster_index == i, 0],
                    datas[cluster_index == i,1], label='cluster='+str(i), alpha=.7)

    for center in som.get_weights():
        plt.scatter(center[:, 0], center[:, 1], marker='x',
                    s=80, linewidths=35, color='k', label='center')
    plt.legend()
    plt.show()

#example2()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Digits
def example3():
    from sklearn import datasets
    from sklearn.preprocessing import scale

    # load the digits dataset from scikit-learn
    digits = datasets.load_digits(n_class=10)
    data = digits.data  # matrix where each row is a vector that represent a digit.
    data = scale(data)

    num = digits.target  # num[i] is the digit represented by data[i]

    som = MiniSom(30, 30, 64, sigma=4,
                  learning_rate=0.5, neighborhood_function='triangle')

    som.train(data, 2000, verbose=True)  # random training



    plt.figure(figsize=(8, 8))
    wmap = {}
    im = 0
    for x, t in zip(data, num):  # scatterplot
        w = som.winner(x)
        wmap[w] = im
        plt. text(w[0]+.5,  w[1]+.5,  str(t),
                  color=plt.cm.rainbow(t / 10.), fontdict={'weight': 'bold',  'size': 11})
        im = im + 1
    plt.axis([0, som.get_weights().shape[0], 0,  som.get_weights().shape[1]])
    plt.show()

#example3()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
