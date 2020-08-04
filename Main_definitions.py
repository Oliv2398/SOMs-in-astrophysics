import numpy as np
import matplotlib.pyplot as plt

from minisom import MiniSom

"""
# All package used :

from matplotlib.colors import LogNorm
from matplotlib.colors import to_hex
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection

from ipywidgets import (interactive,
    IntSlider, FloatSlider, RadioButtons, fixed, Layout,
    Checkbox, VBox, interactive_output)

from IPython.display import clear_output

from astropy.io import fits

from sklearn.preprocessing import MinMaxScaler


# Special:

from som_package.minisom_perso.minisom_perso import MiniSom_perso
"""

#------------------------------------------------------------

# random colors
def dat_color(nb=40000, more_dim=0):
    """
    Create a random dataset of colors

    Optional params:
    - nb : int, number of rows (= number of colors)
    - more_dim : int, number of cols
    """
    dat1 = np.random.uniform(0,1,nb)
    dat2 = np.random.uniform(0,1,nb)
    dat3 = np.random.uniform(0,1,nb)

    data = np.vstack((dat1,dat2,dat3)).T

    if more_dim:
        for i in range(3,more_dim):
            np.random.shuffle(dat3)
            data = np.vstack((data.T, dat3)).T

    names = ['Red', 'Green', 'Blue']
    return data

# random normalized uniform colors
def dat_color_norm(nb=40000):
    """
    Create a random uniform dataset of colors

    Optional params:
    - nb : int, number of rows (= number of colors)
    """
    return np.random.dirichlet(np.ones(3),size=(nb))

# soft blue dataset
def data_blue(nb=10000, loc_r=.2, loc_g=.2, loc_b=.7, s_r=.04, s_g=.04, s_b=.05):
    """
    Create a blue dataset

    Optinal params:
    - nb : int, number of rows (= number of colors)
    - loc_r, loc_g, loc_b : floats, mean of the red, green and blue distribution
    - s_r, s_g, s_b : floats, standard deviation of the red, green and blue distribution
    """
    r = np.random.normal(loc_r,s_r,nb) # red
    g = np.random.normal(loc_g,s_g,nb) # green
    b = np.random.normal(loc_b,s_b,nb) # blue ----
    rgb = np.vstack([r,g,b]).T

    # normalize each row
    sum_of_rows = rgb.sum(axis=1)
    data = rgb / sum_of_rows[:, np.newaxis]
    return data

#------------------------------------------------------------

# training
def train_som(data, sigma, learning_rate, iterations,
              topology='rectangular', size='default', verbose=False):
    """
    Params:
    - data : array, training dataset
    - sigma : float, spread of the neighborhood function, needs to be adequate to the dimensions of the map.
    - learning rate : float, determines the step size at each iteration
    - iterations : int, determine the time of the training

    Optional params:
    - verbose : bool, informations along the training
    - topology : str, topology of the map, -rectangular- or -hexagonal-
    - size : tuple, size of the map

    Return :
    - som : MiniSom, trained SOM
    - weights : array, weights of the map
    """

    rows, cols = data.shape

    if size=='default':
        som_x = int(np.sqrt(5*np.sqrt(rows)))
        som_y = som_x
    else:
        try:
            som_x, som_y = size
        except:
            raise ValueError("wrong input for -size-")

    som = MiniSom(som_x, som_y,
                  input_len = cols,
                  sigma = sigma,
                  learning_rate = learning_rate,
                  topology = topology)

    som.random_weights_init(data)

    som.train_random(data, iterations, verbose=verbose)

    weights = som.get_weights()

    return som, weights

#------------------------------------------------------------

# manual training
try :
    from som_package.minisom_perso.minisom_perso import MiniSom_perso

    def train_and_get_error(data, sigma, learning_rate, iterations,
        frequence, topological_error=False, animated=True, sig_view=True):
        """
        Training with a modified MiniSom to get the sigma and learning rate through the process

        Params:
        - data : array, training dataset
        - sigma : float, spread of the neighborhood function, needs to be adequate to the dimensions of the map.
        - learning rate : float, determines the step size at each iteration
        - iterations : int, determine the time of the training
        - frequence : int, learning interval for errors

        Optional params:
        - topological_error : bool, whether the topological error is to be learned or not
        - animated : bool, show the SOM on live during the training
        - sig_view : bool, show the sigma on live during the training

        Return :
        - dict_vars : dict,
            iterations,
            quantization error for the training dataset,
            topological error if -topological_error- is activated,
            mean of the distance map (U-matrix),
            sigma,
            learning rate
        """
        rows, cols = data.shape
        som_x = int(np.sqrt(5*np.sqrt(rows)))
        som_y = som_x

        som = MiniSom_perso(som_x, som_y, cols, sigma, learning_rate)
        som.random_weights_init(data)

        dict_vars ={"iter_x":[],
                    "q_error":[],
                    "mapmean":[],
                    "sigma":[],
                    "learning_rate":[]}

        if topological_error:
            dict_vars["t_error"]=[]

        if animated:
            from IPython.display import clear_output

        for i in range(iterations):
            rand_i = np.random.randint(len(data)) # This corresponds to train_random() method.

            ####
            # modification here, original -som.update()- doesn't return anything
            sigma_i, learning_rate_i = som.update_perso(data[rand_i], som.winner(data[rand_i]), i, iterations)
            ####

            if (i+1) % frequence == 0:
                q_error = som.quantization_error(data)
                dict_vars["q_error"].append(q_error)

                if topological_error:
                    t_error = som.topographic_error(data)
                    dict_vars["t_error"].append(t_error)

                dict_vars["iter_x"].append(i)
                dict_vars["mapmean"].append(np.mean(som.distance_map_perso()))
                dict_vars["sigma"].append(sigma_i)
                dict_vars["learning_rate"].append(learning_rate_i)

                if not animated:
                    print('\r [ %d / %d ] ; %d %%'%(i+1, iterations, 100*(i+1)/iterations), end='')

                if animated: # imshow weights and distance map during the training
                    fig, ax = plt.subplots(1, 2, figsize=(14,7))

                    ax[0].imshow(som.get_weights())
                    ax[0].set_title('sigma %.2f' % sigma_i)
                    ax[0].axis('off')
                    ax[1].imshow(som.distance_map())
                    ax[1].set_title('distance map')
                    ax[1].axis('off')

                    if sig_view:
                        win = som.winner(data[rand_i])
                        circle = plt.Circle(xy = win[::-1], radius = sigma_i, edgecolor='k', fill=False)
                        ax[0].add_artist(circle)

                    plt.suptitle('SOM %d x %d ; iteration [ %d / %d ] - %d %%'%(som_x, som_y, i+1, iterations, 100*(i+1)/iterations))
                    plt.show()
                    clear_output(wait=True)

        return dict_vars

except:
    def train_and_get_error(data, sigma, learning_rate, iterations,
        frequence, topological_error=False, animated=True, sig_view=True):
        """
        Manual training to see the evolution all along.

        Params:
        - data : array, training dataset
        - sigma : float, spread of the neighborhood function, needs to be adequate to the dimensions of the map.
        - learning rate : float, determines the step size at each iteration
        - iterations : int, determine the time of the training
        - frequence : int, learning interval for errors

        Optional params:
        - topological_error : bool, whether the topological error is to be learned or not
        - animated : bool, show the SOM on live during the training

        Return :
        - dict_vars : dict,
            iterations,
            quantization error for the training dataset,
            topological error if -topological_error- is activated
        """
        rows, cols = data.shape
        som_x = int(np.sqrt(5*np.sqrt(rows)))
        som_y = som_x

        som = MiniSom(som_x, som_y, cols, sigma, learning_rate)
        som.random_weights_init(data)

        dict_vars ={"iter_x":[],
                    "q_error":[]}

        if topological_error:
            dict_vars["t_error"]=[]

        if animated:
            from IPython.display import clear_output

        for i in range(iterations):
            rand_i = np.random.randint(len(data)) # This corresponds to train_random() method.

            som.update(data[rand_i], som.winner(data[rand_i]), i, iterations)

            if (i+1) % frequence == 0:
                dict_vars["iter_x"].append(i)
                q_error = som.quantization_error(data)
                dict_vars["q_error"].append(q_error)

                if topological_error:
                    t_error = som.topographic_error(data)
                    dict_vars["t_error"].append(t_error)

                if not animated:
                    print('\r [ %d / %d ] ; %d %%'%(i+1, iterations, 100*(i+1)/iterations), end='')

                if animated: # imshow weights and distance map during the training
                    fig, ax = plt.subplots(1, 2, figsize=(14,7))

                    ax[0].imshow(som.get_weights())
                    ax[0].axis('off')
                    ax[1].imshow(som.distance_map())
                    ax[1].set_title('distance map')
                    ax[1].axis('off')

                    plt.suptitle('SOM %d x %d ; iteration [ %d / %d ] - %d %%'%(som_x, som_y, i+1, iterations, 100*(i+1)/iterations))
                    plt.show()
                    clear_output(wait=True)

        return dict_vars

def plot_error(dict_vars):
    """
    Some plots from the manual training
    """

    if "sigma" in dict_vars:
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(12,10))
    else:
        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(10,10))


    ax[0,0].plot(dict_vars["iter_x"], dict_vars["q_error"])
    ax[0,0].set_ylabel('quantization error')

    if "t_error" in dict_vars:
        ax[1,0].plot(dict_vars["iter_x"], dict_vars["t_error"])
        ax[1,0].set_ylabel('topological error')

        ax[2,0].plot(dict_vars["iter_x"], dict_vars["mapmean"])
        ax[2,0].set_ylabel('moyenne de la distance map')
        ax[2,0].set_xlabel('iterations')
    else:
        ax[1,0].plot(dict_vars["iter_x"], dict_vars["mapmean"])
        ax[1,0].set_ylabel('moyenne de la distance map')
        ax[1,0].set_xlabel('iterations')

        ax[2,0].axis('off')


    if "sigma" in dict_vars:
        ax[0,1].plot(dict_vars["iter_x"], dict_vars["sigma"])
        ax[0,1].set_ylabel("sigma")
        ax[1,1].plot(dict_vars["iter_x"], dict_vars["learning_rate"])
        ax[1,1].set_ylabel('learning rate')
        ax[1,1].set_xlabel('iterations')
        ax[1,1].set_xticks(np.arange(0, 300, step=50))

        ax[2,1].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=.4, hspace=.03)
    plt.show()

#------------------------------------------------------------

# quantization and topographic error plots for different sigma
def multi_sigma_train(nb_sigma, data):
    """
    Quantization and topographic error plots for different sigma.

    Params:
    - nb_sigma : int, number of sigma to be tested
    - data : array, training dataset

    Return:
    - weights_multi : array, weigths from all the trainings
    - q_error : array, quantization error from all the trainings
    - t_error : array, topographic error from all the trainings
    - sigma : array, all sigma used for the trainings
    """
    sigma = np.linspace(1.5,14,nb_sigma)
    learning_rate = 1

    q_error = np.zeros(nb_sigma)
    t_error = np.zeros(nb_sigma)

    weights_multi = []

    for i, sig in enumerate(sigma):
        som, weights = train_som(data, sig, learning_rate, 2000, size=(30,30))
        weights_multi.append(weights)
        q_error[i] = som.quantization_error(data)
        t_error[i] = som.topographic_error(data)
        print("\r",i+1, "/", nb_sigma, end='')

    return weights_multi, q_error, t_error, sigma

# weights for different sigma
def multi_sigma_plot(weights_multi, q_error, t_error, sigma_multi, choice='errors', idx=[0,-1]):
    """
    Params:
    - weights_multi : array, weigths from all the trainings
    - q_error : array, quantization error from all the trainings
    - t_error : array, topographic error from all the trainings
    - choice : str or list, "errors", "sigmas" or both

    Optional params:
    - idx : list, sigma_multi index
    """

    if 'errors' in choice:
        plt.figure(figsize=(12,4))
        plt.subplot(121)
        plt.plot(sigma_multi, q_error, 'xk')
        plt.xlabel('sigma')
        plt.ylabel('quantization error')

        plt.subplot(122)
        plt.plot(sigma_multi, t_error, 'xk')
        plt.xlabel('sigma')
        plt.ylabel('topographic error')
        plt.show()

    if 'sigmas' in choice:
        plt.figure(figsize=(12,6))
        for i,j in enumerate(idx):
            plt.subplot(1,len(idx),i+1)
            plt.imshow(weights_multi[j])
            plt.title('q error=%.3f ; t_error=%.3f' % (q_error[j], t_error[j]))
            plt.axis('off')
        plt.tight_layout()
        plt.show()

#------------------------------------------------------------

# colorbar for subplots
def _colorbars_perso(ax):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size="5%", pad=0.07)
    fig = ax.get_figure()
    fig.add_axes(ax_cb)
    return ax_cb

# subplots
def PlotSOMs(som, var_names=(["R","G","B"]), topology='rectangular', rescale_weigths=False, colorbars=False):
    """
    Show the SOM, the distance map and the variables weights

    Params:
    - som : MiniSom, trained SOM

    Optional params:
    - var_names : list, variable names
    - topology : str, -rectangular- or -hexagonal-
    - rescale_weigths : bool, in case the SOM is uninterpretable
    - colorbars : bool, adding colorbars the plots, only rectangular topology
    """
    weights = som.get_weights().copy()
    weights2 = weights.copy()
    som_x, som_y, cols = weights.shape

    fig, ax = plt.subplots(nrows=2, ncols=cols, figsize=(14,9))

    # rescale weights for a better visualization
    if rescale_weigths:
        for i in range(cols):
            weights[:,:,i] = (weights[:,:,i] - np.min(weights[:,:,i]))/(np.max(weights[:,:,i])-np.min(weights[:,:,i]))


    # rectangular subplots
    #------------------------------------
    if topology=='rectangular':
        # distance map
        ax1 = ax[0,1].imshow(som.distance_map(), cmap='bone')
        if colorbars:
            ax_cb1 = _colorbars_perso(ax[0,1])
            plt.colorbar(ax1, cax=ax_cb1)

        # variables plots
        for i in range(cols):
            axi = ax[1,i].imshow(weights2[:,:,i])
            if colorbars:
                ax_cbi = _colorbars_perso(ax[1,i])
                plt.colorbar(axi, cax=ax_cbi)


        if rescale_weigths and cols==4:
            weights[:,:,-1] = weights[:,:,-1]*0+1

        # SOMs
        if cols in (3, 4):
            ax[0,0].imshow(weights)

        elif cols<=2: # if data shape == 2, adding a dimension to the imshow
            cube = np.zeros((som_x,som_y,3))
            for i in range(cols):
                cube[:,:,i] = weights[:,:,i]
            ax[0,0].imshow(cube)

        else:
            print("Can't show a %dD matrix "%(cols))
            print("-----------------------\n")


    # hexagonal subplots
    #------------------------------------
    if topology=='hexagonal':
        from matplotlib.patches import RegularPolygon
        from matplotlib.collections import PatchCollection
        from matplotlib.colors import to_hex

        xx, yy = som.get_euclidean_coordinates()
        wy = yy*np.sqrt(3)/2
        umatrix = som.distance_map().flatten()

        pixel_color0 = [] ; pixel_color1 = []
        pixel_colori = []
        wre = weights.reshape(np.prod(weights.shape[:2]), cols)
        for i, j in zip(wre, umatrix): # colors
            pixel_color0.append(i)
            pixel_color1.append(plt.cm.bone(j))

        # create patches
        patch_list0 = [] ; patch_list1 = []
        patch_listi = [[] for i in range(cols)]
        for c0, c1, x, y in zip(pixel_color0, pixel_color1, xx.flat, wy.flat):
            # distance map
            patch_list1.append(
                    RegularPolygon(xy = (x, y),
                                numVertices = 6,
                                radius = .95/np.sqrt(3)+.03,
                                facecolor = c1))
            # layers
            for idx, k in enumerate(c0):
                fcolor = (k-np.min(weights[:,:,idx]))/(np.max(weights[:,:,idx])-np.min(weights[:,:,idx]))
                patch_listi[idx].append(
                    RegularPolygon(xy = (x, y),
                                numVertices = 6,
                                radius = .95/np.sqrt(3)+.03,
                                facecolor = plt.cm.viridis(fcolor)))

            if len(c0)==2:
                c0 = np.concatenate((c0,[0]))
            # SOM
            patch_list0.append(
                    RegularPolygon(xy = (x, y),
                                numVertices = 6,
                                radius = .95/np.sqrt(3)+.03,
                                facecolor = to_hex(c0)))

        # SOM
        pc0 = PatchCollection(patch_list0, match_original=True)
        ax[0,0].add_collection(pc0)

        # distance map
        pc1 = PatchCollection(patch_list1, match_original=True)
        ax1 = ax[0,1].add_collection(pc1)

        # each layer of the weights
        for i in range(cols):
            pci = PatchCollection(patch_listi[i], match_original=True)
            axi = ax[1,i].add_collection(pci)
            if colorbars:
                ax_cbi = _colorbars_perso(ax[1,i])
                plt.colorbar(axi, cax=ax_cbi)

        # limits and axis
        for i in range(2):
            for j in range(cols):
                ax[i,j].axis([-1, som_x, -.7, som_y*np.sqrt(3)/2])
                ax[i,j].set_aspect('equal')


    # invisible axis
    for i in range(2):
        for j in range(cols):
            ax[i,j].axis('off')

    # titles
    ax[0,0].set_title('SOM '+str(som_x)+' x '+str(som_y))
    ax[0,1].set_title('distance map')
    for i in range(cols):
        try:
            ax[1,i].set_title(var_names[i])
        except:
            pass

    plt.tight_layout()
    if colorbars:
        plt.subplots_adjust(bottom=.25)
    plt.show()

#------------------------------------------------------------

# hitmap
def Hitmap(som, data, topology="rectangular", normed=True, hit_count=True, colorbars=False, figsize='default', fontsize=None,
           compare=None, hit_count_compare=False, normed_compare=False):
    """
    Show the activation response of the SOM to a certain dataset

    Params:
    - som : MiniSom, trained SOM
    - data : array, activation dataset

    Optional params:
    - topology : str, -rectangular- or -hexagonal-
    - normed : bool, imshow with LogNorm
    - hit_count : bool, number of hit in each cell
    - colorbars : bool, add colorbar to the figure(s) (only on rectangular topology)
    - figsize : tuple, size of the figure
    - fontsize : int, size of the fonts (hit_count must be activated)
    - compare : array, to compare the hitmap with an array (could be a weight or other) of dimension 1
    - hit_count_compare : bool, number of hit in each cell of the compared dataset (work for activation_response data only)
    - normed_compare : bool, imshow with LogNorm of the compared dataset
    """

    # activation response
    activ_resp = som.activation_response(data)

    # size of the figure
    if figsize=='default':
        figsize=(14,14)
    else:
        pass

    # subplots creation
    if compare is not None:
        fig, ax = plt.subplots(1,2,figsize=figsize)
    else:
        fig, ax = plt.subplots(1,figsize=figsize)


    from matplotlib.colors import LogNorm


    # rectangular imshow
    #------------------------------------
    if topology=='rectangular':

        if normed:
            norm = LogNorm()
        else:
            norm = None

        if normed_compare:
            norm_compare = LogNorm()
        else:
            norm_compare = None

        if compare is not None:
            ax0 = ax[0].imshow(activ_resp, norm = norm)
            ax1 = ax[1].imshow(compare, norm = norm_compare)
            if colorbars:
                ax_cb0 = _colorbars_perso(ax[0])
                ax_cb1 = _colorbars_perso(ax[1])
                plt.colorbar(ax0, cax=ax_cb0)
                plt.colorbar(ax1, cax=ax_cb1)
        else:
            ax1 = ax.imshow(activ_resp, norm = norm)
            if colorbars:
                ax_cb1 = _colorbars_perso(ax)
                plt.colorbar(ax1, cax=ax_cb1)

        if hit_count:
            som_x, som_y = som.get_weights().shape[:2]

            for i in range(som_x):
                for j in range(som_y):
                    if activ_resp[i,j]!=0: # don't show inactivated cells
                        # i,j inverted in plt.text because of the minisom's coordinates problem
                        if compare is not None:
                            ax[0].text(j, i, int(activ_resp[i,j]),
                                    horizontalalignment='center',
                                    verticalalignment='center', fontsize=fontsize)
                            if hit_count_compare and compare[i,j]!=0:
                                ax[1].text(j, i, int(compare[i,j]),
                                        horizontalalignment='center',
                                        verticalalignment='center', fontsize=fontsize)

                        else:
                            ax.text(j, i, int(activ_resp[i,j]),
                                    horizontalalignment='center',
                                    verticalalignment='center', fontsize=fontsize)

    # hexagonal imshow
    #------------------------------------
    if topology=='hexagonal':
        from matplotlib.patches import RegularPolygon
        from matplotlib.collections import PatchCollection

        lognorm = LogNorm(1,np.max(activ_resp))

        xx, yy = som.get_euclidean_coordinates()
        wy = yy*np.sqrt(3)/2

        som_x, som_y = som.get_weights().shape[:2]

        fcolor=[]
        for z in activ_resp.flatten():
            if z!=0:
                fcolor.append(plt.cm.viridis(lognorm(z)))
            elif z==0:
                fcolor.append([1,1,1,1])
            else:
                fcolor.append(plt.cm.viridis(z))

        patch_list = []
        for c, x, y in zip(fcolor, xx.flat, wy.flat):
            patch_list.append(
                    RegularPolygon(xy = (x, y),
                                numVertices = 6,
                                radius = .95/np.sqrt(3)+.03,
                                facecolor = c))

        pc = PatchCollection(patch_list, match_original=True)


        if compare is not None:
            ax[0].add_collection(pc)

            compare2 = compare.copy()
            compare2 =  compare2.flatten()
            compare2 = (compare2 - np.min(compare2))/(np.max(compare2) - np.min(compare2))
            pixel_color0 = plt.cm.viridis(compare2)
            patch_list1=[]
            for c0,x,y in zip(pixel_color0, xx.flat, wy.flat):
                patch_list1.append(RegularPolygon(xy = (x, y),
                                                numVertices = 6,
                                                radius = .95/np.sqrt(3)+.03,
                                                facecolor = c0))
            pc1 = PatchCollection(patch_list1, match_original=True)
            ax[1].add_collection(pc1)
        else:
            ax.add_collection(pc)

        for i in range(som_x):
            for j in range(som_y):
                if activ_resp[i,j]!=0 and hit_count:
                    if compare is not None:
                        ax[0].text(xx[i, j], wy[i,j], int(activ_resp[i,j]),
                                horizontalalignment='center',
                                verticalalignment='center', fontsize=fontsize)
                    else:
                        ax.text(xx[i, j], wy[i,j], int(activ_resp[i,j]),
                                horizontalalignment='center',
                                verticalalignment='center', fontsize=fontsize)

        if compare is not None:
            ax[0].axis([-1, som_x, -.7, som_y*np.sqrt(3)/2])
            ax[1].axis([-1, som_x, -.7, som_y*np.sqrt(3)/2])
            ax[0].set_aspect('equal')
            ax[1].set_aspect('equal')
        else:
            ax.axis([-1, som_x, -.7, som_y*np.sqrt(3)/2])
            ax.set_aspect('equal')


    if compare is not None:
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()

def Hitmap2(som, data, compare, normed=True, hit_count=True, fontsize=None):
    """
    Special use, hexagonal compare weights
    """

    activ_resp = som.activation_response(data)

    fig, ax = plt.subplots(1,2, figsize=(14,14))

    from matplotlib.colors import LogNorm
    from matplotlib.patches import RegularPolygon
    from matplotlib.collections import PatchCollection

    lognorm = LogNorm(1,np.max(activ_resp))

    xx, yy = som.get_euclidean_coordinates()
    wy = yy*np.sqrt(3)/2

    som_x, som_y, cols = som.get_weights().shape

    fcolor=[]
    for z in activ_resp.flatten():
        if z!=0:
            fcolor.append(plt.cm.viridis(lognorm(z)))
        elif z==0:
            fcolor.append([1,1,1,1])
        else:
            fcolor.append(plt.cm.viridis(z))

    patch_list = []
    for c, x, y in zip(fcolor, xx.flat, wy.flat):
        patch_list.append(
                RegularPolygon(xy = (x, y),
                            numVertices = 6,
                            radius = .95/np.sqrt(3)+.03,
                            facecolor = c))

    compare2 = compare.copy()
    wre = compare2.reshape(som_x*som_y, cols)
    pixel_color0 = wre.copy()

    patch_list1=[]
    for c0,x,y in zip(pixel_color0, xx.flat, wy.flat):
        patch_list1.append(RegularPolygon(xy = (x, y),
                                        numVertices = 6,
                                        radius = .95/np.sqrt(3)+.03,
                                        facecolor = c0))

    pc = PatchCollection(patch_list, match_original=True)
    ax[0].add_collection(pc)

    pc1 = PatchCollection(patch_list1, match_original=True)
    ax[1].add_collection(pc1)


    for i in range(som_x):
        for j in range(som_y):
            if activ_resp[i,j]!=0 and hit_count:
                ax[0].text(xx[i, j], wy[i,j], int(activ_resp[i,j]),
                        horizontalalignment='center',
                        verticalalignment='center', fontsize=fontsize)

    ax[0].axis([-1, som_x, -.7, som_y*np.sqrt(3)/2])
    ax[1].axis([-1, som_x, -.7, som_y*np.sqrt(3)/2])
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[0].axis('off')
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()

#------------------------------------------------------------

# interactive SOM
def _interactive_som(data, names, sigma, learning_rate, iterations, topology, size, info):
    """The function to which the interactive widgets are tied"""
    som, wts = train_som(data, sigma, learning_rate, iterations,
        topology, size)

    PlotSOMs(som, names, topology=topology,
        rescale_weigths=False, colorbars=False)

    if info:
        print('quantization error :', som.quantization_error(data))
        if topology=='rectangular':
            print('topographic error :', som.topographic_error(data))

def interactive_plot(data, size='default', names=(["R","G","B"]), infos=False):
    """
    Interactive SOM, the sigma, learning rate and iterations can be changed by sliders and the topology by a button.

    Params:
    - data : array, training dataset

    Optional params :
    - size : tuple, size of the SOM
    - names : list, variable names
    - infos : bool, information on SOM's errors
    """
    from ipywidgets import (interactive,
        IntSlider, FloatSlider, RadioButtons, fixed, Layout)

    if size=='default':
        som_x = int(np.sqrt(5*np.sqrt(data.shape[0])))
        som_y = som_x
    else:
        try:
            som_x, som_y = size
        except:
            raise ValueError("wrong input for -size-")

    layout = Layout(width='50%', height='20px')

    interact = interactive(_interactive_som,
                           data = fixed(data),
                           names = fixed(names),
                           sigma = FloatSlider(min=1,
                                               max=int(som_x/2.01),
                                               step=0.2,
                                               value=int(som_x/4),
                                               layout=layout),
                           learning_rate = FloatSlider(min=0.1,
                                                       max=5,
                                                       step=0.1,
                                                       value=1,
                                                       layout=layout),
                           iterations = IntSlider(min=20,
                                                  max=5000,
                                                  step=20,
                                                  value=1500,
                                                  layout=layout),
                           topology = RadioButtons(
                                options=['rectangular', 'hexagonal'],
                                value='rectangular',
                                layout={'width': 'max-content'},
                                description='topology:'),
                           size = fixed(size),
                           info = fixed(infos))
    return interact

#------------------------------------------------------------

# 3D plot of the weights in color
def weights_3D(weights, weights2=None):
    """
    3D plot of the weights in color.

    Params:
    - weights : array, data to plot

    Optional params:
    - weights2 : array, data to plot
    """
    if weights.ndim in (3,4):
        som_x, som_y, cols = weights.shape
        wr = weights.reshape(som_x * som_y, cols)
    else:
        wr = weights.copy()

    if weights2 is not None:
        if weights2.ndim in (3,4):
            som_x2, som_y2, cols2 = weights2.shape
            wr2 = weights2.reshape(som_x2 * som_y2, cols2)
        else:
            wr2 = weights2.copy()


    fig = plt.figure(figsize=(18,7))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter3D(wr[:,0], wr[:,1], wr[:, 2], c=wr)
    ax1.set_xlabel('R')
    ax1.set_ylabel('G')
    ax1.set_zlabel('B')
    ax1.view_init(30, 20)


    if weights2 is not None:
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter3D(wr2[:,0], wr2[:,1], wr2[:, 2], c=wr2)
        ax2.set_xlabel('R')
        ax2.set_ylabel('G')
        ax2.set_zlabel('B')
        ax2.view_init(30, 20)

    plt.show()

# cut extrem values in a dataset
def cut_extrem(data, inf=0.15, sup=0.8):
    """
    cut extrem values in a dataset.

    Params:
    - data : array, dataset

    Optional params:
    - inf : float, lower bound
    - sup : float, upper bound
    """
    new_triangle = data[np.where((data[:, 0] < sup) & (data[:, 1] < sup) & (data[:, 2] < sup) & (data[:, 0] > inf) & (data[:, 1] > inf) & (data[:, 2] > inf))]
    return new_triangle

#------------------------------------------------------------
#---------------------- Catalog defs ------------------------
#------------------------------------------------------------

# loading fits file
def load_cat(path, cat_name):
    """
    Load galaxy catalogs from fits file

    Params:
    - path : str, path and name of the file
    - cat_name : str, 'CS' for COSMOS or 'TU' for True Universe

    Return:
    - data_ex : dict, catalog
    """
    if cat_name not in ('CS','TU'):
        raise ValueError("Choose between 'CS' and 'TU' catalog")

    from astropy.io import fits

    with fits.open(path) as hdul:
        cat = hdul[1].data

    if cat_name=='CS':
        data_ex = {"mag" : cat['mag_auto'],
                    "hlr" : cat['sersicfit'][:,1],
                    "sersic" : cat['sersicfit'][:,2],
                    "q" : cat['sersicfit'][:,3]}
    if cat_name=='TU':
        data_ex = {"mag" : cat['mag'],
                "hlr" : cat['half_light_radius'],
                "sersic" : cat['SSersic_n'],
                "q" : cat['q']}
    return data_ex

# merge dict and keep values of common keys in list
def mergeDict(list_dicts):
    """
    Merge all TU dictionaries

    Params:
    - list_dicts : list, list of all TU dict

    Return:
    - dict_fuse : dict, dict merge
    """
    dict_fuse = list_dicts[0].copy()
    for key, value in dict_fuse.items():
        dict_fuse[key] = np.concatenate([list_dicts[i][key] for i in range(len(list_dicts))])

    return dict_fuse

#------------------------------------------------------------

# remove hlr and sersic issues
def delete_issues(cat_cs_init, cat_tu_init, infos=False):
    """
    Remove hlr and sersic issues.

    Params:
    - cat_cs_init : dict, COSMOS dict
    - cat_tu_init : dict, True Universe dict

    Optional params:
    - infos : bool, print the number of deleted elements

    Return:
    The two dict.
    """
    cat_cs = cat_cs_init.copy()
    cat_tu = cat_tu_init.copy()

    # CS hlr convert
    cat_cs["hlr"] *= 0.03*np.sqrt(cat_cs["q"])

    # delete hlr issues in CS
    idx = np.where(cat_cs['hlr']>10)[0]
    if infos:
        print("hlr CS : nb d'elements suppr", idx.shape[0])
    for i in cat_cs:
        cat_cs[i] = np.delete(cat_cs[i], idx)

    # delete sersic issues in CS
    idx_sup = np.where(cat_cs['sersic']>max(cat_cs['sersic'])-.001)[0]
    idx_inf = np.where(cat_cs['sersic']<min(cat_cs['sersic'])+.001)[0]
    if infos:
        print("sersic CS : nb d'elements suppr", idx_sup.shape[0]+idx_inf.shape[0])
    for i in cat_cs:
        cat_cs[i] = np.delete(cat_cs[i],np.hstack([idx_sup,idx_inf]))

    # delete hlr issues in TU
    idx = np.where(cat_tu['hlr']>10)[0]
    if infos:
        print("hlr TU : nb d'elements suppr", idx.shape[0])
    for i in cat_tu:
        cat_tu[i] = np.delete(cat_tu[i], idx)

    return cat_cs, cat_tu

# rescale the variables of the catalogs
def rescale_cats(cat_cs_init, cat_tu_init):
    """
    Rescale/normalize distributions.

    Params:
    - cat_cs_init : dict, COSMOS dict
    - cat_tu_init : dict, True Universe dict

    Return:
    The two dict.
    """
    cat_cs = cat_cs_init.copy()
    cat_tu = cat_tu_init.copy()

    # normalize mag
    cat_cs['mag'] /= 35
    cat_tu['mag'] /= 35

    # rescale hlr - sersic
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    for i in cat_cs:
        if i!="q" and i!="mag":
            # CS rescale
            X = cat_cs[i].reshape(-1,1).copy()
            scaler.fit(X)
            cat_cs[i] = scaler.transform(X)
            cat_cs[i] = cat_cs[i].flatten()

            # TU rescale
            Y = cat_tu[i].reshape(-1,1).copy()
            scaler.fit(Y)
            cat_tu[i] = scaler.transform(Y)
            cat_tu[i] = cat_tu[i].flatten()

    # rescale hlr
    cat_cs["hlr"] = (cat_cs["hlr"]-0)/(.3-0)
    cat_tu["hlr"] = (cat_tu["hlr"]-0)/(.3-0)

    return cat_cs, cat_tu

# compare COSMOS and TU histograms
def compare_CS_TU(cat1, cat2, norm=True):
    """
    Compare COSMOS and TU histograms

    Params:
    - cat1 : dict, first catalog
    - cat2 : dict, second catalog

    Optional params:
    - norm : bool, if the catalogs are normed or not
    """

    bins = {"mag":200,"hlr":500,"sersic":300,"q":200}
    if norm:
        limits = {"mag":[-.1,1.1],"hlr":[-.1,1.1],"sersic":[-.1,1.1],"q":[-.1,1.1]}
    else:
        limits={"mag":[15,35],"hlr":[-.1,2],"sersic":[-.1,6.1],"q":[-.05,1.05]}

    plt.figure(figsize=(20,4))
    for i, vars in enumerate(cat1):
        plt.subplot(1,4,i+1)
        plt.title(vars)
        plt.hist(cat1[vars], bins=bins[vars], density=True, label="COSMOS")
        plt.hist(cat2[vars], bins=bins[vars], density=True, label="TU", alpha=.7)
        plt.xlim(limits[vars])
        if i==0:
            plt.legend()
    plt.show()

#------------------------------------------------------------

# checkbox for variables selection
def check_vars(cat_cs, cat_tu):
    """
    Return catalogs after variables selection.

    Params:
    - cat_cs : dict, COSMOS catalog
    - cat_tu : dict, TU catalog

    Return:
    - selected_data : list, array of variables from the selection
    - selected_vars : list, variable names from the selection
    """
    from ipywidgets import (Checkbox, VBox, interactive_output)

    names = []
    checkbox_objects = []
    for key in cat_cs:
        checkbox_objects.append(Checkbox(value=False, description=key))
        names.append(key)

    arg_dict = {names[i]: checkbox for i, checkbox in enumerate(checkbox_objects)}

    ui = VBox(children=checkbox_objects)

    select_cs = []
    select_tu = []
    selected_vars = []
    def select_data(**kwargs):
        select_cs.clear()
        select_tu.clear()
        selected_vars.clear()

        for key in kwargs:
            if kwargs[key] is True:
                select_cs.append(cat_cs[key])
                select_tu.append(cat_tu[key])
                selected_vars.append(key)
        try:
            print("Selected variables :",selected_vars)
        except:
            pass
        return  select_cs, select_tu, selected_vars

    out = interactive_output(select_data, arg_dict)
    display(ui, out)

    return select_cs, select_tu, selected_vars

#------------------------------------------------------------

# TU catalog in two parts, cut at max(COSMOS["mag"])
def mag_sup_inf(som, cat_cs, cat_tu):
    """
    TU catalog in two parts, cut at max(COSMOS["mag"])
    """
    mag_max = np.max(cat_cs["mag"])

    mag_tu_sup_cs = cat_tu['mag'][np.where(cat_tu['mag']>mag_max)[0]]
    hlr_mag_tu_sup_cs = cat_tu['hlr'][np.where(cat_tu['mag']>mag_max)[0]]
    cut_mag_sup = np.vstack([mag_tu_sup_cs, hlr_mag_tu_sup_cs]).T

    mag_tu_inf_cs = cat_tu['mag'][np.where(cat_tu['mag']<mag_max)[0]]
    hlr_mag_tu_inf_cs = cat_tu['hlr'][np.where(cat_tu['mag']<mag_max)[0]]
    cut_mag_inf = np.vstack([mag_tu_inf_cs, hlr_mag_tu_inf_cs]).T

    return cut_mag_sup, activ_inf

#------------------------------------------------------------

# check gx properties near mag 25.2

def get_loc(som, data, voisins_nb=8, activ_2_val=0):
    """
    Return a random localisation of a galaxy near a depopulated area

    Params:
    - som: MiniSom, trained SOM
    - data : array, training dataset

    Optinal params:
    - voisins_nb : int, -4- direct neighbors or -8- neighbors
    - activ_2_val : int, number of hit of the neighbor (default = 0 to search in depopulated areas)
    """
    activ_resp = som.activation_response(data)
    som_x, som_y = activ_resp.shape

    activ_1_loc = np.argwhere(activ_resp==1)
    activ_2_loc = np.argwhere(activ_resp==activ_2_val)

    if voisins_nb==4:
        voisins=[[+1,0],[-1,0],
                  [0,-1],[0,+1]]
    elif voisins_nb==8:
        voisins=[[+1,-1],[+1,0],[+1,+1],
                [0,-1],[0,+1],
                [-1,-1],[-1,0],[-1,+1]]
    else:
        raise ValueError("voisins should be 4 or 8 not "+str(voisins_nb))

    tab=[]
    for act1 in activ_1_loc[:len(activ_2_loc)]:
        for act2 in activ_2_loc:
            for vois in voisins:

                if (act1[0]+vois[0]==act2[0] and act1[1]+vois[1]==act2[1]
                    and act1[0]!=0 and act1[0]!=som_x and act1[1]!=0 and act1[1]!=som_x):

                    tab.append(act1)

    assert len(tab)>0, "can't find a corresponding element"

    rng = np.random.default_rng()
    return rng.choice(tab)

def get_idx(som, cat, loc):
    """
    Return the galaxy index of the get_loc() result

    Params:
    - som: MiniSom, trained SOM
    - data : array, training dataset
    - loc : tuple, get_loc() result -> galaxy position
    """
    get_idx=[]
    for i,j in enumerate(cat):
        idx = np.argwhere(som.activation_response([j]))[0]
        if sum(idx==loc)==2:
            get_idx.append(i)
    return get_idx

def act_show(som, point, data):
    """
    Show the activation map and the position of the galaxy find with get_loc()

    Params
    - som: MiniSom, trained SOM
    - point : tuple, get_loc() result -> galaxy position
    - data : array, training dataset
    """
    from matplotlib.colors import LogNorm

    activ_resp = som.activation_response(data)

    plt.figure(figsize=(9,9))
    plt.imshow(activ_resp, norm=LogNorm())
    plt.scatter(x=point[1], y=point[0], marker='x', s=500, linewidth=7, c='r')
    plt.xticks([]) ; plt.yticks([])
    plt.show()

def check_hist_pos(dat, cat):
    """
    Position of the galaxy in the histograms of a catalog

    Params:
    - dat : list, galaxy parameters
    - cat : dict, galaxy catalog
    """
    cat["mag"]*=35
    dat[0]*=35

    cat["sersic"]*=6
    dat[2]*=6

    plt.figure(figsize=(18,4))
    for i, vars in enumerate(cat):
        plt.subplot(1,4,i+1)
        plt.title(vars + ' = %.2f' % dat[i])
        plt.axvline(dat[i], color='k')
        if vars=='hlr':
            bins=500
        else:
            bins=200
        plt.hist(cat[vars], bins=bins, density=True)
        if vars=="hlr":
            plt.xlim(-.1,1.1)
    plt.tight_layout()
    plt.show()

    cat["mag"]/=35
    dat[0]/=35
    cat["sersic"]/=6
    dat[2]/=6

#point = get_loc(som_tu, choice_cs, voisins_nb=8, activ_2_val=0)
#act_show(som_tu, point, choice_cs)

#idx = get_idx(som_tu, choice_cs, point)[0]
#check_hist_pos(choice_cs[idx], data_cs)

#------------------------------------------------------------
