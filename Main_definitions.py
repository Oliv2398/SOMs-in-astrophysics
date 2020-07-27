import numpy as np
import matplotlib.pyplot as plt

from minisom import MiniSom

from astropy.io import fits

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
                dict_vars["mapmean"].append(np.mean(som.distance_map()))
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
                    ax[1].set_title('distance map ; mean = %.2f' % np.mean(som.distance_map()))
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
        Manual training to see the evolution all along the training.

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
            topological error if -topological_error- is activated,
            mean of the distance map (U-matrix),
        """
        rows, cols = data.shape
        som_x = int(np.sqrt(5*np.sqrt(rows)))
        som_y = som_x

        som = MiniSom(som_x, som_y, cols, sigma, learning_rate)
        som.random_weights_init(data)

        dict_vars ={"iter_x":[],
                    "q_error":[],
                    "mapmean":[]}

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
                dict_vars["mapmean"].append(np.mean(som.distance_map()))

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
                    ax[1].set_title('distance map ; mean = %.2f' % dict_vars["mapmean"][-1])
                    ax[1].axis('off')

                    plt.suptitle('SOM %d x %d ; iteration [ %d / %d ] - %d %%'%(som_x, som_y, i+1, iterations, 100*(i+1)/iterations))
                    plt.show()
                    clear_output(wait=True)

        return dict_vars



def plot_error(dict_vars):
        """
        Some plots from the manual training
        """
        plt.figure(figsize=(10,4))
        plt.plot(dict_vars["iter_x"], dict_vars["q_error"])
        plt.ylabel('quantization error')
        plt.xlabel('iteration')

        if "t_error" in dict_vars:
            plt.figure(figsize=(10,4))
            plt.plot(dict_vars["iter_x"], dict_vars["t_error"])
            plt.ylabel('topological error')

        if "sigma" in dict_vars:
            fig, ax = plt.subplots(1,2,figsize=(10,4))
            ax[0].plot(dict_vars["iter_x"], dict_vars["sigma"])
            ax[0].set_ylabel("sigma")
            ax[1].plot(dict_vars["iter_x"], dict_vars["learning_rate"])
            ax[1].set_ylabel('learning rate')

        plt.figure(figsize=(10,4))
        plt.plot(dict_vars["iter_x"], dict_vars["mapmean"])
        plt.ylabel('mean of the distance map')

        plt.show()


#------------------------------------------------------------

# subplots
def PlotSOMs(som, var_names=(["R","G","B"]), topology='rectangular', rescale_weigths=False):
    """
    Show the SOM, the distance map and the variables weights

    Params:
    - som, MiniSom, trained SOM

    Optional params:
    - var_names : list, variable names
    - topology : str, -rectangular- or -hexagonal-
    - rescale_weigths : bool, in case the SOM is uninterpretable
    """
    weights = som.get_weights().copy()
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
        ax1 = ax[0,1].imshow(som.distance_map())

        # variables plots
        for i in range(cols):
            ax[1,i].imshow(weights[:,:,i])

        if rescale_weigths and cols==4 and topology!='hexagonal':
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
            pixel_color1.append(plt.cm.viridis(j))

        # create patches
        patch_list0 = [] ; patch_list1 = []
        patch_listi = [[] for i in range(cols)]
        for c0, c1, x, y in zip(pixel_color0, pixel_color1, xx.flat, wy.flat):
            patch_list0.append(
                    RegularPolygon(xy = (x, y),
                                numVertices = 6,
                                radius = .95/np.sqrt(3)+.03,
                                facecolor = to_hex(c0)))
            patch_list1.append(
                    RegularPolygon(xy = (x, y),
                                numVertices = 6,
                                radius = .95/np.sqrt(3)+.03,
                                facecolor = c1))

            for idx, k in enumerate(c0):
                fcolor = (k-np.min(weights[:,:,idx]))/(np.max(weights[:,:,idx])-np.min(weights[:,:,idx]))
                patch_listi[idx].append(
                    RegularPolygon(xy = (x, y),
                                numVertices = 6,
                                radius = .95/np.sqrt(3)+.03,
                                facecolor = plt.cm.viridis(fcolor)))

        # SOM
        pc0 = PatchCollection(patch_list0, match_original=True)
        ax[0,0].add_collection(pc0)

        # distance map
        pc1 = PatchCollection(patch_list1, match_original=True)
        ax[0,1].add_collection(pc1)

        # each layer of the weights
        for i in range(cols):
            pci = PatchCollection(patch_listi[i], match_original=True)
            ax[1,i].add_collection(pci)


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
    ax[0,1].set_title('distance map ; mean = %.2f' % np.mean(som.distance_map()))
    for i in range(cols):
        try:
            ax[1,i].set_title(var_names[i])
        except:
            pass

    plt.tight_layout()
    plt.show()

#------------------------------------------------------------

# heatmap
def Heatmap(som, data, topology="rectangular", normed=True, hit_count=True, hist_vars=False, figsize='default'):
    """
    Show the activation response of the SOM to a certain dataset

    Params:
    - som : MiniSom, trained SOM
    - data : array, activation dataset

    Optional params:
    - topology : str, -rectangular- or -hexagonal-
    - normed : bool, imshow with LogNorm
    - hit_count : bool, number of hit in each cell
    - hist_vars : bool, histogram of the dataset's variables
    - figsize : tuple, size of the figure
    """

    if hist_vars:
        fig, ax = plt.subplots(1,data.shape[1], figsize=(17,6))
        for i, cols in enumerate(data.T):
            ax[i].hist(cols, bins=100, color="black")
            ax[i].set_title("variable " + str(i+1))
        plt.show()


    # activation response
    activ_resp = som.activation_response(data)

    # size of the figure
    if figsize=='default':
        fig, ax = plt.subplots(1,figsize=(14,14))
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

        ax.imshow(activ_resp, norm = norm) ###


        if hit_count:
            som_x, som_y = som.get_weights().shape[:2]

            for i in range(som_x):
                for j in range(som_y):
                    if activ_resp[i,j]!=0: # don't show inactivated cells
                        # i,j inverted in plt.text because of the minisom's coordinates problem
                        ax.text(j, i, int(activ_resp[i,j]),
                                horizontalalignment='center',
                                verticalalignment='center')

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
        ax.add_collection(pc)

        for i in range(som_x):
            for j in range(som_y):
              if activ_resp[i,j]!=0 and hit_count:
                    ax.text(xx[i, j], wy[i,j], int(activ_resp[i,j]),
                                    horizontalalignment='center',
                                    verticalalignment='center')

        ax.axis([-1, som_x, -.7, som_y*np.sqrt(3)/2])
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

#------------------------------------------------------------

# interactive SOM
def _interactive_som(data, names, sigma, learning_rate, iterations, topology, size, info):
    """The function to which the interactive widgets are tied"""
    som, wts = train_som(data,
                         sigma = sigma,
                         learning_rate = learning_rate,
                         iterations = iterations,
                         topology = topology,
                         size=size)

    PlotSOMs(som, names, topology=topology, rescale_weigths=False)

    if info:
        print('quantization error :', som.quantization_error(data))
        if topology=='rectangular':
            print('topographic error :', som.topographic_error(data))

def interactive_plot(data, size='default', names=(["R","G","B"]), infos=False):
    """
    Interactive SOM, the sigma, learning rate and iterations can be changed by sliders

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

# random colors
def dat_color(nb=25000, more_dim=0):
    """
    Create a random dataset of colors

    Optional params:
    - nb : int, number of rows (= number of colors)
    - more_dim : int, number of cols

    Return:
    - nb : int, number of rows
    - data : array, dataset of colors
    - names : list, ['Red', 'Green', 'Blue']
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
    return nb, data, names

# random normalized colors
def dat_color_norm(nb=25000):
    """
    Create a random uniform dataset of colors

    Optional params:
    - nb : int, number of rows (= number of colors)

    Return:
    - data : array, dataset of colors
    """
    data=[]
    for i in range(nb):
        theta = np.random.uniform(np.pi)
        phi = np.random.uniform(np.pi*2)

        x = np.sin(theta)*np.cos(phi)
        y = np.sin(theta)*np.sin(phi)
        z = np.cos(theta)

        data.append([x,y,z])

    return np.abs(data)

#------------------------------------------------------------
#---------------------- Catalog defs ------------------------
#------------------------------------------------------------

# loading fits file
def load_cat(path):
    """
    Load galaxy catalogs from fits file

    Params:
    - path : str, path and name of the file

    Return:
    - cat : astropy.io.fits, catalog
    """
    with fits.open(path) as hdul:
        cat = hdul[1].data
    return cat

# turning TU catalog into dict
def extract_tu(data):
    """
    TU fits catalog into dict

    Params:
    - data : astropy.io.fits, catalog

    Return:
    - datas : dict, catalog with "mag", "hlr", "sersic", "q"
    """

    cat_tu = data.copy()
    datas = {"mag" : cat_tu['mag'],
            "hlr" : cat_tu['half_light_radius'],
            "sersic" : cat_tu['SSersic_n'],
            "q" : cat_tu['q']}
    return datas

# merge dict and keep values of common keys in list
def mergeDict(dict1, dict2):
    """
    Merge two dict and keep values of common keys in list

    Params:
    - dict1 : dict, first dict to merge
    - dict2 : dict, second dict to merge

    Return:
    - dict3 : dict, dict merge
    """

    dict3 = {**dict1, **dict2}
    for key, value in dict3.items():
        if key in dict1 and key in dict2:
            dict3[key] = np.concatenate([ value , dict1[key] ])
    return dict3

#------------------------------------------------------------

# histograms, cuts and normalizations on a catalog
def cut_normalize_view(catalog, init_view=True, norm_and_view=(True,True), density=False):
    """
    Histograms, cuts and normalizations on a catalog

    Params:
    -catalog : dict, "TU", "TU_fuse", "COSMOS" and the corresponding catalog

    Optional params:
    - init_view : bool, histograms of the catalog before normalizations
    - norm_and_view : tuple, bool : norm
                             bool : histograms  of the catalog after normalizations
    - density : bool, plt.hist() density function

    Return:
    - Datas : dict, catalog normalized or not
    """
    if "TU" in catalog.keys() or "TU_fuse" in catalog.keys():
        Datas = list(catalog.values())[0].copy()
        print(list(catalog.keys())[0]+" catalog loaded \n")

    elif "COSMOS" in catalog.keys():
        cat_cs = list(catalog.values())[0].copy()
        Datas = {"mag" : cat_cs['mag_auto'],
                "hlr" : cat_cs['sersicfit'][:,1],
                "sersic" : cat_cs['sersicfit'][:,2],
                "q" : cat_cs['sersicfit'][:,3]}
        print("\nCOSMOS catalog loaded \n")
    else:
        raise ValueError("choose between -TU-, -TU_fuse- and -COSMOS- ; not " + list(catalog.keys())[0])

    if init_view:
        plt.figure(figsize=(16,4))
        for i, vars in enumerate(Datas):
            plt.subplot(1,4,i+1)
            plt.title(vars +' ; min=%.2f, max=%.2f\n' % (min(Datas[vars]), max(Datas[vars])))
            plt.hist(Datas[vars], bins=100, density=density)
        plt.tight_layout()
        plt.show()


    if norm_and_view[0]: # modifications in the catalog

        # convert hlr
        if "COSMOS" in catalog.keys():
            Datas["hlr"] *= 0.03*np.sqrt(Datas["q"])

        # delete hlr problems
        idx = np.where(Datas['hlr']>10)[0]
        print("hlr : nb d'elements suppr", idx.shape[0])
        for i in Datas:
            Datas[i] = np.delete(Datas[i], idx)

        # delete sersic problems
        if "COSMOS" in catalog.keys():
            idx_sup = np.where(Datas['sersic']>max(Datas['sersic'])-.001)[0]
            idx_inf = np.where(Datas['sersic']<min(Datas['sersic'])+.001)[0]
            print("sersic : nb d'elements suppr", idx_sup.shape[0]+idx_inf.shape[0])
            for i in Datas:
                Datas[i] = np.delete(Datas[i],np.hstack([idx_sup,idx_inf]))

        # normalize mag
        Datas['mag'] /= 35

        # rescale hlr sersic
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()

        for i in Datas.keys():
            if i!="q" and i!="mag":
                X = Datas[i].reshape(-1,1).copy()
                scaler.fit(X)
                Datas[i] = scaler.transform(X)
                Datas[i] = Datas[i].flatten()

        # rescale hlr
        Datas["hlr"] = (Datas["hlr"]-0)/(.3-0)

        # final view
        if norm_and_view[1]:
            plt.figure(figsize=(16,4))
            for i, vars in enumerate(Datas):
                plt.subplot(1,4,i+1)
                plt.title(vars +' ; min=%.2f, max=%.2f\n' % (min(Datas[vars]), max(Datas[vars])))
                plt.hist(Datas[vars], bins=200, density=density)
                plt.xlim(-.1,1.1)
            plt.tight_layout()
            plt.show()

    print("--------------------------\n\n")
    return Datas

# compare COSMOS and TU histograms
def compare_CS_TU(cat1, cat2, norm=True):
    """
    Compare COSMOS and TU histograms

    Params:
    - cat1 : dict, first catalog
    - cat2 : dict, second catalog

    Optionnal params:
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
    plt.legend()
    plt.show()

#------------------------------------------------------------

# checkbox for variables selection
def check_vars(data):
    """
    Return variables of a catalog after selection

    Params:
    - data : dict, catalog of galaxies

    Return:
    - selected_data : list, array of variables from the selection
    - selected_vars : list, variable names from the selection
    """
    from ipywidgets import (Checkbox, VBox, interactive_output)

    names = []
    checkbox_objects = []
    for key in data:
        checkbox_objects.append(Checkbox(value=False, description=key))
        names.append(key)

    arg_dict = {names[i]: checkbox for i, checkbox in enumerate(checkbox_objects)}

    ui = VBox(children=checkbox_objects)

    selected_data = []
    selected_vars = []
    def select_data(**kwargs):
        selected_data.clear()
        selected_vars.clear()

        for key in kwargs:
            if kwargs[key] is True:
                selected_data.append(data[key])
                selected_vars.append(key)
        try:
            print("\nNumber of rows :",len(selected_data[0]))
            print("Variables :",selected_vars)
        except:
            pass
        return  selected_data, selected_vars

    out = interactive_output(select_data, arg_dict)
    display(ui, out)

    return selected_data, selected_vars

#------------------------------------------------------------
