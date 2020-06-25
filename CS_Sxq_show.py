import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# Loading datas

path_CS = "datas/real_galaxy_catalog_25.2.fits"
path_CS_fits = "datas/real_galaxy_catalog_25.2_fits.fits"

with fits.open(path_CS) as hdul:
    hdr = hdul[0].header # header
    data = hdul[1].data # data
    cols = hdul[1].columns # cols information
    Names = cols.names # cols names

with fits.open(path_CS_fits) as hdul_fits:
    hdr_fits = hdul_fits[0].header # header
    data_fits = hdul_fits[1].data # data
    cols_fits = hdul_fits[1].columns # cols information
    Names_fits = cols_fits.names # cols names

#----------------------------------------------------

# hlr cut and normalization

data_fits['sersicfit'][:,1] *= 0.03 # converting hlr

"""
# cut hight hlr values on the 2 catalogs
idx = np.where(data_fits['sersicfit'][:,1]>2)[0]
#print("nb d'elements suppr", idx.shape)
data_fits = np.delete(data_fits,idx)
data = np.delete(data,idx)

#data_fits['sersicfit'][:,1] /= max(data_fits['sersicfit'][:,1]) # normalize
"""
nb_data = data_fits.shape[0]

#----------------------------------------------------

# Make subplots

def subplots_QxS(nb=5, vars_names=["q","sersic"],
                show_axis=True, start=0, flux_min=80):
    # nb : size of the subplot
    # vars_names : variables to plot
    # start : file number to start the research

    print("Mapping a {}x{} figure\n ".format(nb,nb))

    choice = {"sersic" : data_fits['sersicfit'][:,2],
            "hlr" : data_fits['sersicfit'][:,1],
            "q" : data_fits['sersicfit'][:,3],
            "i" : data_fits['sersicfit'][:,0],
            "mag" : data_fits['mag_auto']
            }

    intervals = {"sersic" : np.linspace(0, 6, nb+1),
            "hlr" : np.linspace(0, 2, nb+1),
            "q" : np.linspace(.1, .9, nb+1),
            "i" : np.linspace(0, .15, nb+1),
            "mag" : np.linspace(18, 25.2, nb+1)
            }

    vars0 = choice[vars_names[0]]
    vars1 = choice[vars_names[1]]

    int0 = intervals[vars_names[0]]
    int1 = intervals[vars_names[1]]

    get0 = np.zeros((nb,nb))
    get1 = np.zeros((nb,nb))

    idx_get = np.zeros((nb,nb))
    file_get = np.zeros((nb,nb))

    if start!=0:
        start = 1000 * (start-1)
    else:
        start = 0

    print("Searching corresponding galaxies...")
    for i in range(nb):
        for j in range(nb):
            for k in range(start, nb_data):

                elmt_vars0 = vars0[k]
                elmt_vars1 = vars1[k]

                if (elmt_vars0 > int0[i] and elmt_vars0 < int0[i+1]
                    and (elmt_vars1 > int1[j] and elmt_vars1 < int1[j+1])
                    and data['stamp_flux'][k] > flux_min
                    ):

                    get0[i,j] = elmt_vars0
                    get1[i,j] = elmt_vars1

                    idx_get[i,j] = data['GAL_HDU'][k] # get the index

                    # get the file name
                    if data['GAL_FILENAME'][k][26]!='.': # if 2 digits
                        file_get[i,j] = data['GAL_FILENAME'][k][25:26+1]
                    else: # if 1 digit
                        file_get[i,j] = data['GAL_FILENAME'][k][25]

                    print('[ {} / {} ]'.format(i*nb+j+1, nb**2), end='\r')
                    break

    print("\n\n",vars_names[0]," :\n", get0)
    print("\n",vars_names[1]," :\n", get1)
    print("\n-------------------------")
    print("\nindex :\n", idx_get)
    print("\nfile :\n", file_get)

    files = np.unique(file_get)
    print('\nfiles : real_galaxy_images_25.2_n( ).fits  ', files, '\n')


    path = 'datas_full/COSMOS_25.2_training_sample/'
    filename = 'real_galaxy_images_25.2_n'

    # collecting galaxies from file
    images_file = []
    hdul_image = []
    for i in range(len(files)):
        if files[i]!=0:
            images_file.append((path+filename+'%d.fits')%(files[i]))
        else:
            images_file.append(0)

        try:
            hdul_image.append(fits.open(images_file[-1]))
        except:
            hdul_image.append(0)


    images = np.zeros((nb,nb)).astype(object)
    for i in range(nb):
        for j in range(nb):
            if file_get[i,j]!=0:
                num_file = np.argwhere(file_get[i,j]==files)[0,0]
            else:
                num_file = np.argwhere(file_get[i,j]==files)[0,0] - 1

            try:
                images[i,j]=hdul_image[ num_file ][ int(idx_get[i,j]) ].data
            except:
                pass


    # plotting
    fig, ax = plt.subplots(nrows=nb, ncols=nb, figsize=(7.5,7.5))

    # cadre
    if show_axis:
        ax1 = fig.add_axes([0.123, 0.11, 0.78, 0.77], frameon=False)
        ax1.patch.set_alpha(0.)
        ax1.set_xticks(int1) ; ax1.set_xlabel(vars_names[1])
        ax1.set_yticks(int0) ; ax1.set_ylabel(vars_names[0],rotation=0)
        ax1.set_xlim(int1[0],int1[-1])
        ax1.set_ylim(int0[-1],int0[0])

    for i in range(nb): # galaxies
        for j in range(nb):
            # hide axis
            ax[j,i].set_axis_off()

            # subplot
            if file_get[j,i]!=0:
                ax[j,i].imshow(images[j,i], cmap='gray')
            else:
                black=np.zeros((2,2,3))
                ax[j,i].imshow(black)

    plt.subplots_adjust(wspace=-0.05, hspace=-0.05) # side by side
    plt.show()


subplots_QxS(nb=9, vars_names=["q","sersic"]) # vars_names y, x
# sersic, hlr, q, i, mag

def show_image_from_file(idx, num_file, infos=True):
    path = 'datas_full/COSMOS_25.2_training_sample/'
    file = (path+'real_galaxy_images_25.2_n%d.fits')%(num_file)
    hdul_image = fits.open(file)

    if infos:
        print("magnitude :",data_fits['mag_auto'][idx+(num_file-1)*1000])
        print("intensity at the hlr :",data_fits['sersicfit'][idx+(num_file-1)*1000][0])
        print("hlr :",data_fits['sersicfit'][idx+(num_file-1)*1000][1])
        print("Sersic :",data_fits['sersicfit'][idx+(num_file-1)*1000][2])
        print("q :",data_fits['sersicfit'][idx+(num_file-1)*1000][3])

    plt.imshow(hdul_image[idx].data)
    plt.show()
