import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from time import time

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


# Make subplots

def subplots_QxS(nb=5, show_vars=False, start=0,
                flux_min=80, noise_min=0, mag_min=0):
    # nb : size of the subplot
    # show_vars : show q and sersic in the subplots
    # start : file number to start the research

    q_nb = np.linspace(0.1, 0.9, nb+1) # intervals
    q_get = np.zeros((nb,nb))

    s_nb = np.linspace(0, 6, nb+1) # intervals
    s_get = np.zeros((nb,nb))

    idx_get = np.zeros((nb,nb))
    file_get = np.zeros((nb,nb))

    if start!=0:
        start = 1000 * (start-1)
    else:
        start = 0

    for i in range(nb): # for each q
        for j in range(nb): # for each S
            for k in range(start, data['GAL_HDU'].shape[0]):

                Q = data_fits['sersicfit'][k][3]
                S = data_fits['sersicfit'][k][2]

                if (Q > q_nb[i] and Q < q_nb[i+1] # if q is in the interval
                    and (S > s_nb[j] and S < s_nb[j+1]) # if S is in the interval
                    and data['stamp_flux'][k] > flux_min
                    and data['NOISE_MEAN'][k] > noise_min # 3e-4
                    and data['mag'][k] > mag_min # 18
                    ):

                    q_get[i,j] = Q
                    s_get[i,j] = S

                    idx_get[i,j] = data['GAL_HDU'][k] # get the index

                    # get the file name
                    if data['GAL_FILENAME'][k][26]!='.': # if 2 digits
                        file_get[i,j] = data['GAL_FILENAME'][k][25:26+1]
                    else: # if 1 digit
                        file_get[i,j] = data['GAL_FILENAME'][k][25]

                    #print(data['GAL_FILENAME'][k])
                    print('[ {} / {} ]'.format(i*nb+j+1, nb**2), end='\r')
                    break

    assert (file_get!=0).all(), "can't find an element, change init choices"

    print("\n\n1 - e :\n", q_get)
    print("\nsersic :\n", s_get)
    print("\n-------------------------")
    print("\nindex :\n", idx_get)
    print("\nfile :\n", file_get)

    files = np.unique(file_get)

    print('\nfiles : real_galaxy_images_25.2_n( ).fits  ', files, '\n')


    path = 'datas_full/COSMOS_25.2_training_sample/'

    # collecting galaxies from file
    images_file = []
    hdul_image = []
    for i in range(len(files)):
        images_file.append((path+'real_galaxy_images_25.2_n%d.fits')%(files[i]))
        hdul_image.append(fits.open(images_file[-1]))

    images = np.zeros((nb,nb)).astype(object)
    for i in range(nb):
        for j in range(nb):
            num_file = np.argwhere(file_get[i,j]==files)[0,0]
            images[i,j]=hdul_image[ num_file ][ int(idx_get[i,j]) ].data


    # plotting
    fig, ax = plt.subplots(nrows=nb, ncols=nb, figsize=(7.5,7.5))
    for i in range(nb):
        for j in range(nb):

            # subplot
            ax[i,j].imshow(images[i,j], cmap='gray')

            # to see q and S in the titles
            if show_vars:
                ax[i,j].set_title('q={:.2f} s={:.2f}'.format(q_get[i,j],s_get[i,j]))

            # hide axis
            ax[i,j].set_axis_off()

    if not show_vars and nb==5: # arrows
        arrowproperties = dict(arrowstyle="simple", facecolor="black")

        ax[-1,0].annotate('Q', xy=(-0.4, 0.3),
                        xycoords='axes fraction',
                        xytext=(-0.5, 5),
                        bbox=dict(boxstyle="round", fc="0.8"),
                        arrowprops=arrowproperties)

        ax[0,0].annotate('Sersic', xy=(5, 1.42),
                        xycoords='axes fraction',
                        xytext=(0.5, 1.4),
                        bbox=dict(boxstyle="round", fc="0.8"),
                        arrowprops=arrowproperties)


    plt.tight_layout()
    plt.show()


subplots_QxS() # start = 18 
