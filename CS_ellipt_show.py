import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

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


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# to see if sersicfit[:,3] == e or 1-e (linked with images)

def get_q(num_gx):
    idx = np.argwhere(
        data['GAL_FILENAME']=='real_galaxy_images_25.2_n%d.fits'%num_gx)[0][0]

    q = data_fits['sersicfit'][idx][3]

    print("\nGalaxy n%d:" % num_gx)
    print(' 1 - e =', q)

#get_q(8)
#get_q(11)

# allongée : 8, 24, 26
# ronde : 11
# sersicfit[3] 8:0.27, 11:0.89


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_some_gx(nb, num_fichier, *arg):
    # nb : number of images to plot
    # num_fichier : file number (between 1 and 88)
    # arg : to plot half or less of nb images, allows smaller intervals

    q_nb = np.linspace(0.1,0.9,nb+1) # intervals
    q_get = np.zeros(nb)
    idx_get = np.zeros(nb)

    for i in range(nb): # for each interval
        for j in range(int(1e3*(num_fichier-1)),int(1e3*num_fichier)):

            x = data_fits['sersicfit'][j][3]

            if (x > q_nb[i] and x < q_nb[i+1] # if q is in the interval
                and data['stamp_flux'][j]>80
                #and data['NOISE_MEAN'][j]>1e-4
                #and data['mag'][j]>18
                ):

                q_get[i] = x
                idx_get[i] = j - (1e3*(num_fichier-1)) # get the index
                break

    print('\nFile : real_galaxy_images_25.2_n%d.fits\n' % num_fichier)
    print("-----------------\n")
    print("1 - e :\n", q_get)
    print("\nindex :\n", idx_get)

    path = 'datas_full/COSMOS_25.2_training_sample/'
    image_file = (path+'real_galaxy_images_25.2_n%d.fits')%(num_fichier)

    hdul_image = fits.open(image_file)

    images = []
    for i in range(len(hdul_image)): # collecting galaxies from file
        images.append(hdul_image[i].data)

    if arg: # selecting half of the datas
        nb = int(nb/arg[0])
        idx_get = idx_get[::arg[0]]
        q_get = q_get[::arg[0]]
        print("\n-----------------\n")
        print("Selection :")
        print('1 - e :\n', q_get)
        print('\nindex :\n', idx_get)

    fig, ax = plt.subplots(nrows=1, ncols=nb)
    for i in range(nb): # plotting
        ax[i].imshow(images[int(idx_get[i])], cmap='gray')
        ax[i].set_title('q={:.2f}'.format(q_get[i]))
        ax[i].set_axis_off()
    plt.tight_layout()
    plt.show()


#plot_some_gx(12, 49, 2) # stamp_flux > 80
plot_some_gx(12, 49, 3) # stamp_flux > 80

#plot_some_gx(6, 80) # stamp_flux > 80

#plot_some_gx(12, 81, 3) # stamp_flux > 40
