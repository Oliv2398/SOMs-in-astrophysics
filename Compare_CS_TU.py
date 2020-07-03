# Roth Olivier 07/2020
# Catalog variable comparisons (with histograms)

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


path_TU = "datas/TU_created.fits"
with fits.open(path_TU) as hdul:
    hdr_TU = hdul[0].header # header
    data_TU = hdul[1].data # data
    cols_TU = hdul[1].columns # cols information
    Names_TU = cols_TU.names # cols names
    hdul.info()


path_COSMOS = "datas/real_galaxy_catalog_25.2_fits.fits"
with fits.open(path_COSMOS) as hdul:
    hdr_CS = hdul[0].header # header
    data_CS = hdul[1].data # data
    cols_CS = hdul[1].columns # cols information
    Names_CS = cols_CS.names # cols names
    print('\n')
    hdul.info()


def cut_hight_hlr(cat, lim): # delete problems
    idx = np.where(cat['sersicfit'][:,1]>lim)[0]
    print(idx.shape)
    cat = np.delete(cat,idx)
    return cat

data_CS = cut_hight_hlr(data_CS, 40) # cut hight values

data_CS['sersicfit'][:,1] *= 0.03 # converting hlr



def plot_compare(param_tu, param_cs, bins_tu, bins_cs, limits, *arg):

    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8,6))

    #~~~~~~~~~~~~~~
    ax[0].set_title('True Universe catalog')
    ax[0].hist(data_TU[param_tu], bins=bins_tu, density=1)
    ax[0].axvline(np.mean(data_TU[param_tu]),
        color='k', linestyle='--') # vertical line for the mean value

    ax[0].set_xlim(limits[0])
    ax[0].set_xlabel(param_tu)

    #~~~~~~~~~~~~~~
    ax[1].set_title('COSMOS catalog')
    if arg:# if a column has to be selected
        ax[1].hist(data_CS[param_cs][:,arg], bins=bins_cs, density=True)
        ax[1].axvline(np.mean(data_CS[param_cs][:,arg]),
            color='k', linestyle='--', label='moyenne') # vertical line for the mean value
    else:
        ax[1].hist(data_CS[param_cs], bins=bins_cs, density=True)
        ax[1].axvline(np.mean(data_CS[param_cs]),
            color='k', linestyle='--', label='moyenne') # vertical line for the mean value


    ax[1].set_xlim(limits[1]) # x limits
    ax[1].set_xlabel(param_cs)

    #~~~~~~~~~~~~~~
    plt.tight_layout()
    plt.legend()
    plt.show()


#plot_compare('mag', 'mag_auto', 300, 50, ((15,35), (15,35)))

#plot_compare('half_light_radius', 'sersicfit', 5000, 1000, ((-1,2),(-1,2)),1)

#plot_compare('q', 'sersicfit', 200, 100, ((0,1.2),(0,1.2)),3)

plot_compare('SSersic_n', 'sersicfit', 200, 200, ((-.2,6.5),(-.2,6.5)),2)
