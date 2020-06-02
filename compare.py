import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from minisom import MiniSom


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



def plot_compare(param_tu, param_cs, bins, *args):
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(8,6))

    ax[0].set_title('True Universe catalog')
    ax[0].hist(data_TU[param_tu], bins=bins) # histogram
    ax[0].axvline(np.mean(data_TU[param_tu]),
        color='k', linestyle='--') # vertical line for the mean value
    if len(args[0])==2:
        ax[0].set_xlim(args[0]) # x limits
    ax[0].set_xlabel(param_tu)

    ax[1].set_title('COSMOS catalog')
    if len(args)>2:# if a column has to be selected
        ax[1].hist(data_CS[param_cs][:,args[-1]], bins=bins) # histogram
        ax[1].axvline(np.mean(data_CS[param_cs][:,args[-1]]),
            color='k', linestyle='--', label='moyenne') # vertical line for the mean value
    else:
        ax[1].hist(data_CS[param_cs], bins=bins) # histogram
        ax[1].axvline(np.mean(data_CS[param_cs]),
            color='k', linestyle='--', label='moyenne') # vertical line for the mean value
    if len(args[0])==2:
        ax[1].set_xlim(args[1]) # x limits
    ax[1].set_xlabel(param_cs)

    plt.tight_layout()
    plt.legend()
    plt.show()

#plot_compare('mag', 'mag_auto', 100, (15,35), (15,35))

#plot_compare('half_light_radius', 'hlr', 1000, (-2,2),(-500,2000),1)

#plot_compare('q', 'sersicfit', 100, (-1,2),(-1,2),3)

plot_compare('SSersic_n', 'sersicfit', 100, (-2,8),(-2,8),2)
