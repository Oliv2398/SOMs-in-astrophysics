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

data_fits['sersicfit'][:,1] *= 0.03 # conversion

#------------------------------------

def get_index(vars, lim, out, num):
    if out=='sup':
        idx = np.where(data_fits['sersicfit'][:,vars] > lim)[0][num]
    elif out=='inf':
        idx = np.where(data_fits['sersicfit'][:,vars] < lim)[0][num]
    else:
        raise ValueError("-out- should be 'sup' or 'inf'")
    print("index : %d ; vars : %d" % (idx, vars))
    print("hlr %.2f ; sersic %.2f ; q %.2f \n" %(
        data_fits['sersicfit'][:,1][idx],
        data_fits['sersicfit'][:,2][idx],
        data_fits['sersicfit'][:,3][idx]))
    return idx

def get_file_hdu(idx):
    if data['GAL_FILENAME'][idx][26]=='.':
        file = int(data['GAL_FILENAME'][idx][25])
    else:
        file = int(data['GAL_FILENAME'][idx][25:27])

    hdu = data['GAL_HDU'][idx]
    return file, hdu

def show_outlier(vars, lim, out, num, Show=True):

    idx = get_index(vars, lim, out, num)

    file, hdu = get_file_hdu(idx)

    if Show:
        path = 'datas_full/COSMOS_25.2_training_sample/'
        images_file = (path+'real_galaxy_images_25.2_n%d.fits')%(file)

        hdul_image = fits.open(images_file)
        image = hdul_image[hdu].data

        plt.imshow(image, cmap='gray')
        plt.show()

#------------------------------------

# sersic
#show_outlier(vars=2, lim=5.99, out='sup', num=0) # out of 5.99
#show_outlier(vars=2, lim=0.11, out='inf', num=0) # out of 0.11

#------------------------------------

# hlr
#show_outlier(vars=1, lim=1, out='sup', num=0) # rare visible Gx (larger pix->large hlr), no problem here


#show_outlier(vars=1, lim=50, out='sup', num=0) #maybe a Gx -> wrong hlr (here 250), sersic ok


#show_outlier(vars=1, lim=3, out='sup', num=0) # very noisy, hlr=4.5, sersic=6
# -> in the hlr outliers AND appears in the sersic outliers too


#show_outlier(vars=1, lim=1e6, out='sup', num=0) # very noisy, large image -> hlr very hight = 1e7

#show_outlier(vars=1, lim=1e4, out='sup', num=1) # same here, hlr = 2e4

#------------------------------------

# q
#show_outlier(vars=3, lim=0.06, out='inf', num=100) # noisy, hlr=14.9 -> cluster of q and hlr linked

# out of 1 :
#for i in range(14):
#    show_outlier(vars=3, lim=1, out='sup', num=i, Show=False)
# hlr = 0 for each and noisy image
