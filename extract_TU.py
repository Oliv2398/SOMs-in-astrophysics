import numpy as np
from scipy.stats import skewnorm
import galsim
from astropy.table import Table

def TotRadius(bulge_radius, disk_radius, bt):
    if bt == 0.:
        radius = disk_radius
    elif bt == 1.:
        radius = bulge_radius
    else:
        if (bulge_radius < disk_radius):
            radius = np.power((disk_radius*(1-bt)), 0.8) + \
                np.power((bulge_radius * (bt)), 0.8)
        else:
            radius = np.power((disk_radius*(1-bt)), 2) + \
                np.power((bulge_radius * (bt)), 2)
    return radius

def get_params(cat, i, pixel_scale):   #cat = TU cat, i = row of the cat
    radiusBulge = cat.getFloat(i, 5)
    radiusDisk = cat.getFloat(i, 8)*1.6783
    bt = cat.getFloat(i, 4)
    ell_B = cat.getFloat(i, 6)
    ell_D = cat.getFloat(i, 9)
    q = bt*ell_B + (1-bt)*ell_D
    mag = cat.getFloat(i, 3)
    PA = cat.getFloat(i, 7)

    # sersic n param prop to bt
    Nb = bt * 4 + 0.4
    SSersic_n = skewnorm.rvs(6, Nb, 0.3, size=1)[0]

    if SSersic_n < 0.1:
        SSersic_n = np.random.normal(0.5, 0.1, 1)
        if SSersic_n < 0.1:
            SSersic_n = 0.1
    if SSersic_n > 6:
        SSersic_n = skewnorm.rvs(6, 5.5, 0.1, size=1)[0]
        if SSersic_n > 6:
            SSersic_n = 5.6

    # Light radius in pixel
    light_radius = TotRadius(radiusBulge, radiusDisk, bt) / pixel_scale
    half_light_radius = float(light_radius) * np.sqrt(q)
    return [mag, half_light_radius, q, SSersic_n, PA, bt]


def cut_hight_hlr(cat, lim): # delete hlr problems
    idx = np.where(cat[:,1]>lim)[0]
    print("element suppr :", idx.shape)
    cat = np.delete(cat,idx)
    return cat


def to_fits_file(): # transform list to fits TU catalog
    path_TU = "datas/TU_cat_field_0.list"

    TU = galsim.Catalog(path_TU)

    idx = 0
    Params=[]
    while True: # extracting lines one by one
        try:
            para = get_params(TU, idx, 1)
            Params.append(para)
            idx+=1
            print(idx, end='\r')
        except :
            print(idx, "lines")
            break

    Params = np.concatenate(Params)
    Params = Params.T.reshape(idx,6)

    #Params = cut_hight_hlr(Params,1) # cut hlr

    t = Table(Params, names=('mag', 'half_light_radius', 'q', 'SSersic_n', 'PA', 'bt'))
    t.write('TU_created.fits', format='fits') # creating the fits file


to_fits_file()
