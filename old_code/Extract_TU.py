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
    return [mag, half_light_radius, q, SSersic_n]


def to_fits_file(path, new_file_name): # transform list to fits TU catalog

    TU = galsim.Catalog(path)

    Params=[]
    for i in range(TU.nobjects):
        Params.append(get_params(TU, i, 1))
        if i%10:
            print('\r [ %d / %d ] ; %d %%'%(i+1, TU.nobjects, 100*(i+1)/TU.nobjects), end='')

    Params = np.concatenate(Params)
    Params = Params.T.reshape(TU.nobjects, 4)

    t = Table(Params, names=('mag', 'half_light_radius', 'q', 'SSersic_n'))
    t.write(new_file_name+'.fits', format='fits') # creating the fits file


path = "datas/TU_cat_field_0.list"
file_name = "TU_created0"
#to_fits_file(path, file_name)
