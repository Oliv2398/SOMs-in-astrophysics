import numpy as np
import galsim
from astropy.io import fits
from astropy.table import Table

# path to the file to modify
Path = "datas/TU_cat_field_0.list"


#~~~~~~~~~~~~~~~~~~~~

def get_shape(cat): # to get the shape of the catalog
    lines = 0
    while True:
        try :
            get_lines = cat.get(lines, 0)
            lines+=1
        except:
            print('Lines :', lines)
            break

    cols = 0
    while True:
        try :
            get_col = cat.get(0, cols)
            cols+=1
        except:
            print('Columns :', cols)
            break

    return lines, cols


def extract_full(path): # extract the catalog

    cat = galsim.Catalog(path)
    print("Catalog loaded")

    N, M = get_shape(cat)
    table=np.zeros((N, M))

    print("\nCollecting...")
    for i in range(N):
        for j in range(M):
            table[i,j] = cat.get(i,j)
            print("[ {} / {} ] :".format(i,N), np.round(100*i/N,1),'%', end='\r')
    return table


def to_fits_file_full(table, *arg): # transform list file to fits file
    # arg : 'TU' for True Universe catalog
    #       'choice' to full every column by hand
    #       'auto' filling with [col0,col1,...]

    if arg==('TU',): # TU catalog column's name ()
        header = ['col0','col1','col2',
            'mag', #3
            'bt', #4
            'radiusBulge', #5
            'ell_B', #6
            'PA', #7
            'radiusDisk', #8
            'ell_D', #9
            'col10', 'col11']

    elif arg==('choice',): # full every column's name
        header = []
        for i in range(table.shape[1]):
            header.append(str(input('Colum {} name '.format(i))))

    else:
        header=[ 'col{}'.format(i) for i in range(table.shape[1]) ]

    print('\n\nHeaders :', len(header))
    print(header)

    t = Table(list(table.T),
        names=header)
    t.write('TU_created_full.fits', format='fits') # creating the fits file
    print('saved')


#~~~~~~~~~~~~~~~~~~~~

cat_full = extract_full(Path)

to_fits_file_full(cat_full)
