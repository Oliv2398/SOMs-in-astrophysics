import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from minisom import MiniSom

path = "datas/COSMOS_25.2_training_sample/real_galaxy_catalog_25.2.fits"

with fits.open(path) as hdul:
    hdul.info()
    hdr = hdul[0].header
    data = hdul[1].data
    cols = hdul[1].columns
    #cols.info()
    Names = cols.names

    datacols = []
    print("\nExtracting float datas...")
    print("Columns number :", len(Names))
    x=0 ; idx=[]
    for column in range(len(Names)):
        try:
            datacols.append(data.field(column).astype(None))
        except ValueError:
            datacols.append(np.zeros(data.shape[0]))
            idx.append(column)
            x+=1
    print(x, "columns ignored :",
        [Names[idx[i]] for i in range(len(idx))],
        ", indices :", idx)
    print("\n\n")


    choice=[3,10,11]
    print("Computing SOMs with cols :", [Names[choice[i]] for i in range(len(choice))])
    data_som = np.vstack([datacols[choice[i]] for i in range(len(choice))]).T
    som = MiniSom(30, 30, len(data_som[0]), sigma=0.5,
                  learning_rate=.5,
                  neighborhood_function='gaussian')
    som.train_random(data_som , 5000, verbose=True)

    plt.imshow(abs(som.get_weights()))
    plt.show()
