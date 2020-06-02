import galsim
import matplotlib.pyplot as plt
from astropy.io import fits

"""
First, we create the COSMOSCatalog object. The parameter is the name of the folder containing all the field it needs (psf, "image" (which are not really images), catalogueSSS, etc.
This object has many methods, which allows us to convert any line of the Cosmos Catalog into the corresponding image, with the pixels scale, noise, psf we want.
"""


galsim_cat = galsim.COSMOSCatalog(dir=path_cosmos+'COSMOS_25.2_training_sample')


"""
Few parameters for the production of the image. 128 and 0.03 are the native parameters of HST
"""

stamp_size = 128
pixel_scale = 0.03


"""
This method transform any index (or list of index) into a galsim.GSObject object, which is 
kind of surface brightness profil (see the galsim doc for more info). This is still not 
an image, or an array
"""
gal = galsim_cat.makeGalaxy(index = None, noise_pad_size= stamp_size * pixel_scale)

"""
the GSOObject has also the corresponding psf information. To recover it, use that 
"""
psf = gal.original_psf

"""
Convolve the surface brightness profile with the PSF
"""
gal = galsim.Convolve(gal, psf)

"""
Now, we need to create an image (empty for now), in which we will "draw" the surface
brightness profile? We use ImageF (see the doc for more info), precising the size
Note that this is just an "empty canvas" for now
"""

image_gal = galsim.ImageF(stamp_size, stamp_size)
image_psf = galsim.ImageF(stamp_size, stamp_size)

"""
Finally, you can draw the GSOObject into the image 
"""
gal.drawImage(image=image_gal);
psf.drawImage(image=image_psf);

"""
gal and psf are still galsim objects. You may do a lot of things with that. 
Maybe the most interesting is "gal.array, which is (fianally !) the image".
You can easily show it with plt.imshow
"""

plt.imshow(image_gal.array)

"""
I haven't tried with a list of indexes, but it should be the same. 
I guess that in this case, the image_gal will be a cube of image
"""

# Have Fun !