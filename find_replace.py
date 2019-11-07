# Find and replace data from image.
# Falagas Alekos

import os
import sys
import optparse
import rasterio
import numpy as np

class OptionParser (optparse.OptionParser):

    def check_required(self, opt):
        option = self.get_option(opt)

        # Assumes the option's 'default' is set to None!
        if getattr(self.values, option.dest) is None:
            self.error("{} is required".format(option))

def geoimwrite(name, image, width, height, count, crs, transform, dtype):
    """A simple function to write georeferenced images."""
    print ('Trying to write raster data...')
    # Export stacked color image
    bands=rasterio.open(name,'w',driver='Gtiff',width=width, height=height, count=count, crs=crs, transform=transform, dtype=dtype[0])
    for i in range(1, count+1):
        bands.write(image[i-1].astype(dtype[i-1]),i) #green
    bands.close()
    print('Done!')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        prog = os.path.basename(sys.argv[0])
        print("Run: python3 " + sys.argv[0] + " [options]")
        print("Help: python3 ", prog, " --help")
        print("or: python3 ", prog, " -h")
        sys.exit(-1)
    else:
        usage = "usage: %prog [options] "
        parser = OptionParser(usage=usage)
        parser.add_option("-r", "--reference", dest="ref", action="store", type="string", help="Reference image", default=None)
        parser.add_option("-i","--im", dest="im", action="store", type="string", help="Image that values will be replaced.", default=None)
        parser.add_option("-n", "--new", dest="new", action="store", type="string", help="New image.", default=None)
        (options, args) = parser.parse_args()
        parser.check_required("-r")
        parser.check_required("-i")

        if options.new is None:
            options.new = 'new-image.tif'

        # read reference raster
        print ('Reading reference image: {}...'.format(options.ref))
        ref_im = rasterio.open(options.ref)
        print ('Done!')
        # get metadata
        print ('Getting metadata...')
        width = ref_im.width
        height = ref_im.height
        count = ref_im.count
        crs = ref_im.crs
        transform = ref_im.transform
        dtype = ref_im.dtypes
        #read clean image
        print ('Reading clean image: {}...'.format(options.im))
        image = rasterio.open(options.im)
        print ('Done!')

        # works only if the images have the same bbox
        if ref_im.bounds == image.bounds:
            print ('Reading images as arrays...')
            ref_im_ar = ref_im.read()
            im_ar = image.read()
            # boolean masking
            ref_im_ar[im_ar > 0] = im_ar[im_ar>0]

            #write the result
            geoimwrite(options.new, ref_im_ar, width, height, count, crs, transform, dtype) #, dtype[0])
