# Code used to convert .jpg files to .bmp objects
# Jody LeSage
# ID: HX85135
# CMSC471
# Spring 2016

import sys
import os
from PIL import Image
from io import BytesIO

# takes an image file destination
# returns a bitmap image object
def toBitmap(src, verbose=False):
    if(verbose):
        print('Converting', src, 'to bitmap...')
    srcImg = Image.open(src)
    bmp = BytesIO(srcImg.convert("1", dither=Image.NONE).tobitmap())
    return Image.open(bmp)

# main
# takes two command line arguments
# both are folders, in the format [source] [destination]
# attempts to open all files in the src folder, converts
# them to bitmaps, and saves them in the destination folder
if __name__ == '__main__':
    src = sys.argv[1]
    dst = sys.argv[2]
    for filename in os.listdir(src):
        bmp = toBitmap(src + '/' + filename)
        fn = filename.split('.')
        bmp.save(dst + '/' + fn[0] + '.bmp')
