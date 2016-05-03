# library of hash functions used to featurize bitmap Image objects
# Jody LeSage
# ID: HX85135
# CMSC471
# Spring 2016

from PIL import Image
import numpy as np

# converts a bitmap Image object into a 2d array of booleans (white = True, black = False)
def toArray(bmp, verbose=False):
    if(verbose):
        print('        Converting bitmap to array...')
    array = np.array(list(bmp.getdata()))
    array = array.reshape(bmp.size)
    return array
    
# how many black pixels are there
def positivePixels(src, verbose=False):
    if(verbose):
        print('        Calculating number of black pixels...')
    return (src.size - np.count_nonzero(src))
    
# what is the first row to contain a black pixel (top to bottom)
def firstRow(src, verbose=False):
    if(verbose):
        print('        Finding first row with a black pixel...')
    rowindex = 0
    for row in src:
        for pixel in row:           # if 0 in row:
            if(not pixel):
                return rowindex
        rowindex += 1
    return rowindex - 1 # in case there are no black pixels in the bitmap

# what is the first column to contain a black pixel (left to right)
def firstColumn(src, verbose=False):
    if(verbose):
        print('        Finding first column with a black pixel...')
    return firstRow(src.T, False) # shortcut: just do the row search on a transposed matrix
    
# what is the last row to contain a black pixel (bottom to top)
def lastRow(src, verbose=False):
    if(verbose):
        print('        Finding last row with a black pixel...')
    return len(src) - firstRow(reversed(src), False) # shortcut: reverse row order and subtract

# what is the last column to contain a black pixel (right to left)
def lastColumn(src, verbose=False):
    if(verbose):
        print('        Finding last column with a black pixel...')
    return lastRow(src.T, False) # again, cheat by using the transposed matrix
    
# what is the width of the symbol?
def width(src, verbose=False):
    if(verbose):
        print('        Calculating width of the symbol...')
    return lastColumn(src, False) - firstColumn(src, False)
    
# what is the height of the symbol?
def height(src, verbose=False):
    if(verbose):
        print('        Calculating height of the symbol...')
    return lastRow(src, False) - firstRow(src, False)

# crop out just the symbol
def crop(src, verbose=False):
    if(verbose):
        print('        Cropping symbol...')
    top = firstRow(src)
    bottom = lastRow(src)
    left = firstColumn(src)
    right = lastColumn(src)
    return src[top:bottom,left:right]

# divide src into 16 pieces, count black pixels in each region
def hash16(src, verbose=False):
    if(verbose):
        print('        Hashing bitmap using 16 regions...')
    width = src.shape[0]
    height = src.shape[1]
    r = []
    w = [0, int(width/4), int(width/2), int(3*width/4), int(width)]
    h = [0, int(height/4), int(height/2), int(3*height/4), int(height)]
    for x in range(len(w) - 1):
        for y in range(len(h) - 1):
            r.append(positivePixels(src[w[x]:w[x+1],h[y]:h[y+1]]))
    return r

# divide src into n^2 pieces, count black pizels in each region
def hash_n_by_n(src, n, verbose=False):
    if(verbose):
        print('        Hashing bitmap using', n*n, 'regions...')
    width = src.shape[0]
    height = src.shape[1]
    w = []
    h = []
    for x in range(n + 1):
        w.append(int(x*width/n))
        h.append(int(x*height/n))
    r = []
    for x in range(len(w) - 1):
        for y in range(len(h) - 1):
            r.append(positivePixels(src[w[x]:w[x+1],h[y]:h[y+1]]))
    return r

def vectorize(bitmap, n, verbose=False):
    array = toArray(bitmap, verbose)
    cropped = crop(array)
    total = positivePixels(cropped)
    r = hash_n_by_n(cropped, n, verbose)
    for i in range(len(r)):
        r[i] = r[i]/total   # normalize
    return r

def vectorizeGen2(bitmap, verbose=False):
    array = toArray(bitmap, verbose)
    return hash16(crop(array))

def vercorizeGen1(bitmap, verbose=False):
    HASH_LIST = [positivePixels, firstRow, firstColumn, lastRow, lastColumn, width, height]
    r = []
    for hashFunction in HASH_LIST:
        r += [hashFunction(array, verbose)]
    return r
