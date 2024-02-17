import numpy as np
import matplotlib
import sys
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from PolAngle import *
from beam import * 
from map import *

# CONVOLVE MAP WITH A PERFECT GAUSSIAN BEAM
def make_2d_gaussian_beam(N,pix_size,beam_size_fwhp):
     # make a 2d coordinate system
    N=int(N)
    ones = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.) * pix_size
    X = np.outer(ones,inds)
    Y = np.transpose(X)
    R = np.sqrt(X**2. + Y**2.)
  
    # make a 2d gaussian 
    beam_sigma = beam_size_fwhp / np.sqrt(8.*np.log(2))
    gaussian = np.exp(-.5 *(R/beam_sigma)**2.)
    gaussian = gaussian / np.sum(gaussian)
 
    # return the gaussian
    return(gaussian)

def convolve_map_with_gaussian_beam(N,pix_size,beam_size_fwhp,Map):
    "convolves a map with a gaussian beam pattern.  NOTE: pix_size and beam_size_fwhp need to be in the same units" 
    # make a 2d gaussian 
    gaussian = make_2d_gaussian_beam(N,pix_size,beam_size_fwhp)
  
    # do the convolution
    FT_gaussian = np.fft.fft2(np.fft.fftshift(gaussian))
    FT_Map = np.fft.fft2(np.fft.fftshift(Map))
    convolved_map = np.fft.fftshift(np.real(np.fft.ifft2(FT_gaussian*FT_Map)))
    
    # return the convolved map
    return(convolved_map)




# CONVOLVE MAP WITH ANY GIVEN BEAM

def convolve_map_with_beam(Map, beam):

    FT_beam = np.fft.fft2(np.fft.fftshift(beam))
    FT_Map  = np.fft.fft2(np.fft.fftshift(Map))
    
    convolved_map = np.fft.fftshift(np.real(np.fft.ifft2(FT_beam*FT_Map)))

    return(convolved_map)


#################################################################################################################################################################################################################################################################################################################################################################################################################################
# DECONVOLUTION

# da MAPEXT ( https://github.com/tjrennie/MAPEXT)
def richardson_lucy(image, psf, im_deconv=None, num_iter=1, clip=False, 
                     filter_epsilon=False, eps=1e-12, progress=False):
    
    if im_deconv is None:
        im_deconv = np.full(image.shape, 0.5, dtype=float)
    
    psf_mirror = np.flip(psf)
    
    if progress:
        iter_range = tqdm(range(num_iter)) if not progress == 'notebook' else tqdm_notebook(range(num_iter))
    else:
        iter_range = range(num_iter)
        
    for i in iter_range:
        conv = convolve(im_deconv, psf, mode='same') + eps
        if filter_epsilon:
            relative_blur = np.where(conv < filter_epsilon, 0, image / conv)
        else:
            relative_blur = image / conv
        
        im_deconv *= convolve(relative_blur, psf_mirror, mode='same')
        
        
    return im_deconv



def correct_lr(map, init_beam,N, pix_size, mode='onestep',pad = None, filter_eps=1e-10, eps=1e-12, post_f=None, iter=50):
    from copy import deepcopy
    # GET MAP PARAMETERS 
    N = int(N)
    ones = np.ones(N)
    inds = (np.arange(N)+.5 - N/2.) * pix_size
    X = np.outer(ones,inds)
    Y = np.transpose(X)
    r = np.sqrt(X**2. + Y**2.)
    
    
    
    #shape = [4*(aMap.MAP.shape[0]+1)-1, 4*(aMap.MAP.shape[1]+1)-1]
    #init_bmap, r = init_beam.return_bMap(pixscale=pixscale,shape=[811,811],rad=True, oversamp=1)
    init_bmap = map
    
    init_bmap = 10**(init_bmap/10) * merge_cosine(r, center = 345, width=40)
    init_bmap /= np.nanmax(init_bmap)
  
    init_bmap = init_bmap / np.nanmax(init_bmap)

    init_bmap[init_bmap<0] = 0.
    
    # DECONVOLVE
    if mode=='onestep':
        psf_ = init_bmap / np.nansum(init_bmap)
        dat_ = deepcopy(map)
        if pad!=None:
            pad_pix = int(pad*np.max(np.shape(dat_)))
            new = np.ones([np.shape(dat_)[0]+(2*pad_pix), np.shape(dat_)[1]+(2*pad_pix)])*np.nanmedian(dat_)
            new[pad_pix:-pad_pix, pad_pix:-pad_pix] = dat_
            dat_ = new
        map_ = deepcopy(dat_)
        if iter==None:
            SNAPSHOTS = np.array([0,10,20,30,40,50])
        else:
            SNAPSHOTS = np.array([0,int(iter)])
        
        allframes = np.zeros([np.nanmax(SNAPSHOTS)+1,np.shape(dat_)[0],np.shape(dat_)[1]])
        frames = np.zeros([len(SNAPSHOTS),np.shape(dat_)[0],np.shape(dat_)[1]])
        frames[0] = map_
        allframes[0] = map_
        
        n_iter = 0
        i = 1
        
    return map_



def merge_cosine(X,center=30,width=5):
    mask = np.ones(X.shape)
    mask[X>center] = 0.
    funcmsk = np.all([X>center-width/2,X<center+width/2],axis=0)
    mask[funcmsk] = (1-np.sin((X[funcmsk]-center)*np.pi/width))/2
    return mask**2
#################################################################################################################################################################################################################################################################################################################################################################################################################################
'''
def richardson_lucy_polarized(image, psf_x, psf_y, im_deconv=None, num_iter=50, clip=False, filter_epsilon=False, eps=1e-12, progress=False):
    if im_deconv is None:
        im_deconv_x = np.full(image.shape, 0.5, dtype=float)
        im_deconv_y = np.full(image.shape, 0.5, dtype=float)
    else:
        im_deconv_x, im_deconv_y = im_deconv

    psf_x_mirror = np.flip(psf_x)
    psf_y_mirror = np.flip(psf_y)

    # Small regularization parameter used to avoid 0 divisions
    if progress:
        iter_range = tqdm(range(num_iter))
    else:
        iter_range = range(num_iter)

    for _ in iter_range:
        conv_x = convolve(im_deconv_x, psf_x, mode='same') + eps
        conv_y = convolve(im_deconv_y, psf_y, mode='same') + eps

        if filter_epsilon:
            relative_blur_x = np.where(conv_x < filter_epsilon, 0, image / conv_x)
            relative_blur_y = np.where(conv_y < filter_epsilon, 0, image / conv_y)
        else:
            relative_blur_x = image / conv_x
            relative_blur_y = image / conv_y

        im_deconv_x *= convolve(relative_blur_x, psf_x_mirror, mode='same')
        im_deconv_y *= convolve(relative_blur_y, psf_y_mirror, mode='same')

    if clip:
        im_deconv_x[im_deconv_x > 1] = 1
        im_deconv_x[im_deconv_x < -1] = -1

        im_deconv_y[im_deconv_y > 1] = 1
        im_deconv_y[im_deconv_y < -1] = -1

    return im_deconv_x, im_deconv_y






    def deconvolve_map_with_beam(input_map, beam, iters=50, eps=None, filter_eps=1e-20, plot=False):
    """
    Deconvolve a map from a given beam using Richardson-Lucy deconvolution.

    Parameters:
    - input_map: Input map to deconvolve (NumPy array).
    - beam: Beam to use for deconvolution (NumPy array).
    - iters: Number of Richardson-Lucy iterations.
    - eps: Richardson-Lucy epsilon parameter.
    - filter_eps: Filter epsilon parameter.
    - plot: Boolean, whether to generate and display plots during deconvolution.

    Returns:
    - Deconvolved map (NumPy array).
    """
    # Initialize the deconvolved map with a uniform map
    deconvolved_map = np.full_like(input_map, 0.5, dtype=float)

    # Perform Richardson-Lucy deconvolution
    for iteration in tqdm(range(iters)):
        # Perform one iteration of Richardson-Lucy
        deconvolved_map = richardson_lucy(input_map, beam, im_deconv=deconvolved_map, num_iter=1, eps=eps, filter_epsilon=filter_eps)

        # Plotting intermediate results
        if plot and iteration % 5 == 0:
            plt.figure(figsize=(6, 5))
            plt.imshow(deconvolved_map, cmap='jet')
            plt.title(f'Iteration: {iteration}')
            plt.colorbar()
            plt.show()

    return deconvolved_map



    def convolve_map_polarized_systematic_beam(B_TT,B_QQ,B_UU,B_QT,B_UT,B_QU,B_UQ,QMap,UMap,TMap):
    "convolves a map with a gaussian beam pattern.  NOTE: pix_size and beam_size_fwhp need to be in the same units" 
    # do the convolutions   
    sys_map_T = convlolve(B_TT,TMap)
    sys_map_Q = convlolve(B_QQ,QMap) + convlolve(B_QT,TMap) #+ convlolve(B_QU,UMap)    
    sys_map_U = convlolve(B_UU,QMap) + convlolve(B_UT,TMap) #+ convlolve(B_UQ,QMap) 
    # return the convolved map
    return(sys_map_T,sys_map_Q,sys_map_U)

    '''