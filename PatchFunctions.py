import numpy as np
import matplotlib
import sys
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from PolAngle import *
from map import *
from convolution import *


def Plot_beam_rectangular(beam_to_Plot,c_min,c_max, X_width, Y_width,title, axis):
    print("beam max:",np.max(beam_to_Plot),"beam min:",np.std(beam_to_Plot))
   
    plt.figure(figsize=(4, 4))
    im = plt.imshow(beam_to_Plot, interpolation='bilinear', origin='lower',cmap=cm.magma)
    im.set_clim(c_min,c_max)
    cbar = plt.colorbar()
    im.set_extent([0,X_width,0,Y_width])
    plt.ylabel(axis)
    plt.xlabel(axis)
    cbar.set_label('amplitud (arb)', rotation=270)
    plt.title(title)
    #plt.show()
    return(0)



def make_2d_beam_cordinates_r(Nx, Ny, pix_size):
    # make a 2d coordinate system
    onesx = np.ones(Nx)
    onesy = np.ones(Ny)
    
    indsx = (np.arange(Nx)+.5 - Nx/2.) * pix_size
    indsy = (np.arange(Ny)+.5 - Ny/2.) * pix_size
    
    X = np.outer(onesy, indsx)
    Y = np.outer(indsy, onesx)
    
    R = np.sqrt(X**2. + Y**2.)
    Theta = np.arctan2(Y,X)
    return(X,Y,R,Theta)


def make_little_buddies_r(Nx, Ny,pix_size,beam_size_fwhp,bs,main_beam_peak):
    # make a 2d coordinate system
    X,Y,R,Theta = make_2d_beam_cordinates_r(Nx, Ny,pix_size)
    #set up four buddies
    budy_sigma = bs["budy"]["FWHP"] / np.sqrt(8.*np.log(2))
    X_rot = X * np.cos(bs["budy"]["psi"]) - Y*np.sin(bs["budy"]["psi"])
    Y_rot = X * np.sin(bs["budy"]["psi"]) + Y*np.cos(bs["budy"]["psi"])
    buddy_beam = 0.
    R_temp = np.sqrt((X_rot - bs["budy"]["R"])**2. + Y_rot**2.)
    buddy_beam+= np.exp(-.5 *(R_temp/budy_sigma)**2.)
    R_temp = np.sqrt((X_rot + bs["budy"]["R"])**2. + Y_rot**2.)
    buddy_beam+= np.exp(-.5 *(R_temp/budy_sigma)**2.)
    R_temp = np.sqrt(X_rot**2. + (Y_rot - bs["budy"]["R"])**2.)
    buddy_beam+= np.exp(-.5 *(R_temp/budy_sigma)**2.)
    R_temp = np.sqrt(X_rot**2. + (Y_rot + bs["budy"]["R"])**2.)
    buddy_beam+= np.exp(-.5 *(R_temp/budy_sigma)**2.)
    buddy_beam = buddy_beam / np.max(buddy_beam)   ##  normalize the budy beam to 1
    ## make the buddy beams
    Budy_TT = buddy_beam / np.max(buddy_beam) * main_beam_peak * bs["budy"]["A"]  ## guarentees the budies are peak normalized realative to main beam
    Budy_QT = buddy_beam / np.max(buddy_beam) * main_beam_peak * bs["budy"]["A"] * bs["budy"]["polfracQ"]
    Budy_UT = buddy_beam / np.max(buddy_beam) * main_beam_peak * bs["budy"]["A"] * bs["budy"]["polfracU"]
    ## return the buddies
    return(Budy_TT,Budy_QT,Budy_UT)


def make_ghosting_beam_r(Nx, Ny,pix_size,beam_size_fwhp,bs,main_beam_peak):
    # make a 2d coordinate system
    X,Y,R,Theta = make_2d_beam_cordinates_r(Nx, Ny,pix_size)
    ## make shelf
    in_shelf = np.where(R < bs["ghostshelf"]["Diam"])
    shelf = np.zeros(np.shape(R))
    shelf[in_shelf] = 1.
    roll_off_kernal = np.exp(-.5 *(R/bs["ghostshelf"]["roll_off"])**2.)
    shelf = convlolve(shelf,roll_off_kernal)
    shelf = shelf / np.max(shelf) * main_beam_peak * bs["ghostshelf"]["A"]  ## normalized relative to the peak of the main beam
    return(shelf)



def make_cross_talk_beam_grid_r(Nx, Ny, pix_size, beam_size_fwhp, bs):
    # make a 2d coordinate system
    X, Y, R, Theta = make_2d_beam_cordinates_r(Nx, Ny ,pix_size)

    ## make cross-talk grid
    hex_grid = np.zeros((Ny, Nx))  # Modifica la forma in base a Nx e Ny

    delta_Ny = np.floor(bs["hex_crostalk"]["grid_space"] / pix_size)
    delta_Ny_nextrow = delta_Ny / 2.
    delta_Nx_nextrow = delta_Ny * np.cos(30. * np.pi / 180.)

    i = -1 * bs["hex_crostalk"]["N"]
    while (i <= bs["hex_crostalk"]["N"]):
        j = -1 * bs["hex_crostalk"]["N"]
        while (j <= bs["hex_crostalk"]["N"]):
            i_active = Ny / 2 + i * delta_Ny
            j_active = Nx / 2 + j * delta_Nx_nextrow
            if (i % 2 == 1):
                j_active -= delta_Ny_nextrow  ## offset every other row
            i_active = int(i_active)
            j_active = int(j_active)
            if ((i_active != Ny / 2) or (j_active != Nx / 2)):  # exclude the main beam
                distance = np.sqrt(np.abs((i_active - Ny / 2) / delta_Ny) ** 2 + np.abs(
                    (j_active - Nx / 2) / delta_Ny) ** 2)
                xtalk_tau = 1. / np.log(bs["hex_crostalk"]["neighbor_exp_fall"])
                distance /= xtalk_tau
                hex_grid[i_active, j_active] = np.exp(distance)
            j += 1
        i += 1

    const = np.ones(np.shape(hex_grid))
    hex_grid *= np.sum(const)  ## normalize the gain so that the x-talk amplitude is defined correclty
    return hex_grid






def make_systematics_beams_r(Nx, Ny,pix_size,beam_size_fwhp, beam,bs):
    
        # intitalize the beam to zero
        B_TT=B_QQ=B_UU=B_QT=B_UT=B_QU=B_UQ=0.
    
        ## merge this into the beam
        B_TT += beam
        B_QQ += beam
        B_UU += beam
    
        beam_peak = np.max(beam)
    
        # make the little buddies
        Budy_TT,Budy_QT,Budy_UT = make_little_buddies_r(Nx, Ny,pix_size,beam_size_fwhp,bs,beam_peak)
        ## merge this into the beam
        B_TT += Budy_TT
        B_QT += Budy_QT
        B_UT += Budy_UT

        # make the ghosting shelf
        shelf = make_ghosting_beam_r(Nx, Ny,pix_size,beam_size_fwhp,bs,beam_peak)
        ## merge this into the beam
        B_TT += shelf
        B_QQ += shelf
        B_UU += shelf
    
        # make the cross talk beam
        # make a hex grid centered on the beam
        hex_grid = make_cross_talk_beam_grid_r(Nx,Ny,pix_size,beam_size_fwhp,bs)
        #convolve the hex grid with the beam
        cross_talk = convlolve(hex_grid,beam)
        ## merge this into the beam
        B_TT += cross_talk
        B_QQ += cross_talk
        B_UU += cross_talk
        B_QT += cross_talk
        B_UT += cross_talk
    
        '''
        ## add the monopole + dipole + quadrupole T->P leakages
        # make the beam modes
        mono,dip_x,dip_y,quad_x,quad_45 = make_monopole_dipole_quadrupole(N,pix_size,beam_size_fwhp,bs)
        TtoQ = bs["TtoQ"]["mono"] * mono
        TtoQ += bs["TtoQ"]["dip_x"] * dip_x
        TtoQ += bs["TtoQ"]["dip_y"] * dip_y
        TtoQ += bs["TtoQ"]["quad_x"] * quad_x
        TtoQ += bs["TtoQ"]["quad_45"] * quad_45
        TtoU = bs["TtoU"]["mono"] * mono
        TtoU += bs["TtoU"]["dip_x"] * dip_x
        TtoU += bs["TtoU"]["dip_y"] * dip_y
        TtoU += bs["TtoU"]["quad_x"] * quad_x
        TtoU += bs["TtoU"]["quad_45"] * quad_45
        ## add to the beams
        #B_QT += TtoQ
        #B_UT += TtoU
        '''
    
        return(B_TT,B_QQ,B_UU,B_QT,B_UT,B_QU,B_UQ)  
    
    
    
def cosine_window_r(Nx, Ny, pix_size):
    "makes a cosine window for apodizing to avoid edges effects in the 2d FFT" 
    # make a 2d coordinate system

    Nx = int(Nx)
    Ny = int(Ny)

    onesx = np.ones(Nx)
    onesy = np.ones(Ny)

    indsx = (np.arange(Nx)+.5 - Nx/2.) * pix_size
    indsy = (np.arange(Ny)+.5 - Ny/2.) * pix_size

    X = np.outer(onesy, indsx)
    Y = np.outer(indsy, onesx)

    # make a window map
    window_map = np.cos(X) * np.cos(Y)
   
    # return the window map
    return(window_map)



def correct_lr_r(map, init_beam,Nx, Ny, pix_size, mode='onestep',pad = None, filter_eps=1e-10, eps=1e-12, post_f=None, iter=50):
    from copy import deepcopy
    # GET MAP PARAMETERS 
    Nx = int(Nx)
    Ny = int(Ny)

    onesx = np.ones(Nx)
    onesy = np.ones(Ny)

    indsx = (np.arange(Nx)+.5 - Nx/2.) * pix_size
    indsy = (np.arange(Ny)+.5 - Ny/2.) * pix_size

    X = np.outer(onesy, indsx)
    Y = np.outer(indsy, onesx)
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





def QU2EB_r(Nx, Ny,pix_size,Qmap,Umap,):
    '''Calcalute E, B maps given input Stokes Q, U maps'''


    # Create 2d Fourier coordinate system.
    onesx = np.ones(Nx)
    onesy = np.ones(Ny)
    indsx  = (np.arange(Nx) - Nx/2.) /(Nx-1.)
    indsy  = (np.arange(Ny) - Ny/2.) /(Ny-1.)

    
    kX = np.outer(onesy,indsx) / (pix_size/60. * np.pi/180.)
    kY = np.outer(indsy, onesx) / (pix_size/60. * np.pi/180.)
    
    ang = np.arctan2(kY,kX)
 
    # Convert to Fourier domain.
    fQ = np.fft.fftshift(np.fft.fft2(Qmap))
    fU = np.fft.fftshift(np.fft.fft2(Umap))
    
    # Convert Q, U to E, B in Fourier domain.
    fE = fQ * np.cos(2.*ang) + fU * np.sin(2. *ang)
    fB = - fQ * np.sin(2.*ang) + fU * np.cos(2. *ang)
    
    # Convert E, B from Fourier to real space.
    Emap = np.real(np.fft.ifft2(np.fft.fftshift(fE)))
    Bmap = np.real(np.fft.ifft2(np.fft.fftshift(fB)))

    return Emap, Bmap



def calculate_2d_spectrum_r(Map1,Map2,delta_ell,ell_max,pix_size,Nx, Ny):
    "calcualtes the power spectrum of a 2d map by FFTing, squaring, and azimuthally averaging"
     # Create 2d Fourier coordinate system.
  
    
    Nx = int(Nx)
    Ny = int(Ny)
    # make a 2d ell coordinate system
    onesx = np.ones(Nx)
    onesy = np.ones(Ny)
    indsx  = (np.arange(Nx) - Nx/2.) /(Nx-1.)
    indsy  = (np.arange(Ny) - Ny/2.) /(Ny-1.)
    
    kX = np.outer(onesy,indsx) / (pix_size/60. * np.pi/180.)
    kY = np.outer(indsy, onesx) / (pix_size/60. * np.pi/180.)
    K = np.sqrt(kX**2. + kY**2.)
    ell_scale_factor = 2. * np.pi 
    ell2d = K * ell_scale_factor
    
    # make an array to hold the power spectrum results
    N_bins    = int(ell_max/delta_ell)
    ell_array = np.arange(N_bins)
    CL_array  = np.zeros(N_bins)
    
    # get the 2d fourier transform of the map
    FMap1 = np.fft.ifft2(np.fft.fftshift(Map1))
    FMap2 = np.fft.ifft2(np.fft.fftshift(Map2))
    PSMap = np.fft.fftshift(np.real(np.conj(FMap1) * FMap2))
    # fill out the spectra
    i = 0
    while (i < N_bins):
        ell_array[i] = (i + 0.5) * delta_ell
        inds_in_bin = ((ell2d >= (i* delta_ell)) * (ell2d < ((i+1)* delta_ell))).nonzero()
        CL_array[i] = np.mean(PSMap[inds_in_bin])
        #print i, ell_array[i], inds_in_bin, CL_array[i]
        i = i + 1
 
    # return the power spectrum and ell bins
    return(ell_array,CL_array*np.sqrt(pix_size /60.* np.pi/180.)*2.)