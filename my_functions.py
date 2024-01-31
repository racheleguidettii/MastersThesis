import numpy as np
import matplotlib
import sys
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import astropy.io.fits as fits

def make_CMB_maps(N,pix_size,ell,DlTT,DlEE,DlTE,DlBB):
    "makes a realization of a simulated CMB sky map"

    # convert Dl to Cl, we use np.divide to avoid dividing by zero.
    dell = ell * (ell + 1) / 2 / np.pi
    ClTT = np.divide(DlTT, dell, where=ell>1)
    ClEE = np.divide(DlEE, dell, where=ell>1)
    ClTE = np.divide(DlTE, dell, where=ell>1)
    ClBB = np.divide(DlBB, dell, where=ell>1)
    
    # set the \ell = 0 and \ell =1 modes to zero as these are unmeasurmable and blow up with the above transform
    ClTT[0:2] = 0.
    ClEE[0:2] = 0.
    ClTE[0:2] = 0.
    ClBB[0:2] = 0.

    # separate the correlated and uncorrelated part of the EE spectrum
    correlated_part_of_E = np.divide(ClTE, np.sqrt(ClTT), where=ell>1)
    uncorrelated_part_of_EE = ClEE - np.divide(ClTE**2., ClTT, where=ell>1) * 1e-3 #per non fare uscire valori negativi
    
    correlated_part_of_E[0:2] = 0.
    uncorrelated_part_of_EE[0:2] = 0.
    
    # make a 2d coordinate system
    ones = np.ones(N)
    inds  = (np.arange(N) - N/2.) /(N-1.)
    X = np.outer(ones,inds)
    Y = np.transpose(X)
    R = np.sqrt(X**2. + Y**2.)
    ang = np.arctan2(Y,X)   ## we now need this angle to handle the EB <--> QU rotation
    
    # now make a set of 2d CMB masks for the T, E, and B maps
    ell_scale_factor = 2. * np.pi / (pix_size/60. * np.pi/180.)
    ell2d = R * ell_scale_factor
    ClTT_expanded = np.zeros(int(ell2d.max())+1)
    ClTT_expanded[0:(ClTT.size)] = ClTT
    ClEE_uncor_expanded = np.zeros(int(ell2d.max())+1)
    ClEE_uncor_expanded[0:(uncorrelated_part_of_EE.size)] = uncorrelated_part_of_EE
    ClE_corr_expanded = np.zeros(int(ell2d.max())+1)
    ClE_corr_expanded[0:(correlated_part_of_E.size)] = correlated_part_of_E
    ClBB_expanded = np.zeros(int(ell2d.max())+1)
    ClBB_expanded[0:(ClBB.size)] = ClBB
    CLTT2d = ClTT_expanded[ell2d.astype(int)]
    ClEE_uncor_2d = ClEE_uncor_expanded[ell2d.astype(int)]
    ClE_corr2d = ClE_corr_expanded[ell2d.astype(int)]
    CLBB2d = ClBB_expanded[ell2d.astype(int)]
    
    # now make a set of gaussian random fields that will be turned into the CMB maps
    randomn_array_for_T = np.fft.fft2(np.random.normal(0,1,(N,N)))
    randomn_array_for_E = np.fft.fft2(np.random.normal(0,1,(N,N))) 
    randomn_array_for_B = np.fft.fft2(np.random.normal(0,1,(N,N))) 
    
    ## make the T, E, and B maps by multiplying the masks against the random fields
    FT_2d = np.sqrt(CLTT2d) * randomn_array_for_T
    FE_2d = np.sqrt(ClEE_uncor_2d) * randomn_array_for_E + ClE_corr2d* randomn_array_for_T
    FB_2d = np.sqrt(CLBB2d) * randomn_array_for_B
    
    ## now conver E abd B to Q and U
    FQ_2d = FE_2d* np.cos(2.*ang) - FB_2d * np.sin(2. *ang)
    FU_2d = FE_2d* np.sin(2.*ang) + FB_2d * np.cos(2. *ang)
    
    ## convert from fourier space to real space
    CMB_T = np.fft.ifft2(np.fft.fftshift(FT_2d)) /(pix_size /60.* np.pi/180.)
    CMB_T = np.real(CMB_T)
    CMB_Q = np.fft.ifft2(np.fft.fftshift(FQ_2d)) /(pix_size /60.* np.pi/180.)
    CMB_Q = np.real(CMB_Q)
    CMB_U = np.fft.ifft2(np.fft.fftshift(FU_2d)) /(pix_size /60.* np.pi/180.)
    CMB_U = np.real(CMB_U)

    ## optional code for spitting out E and B maps 
    CMB_E = np.fft.ifft2(np.fft.fftshift(FE_2d)) /(pix_size /60.* np.pi/180.)
    CMB_E = np.real(CMB_E)
    CMB_B = np.fft.ifft2(np.fft.fftshift(FB_2d)) /(pix_size /60.* np.pi/180.)
    CMB_B = np.real(CMB_B)
    
    ## return the maps
    return(CMB_T,CMB_Q,CMB_U,CMB_E,CMB_B)

###########################################################################################################################################
###########################################################################################################################################
###########################################################################################################################################

def Plot_CMB_Map(Map_to_Plot,c_min,c_max,X_width,Y_width):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    #print("map mean:",np.mean(Map_to_Plot),"map rms:",np.std(Map_to_Plot))
    plt.gcf().set_size_inches(4, 4)
    im = plt.imshow(Map_to_Plot, interpolation='bilinear', origin='lower',cmap=cm.RdBu_r)
    im.set_clim(c_min,c_max)
    ax=plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    
    cbar = plt.colorbar(im, cax=cax)
    #cbar = plt.colorbar()
    im.set_extent([0,X_width,0,Y_width])
    plt.ylabel('angle $[^\circ]$')
    plt.xlabel('angle $[^\circ]$')
    cbar.set_label('tempearture [uK]', rotation=270)
    
    plt.show()
    return(0)

###########################################################################################################################################
######################################################################################################################################################################################################################################################################################

def polangle_ps(EE_ps, BB_ps, alpha):
    EE_ps_pert = EE_ps * (np.cos(2*alpha))**2 + BB_ps * (np.sin(2*alpha))**2
    BB_ps_pert = BB_ps * (np.cos(2*alpha))**2 + EE_ps * (np.sin(2*alpha))**2
    
    return (EE_ps_pert, BB_ps_pert)


def polangle_map(Q_map, U_map, gauss_center, std_deviation):
    pol_angle  = np.zeros_like(Q_map)
    pol_angle  = np.random.normal(gauss_center, std_deviation, pol_angle.shape)
    Q_map_pert = Q_map * np.cos(2*pol_angle) + U_map * np.sin(2*pol_angle)
    U_map_pert = U_map * np.cos(2*pol_angle) + Q_map * np.sin(2*pol_angle)
    
    return (Q_map_pert, U_map_pert)
#################################################################################################################################################################################################################################################################################################################################################################################################################################

def plotquiver(N, Q, U, X_width, Y_width, pix_size,title, background=None):
    '''Visualize Stokes Q, U as headless vectors'''
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    # Smooth maps for nicer images
    fwhm_pix = 30
    Q = convolve_map_with_gaussian_beam(N,pix_size,fwhm_pix,Q)
    U = convolve_map_with_gaussian_beam(N,pix_size,fwhm_pix,U)
    
    if background is not None:
        # If provided, we overplot the vectors on top of the smoothed background.
        background = convolve_map_with_gaussian_beam(N,pix_size,fwhm_pix,background)
    
    Q = Q[::int(fwhm_pix),::int(fwhm_pix)]
    U = U[::int(fwhm_pix),::int(fwhm_pix)]
    
    p_amp = np.sqrt(Q**2 + U**2)
    ang = np.arctan2(U, Q) / 2.
    
    u = p_amp * np.cos(ang)
    v = p_amp * np.sin(ang)

    x = np.linspace(0,X_width,u.shape[1])
    y = np.linspace(0,X_width,u.shape[0])    
        
    fig, ax = plt.subplots(figsize=(4,4))
    if background is not None:
        im = ax.imshow(background, interpolation='bilinear', origin='lower',cmap=cm.RdBu_r,
                       extent=([0,X_width,0,Y_width]))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('temperature [uK]', rotation=270)
        
    ax.quiver(x, y, u, v, headwidth=1, headlength=0, pivot='mid', units='xy',
              scale_units='xy', scale=2 * p_amp.max(), linewidth=1)
        
    ax.set_ylabel('angle $[^\circ]$')
    ax.set_xlabel('angle $[^\circ]$')
    ax.set_title(title)
    plt.show(fig)
    
    #return fig, p_amp, ang
    return fig

#################################################################################################################################################################################################################################################################################################################################################################################################################################

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

#################################################################################################################################################################################################################################################################################################################################################################################################################################

def cosine_window(N):
    "makes a cosine window for apodizing to avoid edges effects in the 2d FFT" 
    # make a 2d coordinate system
    ones = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.)/N *np.pi ## eg runs from -pi/2 to pi/2
    X = np.outer(ones,inds)
    Y = np.transpose(X)
  
    # make a window map
    window_map = np.cos(X) * np.cos(Y)
   
    # return the window map
    return(window_map)


#################################################################################################################################################################################################################################################################################################################################################################################################################################

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


#################################################################################################################################################################################################################################################################################################################################################################################################################################
def calculate_2d_spectrum(Map1,Map2,delta_ell,ell_max,pix_size,N):
    "calcualtes the power spectrum of a 2d map by FFTing, squaring, and azimuthally averaging"
    N = int(N)
    # make a 2d ell coordinate system
    ones = np.ones(N)
    inds = (np.arange(N)+.5 - N/2.) /(N-1.)
    kX = np.outer(ones,inds) / (pix_size/60. * np.pi/180.)
    kY = np.transpose(kX)
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



#################################################################################################################################################################################################################################################################################################################################################################################################################################

def make_noise_map(N,pix_size,white_noise_level,atmospheric_noise_level,one_over_f_noise_level):
    "makes a realization of instrument noise, atmosphere and 1/f noise level set at 1 degrees"
    ## make a white noise map
    N=int(N)
    white_noise = np.random.normal(0,1,(N,N)) * white_noise_level/pix_size
 
    ## make an atmosperhic noise map
    atmospheric_noise = 0.
    if (atmospheric_noise_level != 0):
        ones = np.ones(N)
        inds  = (np.arange(N)+.5 - N/2.) 
        X = np.outer(ones,inds)
        Y = np.transpose(X)
        R = np.sqrt(X**2. + Y**2.) * pix_size /60. ## angles relative to 1 degrees  
        mag_k = 2 * np.pi/(R+.01)  ## 0.01 is a regularizaiton factor
        atmospheric_noise = np.fft.fft2(np.random.normal(0,1,(N,N)))
        atmospheric_noise  = np.fft.ifft2(atmospheric_noise * np.fft.fftshift(mag_k**(5/3.)))* atmospheric_noise_level/pix_size

    ## make a 1/f map, along a single direction to illustrate striping 
    oneoverf_noise = 0.
    if (one_over_f_noise_level != 0): 
        ones = np.ones(N)
        inds  = (np.arange(N)+.5 - N/2.) 
        X = np.outer(ones,inds) * pix_size /60. ## angles relative to 1 degrees 
        kx = 2 * np.pi/(X+.01) ## 0.01 is a regularizaiton factor
        oneoverf_noise = np.fft.fft2(np.random.normal(0,1,(N,N)))
        oneoverf_noise = np.fft.ifft2(oneoverf_noise * np.fft.fftshift(kx))* one_over_f_noise_level/pix_size

    ## return the noise map
    noise_map = np.real(white_noise + atmospheric_noise + oneoverf_noise)
    return(noise_map)

#################################################################################################################################################################################################################################################################################################################################################################################################################################

def average_N_spectra(spectra,N_spectra,N_ells):
    avgSpectra = np.zeros(N_ells)
    rmsSpectra = np.zeros(N_ells)
    
    # calcuate the average spectrum
    i = 0
    while (i < N_spectra):
        avgSpectra = avgSpectra + spectra[i,:]
        i = i + 1
    avgSpectra = avgSpectra/(1. * N_spectra)
    
    #calculate the rms of the spectrum
    i =0
    while (i < N_spectra):
        rmsSpectra = rmsSpectra +  (spectra[i,:] - avgSpectra)**2
        i = i + 1
    rmsSpectra = np.sqrt(rmsSpectra/(1. * N_spectra))
    
    return(avgSpectra,rmsSpectra)

#################################################################################################################################################################################################################################################################################################################################################################################################################################


def QU2EB(N,pix_size,Qmap,Umap,):
    '''Calcalute E, B maps given input Stokes Q, U maps'''
    
    # Create 2d Fourier coordinate system.
    ones = np.ones(N)
    inds  = (np.arange(N) - N/2.) /(N-1.)
    kX = np.outer(ones,inds) / (pix_size/60. * np.pi/180.)
    kY = np.transpose(kX)
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

#################################################################################################################################################################################################################################################################################################################################################################################################################################
def Plot_CMB_Map_compact(ax, Map_to_Plot, c_min, c_max, X_width, Y_width):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    im = ax.imshow(Map_to_Plot, interpolation='bilinear', origin='lower', cmap='RdBu_r')
    im.set_clim(c_min, c_max)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    im.set_extent([0, X_width, 0, Y_width])
    ax.set_ylabel('angle $[^\circ]$')
    ax.set_xlabel('angle $[^\circ]$')
    cbar.set_label('temperature [uK]', rotation=270)


#################################################################################################################################################################################################################################################################################################################################################################################################################################


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


def convolve_map_with_beam(N,pix_size,beam,Map):
    "convolves a map with a gaussian beam pattern.  NOTE: pix_size and beam_size_fwhp need to be in the same units" 
    
    # do the convolution
    FT_gaussian = np.fft.fft2(np.fft.fftshift(beam))
    FT_Map = np.fft.fft2(np.fft.fftshift(Map))
    convolved_map = np.fft.fftshift(np.real(np.fft.ifft2(FT_gaussian*FT_Map)))
    
    # return the convolved map
    return(convolved_map)

#################################################################################################################################################################################################################################################################################################################################################################################################################################


def make_2d_gaussian_beam(N,pix_size,beam_size_fwhp):
     # make a 2d coordinate system
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

#################################################################################################################################################################################################################################################################################################################################################################################################################################

def convlolve(A,B): ## the 2d convolution
    fA = np.fft.ifft2(np.fft.fftshift(A))
    fB = np.fft.ifft2(np.fft.fftshift(B))
    convAB = np.fft.fftshift(np.fft.fft2(fA*fB))
    return(np.real(convAB))

#################################################################################################################################################################################################################################################################################################################################################################################################################################

    
def make_2d_beam_cordinates(N,pix_size):
    # make a 2d coordinate system
    ones = np.ones(N)
    inds  = (np.arange(N)+.5 - N/2.) * pix_size
    X = np.outer(ones,inds)
    Y = np.transpose(X)
    R = np.sqrt(X**2. + Y**2.)
    Theta = np.arctan2(Y,X)
    return(X,Y,R,Theta)

#################################################################################################################################################################################################################################################################################################################################################################################################################################


def Plot_beam(beam_to_Plot,c_min,c_max,N,pix_size,plot_width,title):
    print("beam max:",np.max(beam_to_Plot),"beam min:",np.std(beam_to_Plot))
    ## set the range
    N_bins = np.floor(plot_width / pix_size / 2.)
    X_width = N_bins * pix_size * 2.
    Y_width = N_bins * pix_size * 2.
    section_to_Plot = beam_to_Plot[int(N/2 - N_bins):int(N/2 + N_bins),int(N/2 - N_bins):int(N/2 + N_bins)] # HO AGGIUNTO GLI INT
    plt.figure()
    plt.figure(figsize=(4, 4))
    im = plt.imshow(section_to_Plot, interpolation='bilinear', origin='lower',cmap=cm.magma)
    im.set_clim(c_min,c_max)
    cbar = plt.colorbar()
    im.set_extent([0,X_width,0,Y_width])
    plt.ylabel('angle [arcmin]')
    plt.xlabel('angle [arcmin]')
    cbar.set_label('amplitud (arb)', rotation=270)
    plt.title(title)
    plt.show()
    return(0)

#################################################################################################################################################################################################################################################################################################################################################################################################################################

def make_little_buddies(N,pix_size,beam_size_fwhp,bs,main_beam_peak):
    # make a 2d coordinate system
    X,Y,R,Theta = make_2d_beam_cordinates(N,pix_size)
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

#################################################################################################################################################################################################################################################################################################################################################################################################################################

    
def make_ghosting_beam(N,pix_size,beam_size_fwhp,bs,main_beam_peak):
    # make a 2d coordinate system
    X,Y,R,Theta = make_2d_beam_cordinates(N,pix_size)
    ## make shelf
    in_shelf = np.where(R < bs["ghostshelf"]["Diam"])
    shelf = np.zeros(np.shape(R))
    shelf[in_shelf] = 1.
    roll_off_kernal = np.exp(-.5 *(R/bs["ghostshelf"]["roll_off"])**2.)
    shelf = convlolve(shelf,roll_off_kernal)
    shelf = shelf / np.max(shelf) * main_beam_peak * bs["ghostshelf"]["A"]  ## normalized relative to the peak of the main beam
    return(shelf)

#################################################################################################################################################################################################################################################################################################################################################################################################################################

def make_cross_talk_beam_grid(N,pix_size,beam_size_fwhp,bs):
    # make a 2d coordinate system
    X,Y,R,Theta = make_2d_beam_cordinates(N,pix_size)
    ## make cross-talk grid
    hex_grid = np.zeros(np.shape(R)) 
    delta_Nx = np.floor(bs["hex_crostalk"]["grid_space"] / pix_size)
    delta_Nx_nextrow = delta_Nx /2.
    delta_Ny_nextrow = delta_Nx * np.cos(30. * np.pi/180.)
    i = -1* bs["hex_crostalk"]["N"]
    while (i <= bs["hex_crostalk"]["N"]):
        j = -1* bs["hex_crostalk"]["N"]
        while (j <= bs["hex_crostalk"]["N"]):
            i_active = N/2 + i * delta_Nx
            j_active = N/2 + j * delta_Ny_nextrow
            if (i % 2 == 1):
                j_active -= delta_Nx_nextrow  ## offset every other row
            i_active = int(i_active)
            j_active = int(j_active)
            if ((i_active !=N/2) or (j_active !=N/2)):  # exclude the main beam
                distance = np.sqrt(np.abs((i_active - N/2)/ delta_Nx)**2 + np.abs((j_active - N/2)/delta_Nx)**2)
                xtalk_tau = 1./ np.log(bs["hex_crostalk"]["neighbor_exp_fall"])
                distance /= xtalk_tau
                hex_grid[i_active, j_active] = np.exp(distance)
            j+=1
        i+=1
    const = np.ones(np.shape(hex_grid))
    hex_grid *= np.sum(const)   ## normalize the gain so that the x-talk amplitude is defined correclty
    return(hex_grid)


#################################################################################################################################################################################################################################################################################################################################################################################################################################

def make_monopole_dipole_quadrupole(N,pix_size,beam_size_fwhp,bs):
    # make a 2d coordinate system
    X,Y,R,Theta = make_2d_beam_cordinates(N,pix_size)
    # monopole
    beam_sigma = beam_size_fwhp / np.sqrt(8.*np.log(2))
    mono = np.exp(-.5 *(R/beam_sigma)**2.)
    norm = np.sum(mono)
    mono = mono/norm
    # dipoles
    dip_x = mono * R * np.sin(Theta)
    dip_x /= (np.max(dip_x) * norm)
    dip_y = mono * R * np.cos(Theta)
    dip_y /= (np.max(dip_y) * norm)
    # quadrupoles
    quad_x = mono * R * np.sin(2. * Theta)
    quad_x /= (np.max(quad_x) * norm)
    quad_45 = mono * R * np.cos(2. * Theta)
    quad_45 /= (np.max(quad_45) * norm)
    ## return
    return(mono,dip_x,dip_y,quad_x,quad_45)


#################################################################################################################################################################################################################################################################################################################################################################################################################################

def make_systematics_beams(N,pix_size,beam_size_fwhp,bs):
    # intitalize the beam to zero
    B_TT=B_QQ=B_UU=B_QT=B_UT=B_QU=B_UQ=0.
    
    # make a 2d gaussian --- this is the main beam 
    main_beam = make_2d_gaussian_beam(N,pix_size,beam_size_fwhp)
    ## merge this into the beam
    B_TT += main_beam
    B_QQ += main_beam
    B_UU += main_beam
    
    main_beam_peak = np.max(main_beam)
    
    # make the little buddies
    Budy_TT,Budy_QT,Budy_UT = make_little_buddies(N,pix_size,beam_size_fwhp,bs,main_beam_peak)
    ## merge this into the beam
    B_TT += Budy_TT
    B_QT += Budy_QT
    B_UT += Budy_UT

    # make the ghosting shelf
    shelf = make_ghosting_beam(N,pix_size,beam_size_fwhp,bs,main_beam_peak)
    ## merge this into the beam
    B_TT += shelf
    B_QQ += shelf
    B_UU += shelf
    
    # make the cross talk beam
    # make a hex grid centered on the beam
    hex_grid = make_cross_talk_beam_grid(N,pix_size,beam_size_fwhp,bs)
    #convolve the hex grid with the beam
    cross_talk = convlolve(hex_grid,main_beam)
    ## merge this into the beam
    B_TT += cross_talk
    B_QQ += cross_talk
    B_UU += cross_talk
    B_QT += cross_talk
    B_UT += cross_talk
    
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
    B_QT += TtoQ
    B_UT += TtoU
    

    return(B_TT,B_QQ,B_UU,B_QT,B_UT,B_QU,B_UQ)


#################################################################################################################################################################################################################################################################################################################################################################################################################################
def convolve_map_polarized_systmatic_beam(B_TT,B_QQ,B_UU,B_QT,B_UT,B_QU,B_UQ,QMap,UMap,TMap):
    "convolves a map with a gaussian beam pattern.  NOTE: pix_size and beam_size_fwhp need to be in the same units" 
    # do the convolutions   
    sys_map_T = convlolve(B_TT,TMap)
    sys_map_Q = convlolve(B_QQ,QMap) + convlolve(B_QT,TMap) #+ convlolve(B_QU,UMap)    
    sys_map_U = convlolve(B_UU,QMap) + convlolve(B_UT,TMap) #+ convlolve(B_UQ,QMap) 
    # return the convolved map
    return(sys_map_T,sys_map_Q,sys_map_U)

#################################################################################################################################################################################################################################################################################################################################################################################################################################


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
#################################################################################################################################################################################################################################################################################################################################################################################################################################
def convolve_map_with_beam(N,pix_size,beam_x,beam_y, Map):
    
    #FT
    FT_Map    = np.fft.fft2(np.fft.fftshift(Map))
    FT_beam_x = np.fft.fft2(np.fft.fftshift(beam_x))
    FT_beam_y = np.fft.fft2(np.fft.fftshift(beam_y))
    
    convolved_map_x = np.fft.fftshift((np.fft.ifft2(FT_beam_x*FT_Map)))
    convolved_map_y = np.fft.fftshift((np.fft.ifft2(FT_beam_y*FT_Map)))
    
    convolved_map = np.sqrt(convolved_map_x**2 + convolved_map_y**2) # NON SO SE è GIUSTO
    convolved_map = np.real(convolved_map)
    
    return(convolved_map)


#################################################################################################################################################################################################################################################################################################################################################################################################################################
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


#################################################################################################################################################################################################################################################################################################################################################################################################################################
def convolve_map_with_beam(N,pix_size,beam_x,beam_y, Map):
    
    #FT
    FT_Map    = np.fft.fft2(np.fft.fftshift(Map))
    FT_beam_x = np.fft.fft2(np.fft.fftshift(beam_x))
    FT_beam_y = np.fft.fft2(np.fft.fftshift(beam_y))
    
    convolved_map_x = np.fft.fftshift((np.fft.ifft2(FT_beam_x*FT_Map)))
    convolved_map_y = np.fft.fftshift((np.fft.ifft2(FT_beam_y*FT_Map)))
    
    
    combined_map_fourier  = FT_beam_x * FT_Map + FT_beam_y * FT_Map 
    #combined_beam_fourier = np.abs(FT_beam_x)**2 + np.abs(FT_beam_y)**2
    
    
    # REALE
    combined_map_real = np.fft.ifft2(combined_map_fourier)
    combined_map_real = np.real(combined_map_real)

    convolved_map_x_real =  np.fft.ifft2(convolved_map_x)
    convolved_map_y_real =  np.fft.ifft2(convolved_map_y)
    
    convolved_map_x_real = np.real(combined_map_real)
    convolved_map_y_real = np.real(convolved_map_y_real)
    #combined_beam_real = np.fft.ifft2(combined_beam_fourier)
    #combined_beam_real = np.real(combined_beam_real)
    
    
    return(combined_map_real, convolved_map_x_real, convolved_map_y_real)