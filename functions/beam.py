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


def Plot_beam(beam_to_Plot,c_min,c_max,N,pix_size,plot_width,title, axis):
    print("beam max:",np.max(beam_to_Plot),"beam min:",np.std(beam_to_Plot))
    ## set the range
    N_bins = np.floor(plot_width / pix_size / 2.)
    X_width = N_bins * pix_size * 2.
    Y_width = N_bins * pix_size * 2.
    section_to_Plot = beam_to_Plot[int(N/2 - N_bins):int(N/2 + N_bins),int(N/2 - N_bins):int(N/2 + N_bins)] # HO AGGIUNTO GLI INT
    plt.figure(figsize=(4, 4))
    im = plt.imshow(section_to_Plot, interpolation='bilinear', origin='lower',cmap=cm.magma)
    im.set_clim(c_min,c_max)
    cbar = plt.colorbar()
    im.set_extent([0,X_width,0,Y_width])
    plt.ylabel(axis)
    plt.xlabel(axis)
    cbar.set_label('amplitud (arb)', rotation=270)
    plt.title(title)
    #plt.show()
    return(0)


#################################################################################################################################################################################################################################################################################################################################################################################################################################


def create_beam_secpeaks(pix_size, FWHMx, FWHMy, theta, array_dB, r, r1, X, Y, a, ellipticity):
    ''' FWHMx, FWHMy = FWHM of x,y beams
        theta = angle of rotation of the beam (degrees)
        array_dB = array of max values of the secondary peaks
        r = array (same dim as array_dB) of the ANGULAR distance from the center
        r1 = width of the rings (sec peaks)
        X, Y = coordinates
        a = major axis for the elliptical rings
        ellipticity
    '''
    # from degrees to distance in pixels
    r1 = r1/pix_size 

    # if theta is not zero, we rotate the coordinates
    X_rotated = X * np.cos(np.radians(theta)) - Y * np.sin(np.radians(theta))
    Y_rotated = X * np.sin(np.radians(theta)) + Y * np.cos(np.radians(theta))

    X = X_rotated
    Y = Y_rotated

    # MAIN BEAM ###################################################################
    wx = FWHMx / np.sqrt(8 * np.log(2))
    wy = FWHMy / np.sqrt(8 * np.log(2))

    beam_x = np.exp(-2 * (X**2 / wx**2 + Y**2 / wy**2))
    beam_y = np.exp(-2 * (Y**2 / wx**2 + X**2 / wy**2))


    # SECONDARY RINGS ############################################################
    sec_rings_x = np.zeros_like(X)
    sec_rings_y = np.zeros_like(Y)

    for i in range(len(array_dB)):
        # Correction of ellipticity values or the secondary rings turn out flattened. If we want a circle (ell = 1) the correction is not valid
        if (ellipticity ==1):
            a = a
            b = a * (ellipticity)
        else:
            a = a
            b = a * (ellipticity-0.5)

        distance_x = np.sqrt((X / b)**2 + (Y / a)**2)  # ellisse x
        distance_y = np.sqrt((X / a)**2 + (Y / b)**2)  # ellisse y
        
        normalized_distance_x = (distance_x - r1[i]) / r
        normalized_distance_y = (distance_y - r1[i]) / r
        
        max_value = 10**(array_dB[i] / 10)
        
        gaussian_distribution_x = np.exp(-(normalized_distance_x)**2 / 0.8)
        gaussian_distribution_y = np.exp(-(normalized_distance_y)**2 / 0.8)
        
        sec_rings_x += max_value * gaussian_distribution_x
        sec_rings_y += max_value * gaussian_distribution_y

    # TOT BEAM ###################################################################
    beam_x_real = beam_x + sec_rings_x
    beam_y_real = beam_y + sec_rings_y

    # NORMALIZATION #############################################################
    beam_x_real /= np.sum(beam_x_real)
    beam_y_real /= np.sum(beam_y_real)

    return beam_x, beam_y, sec_rings_x, sec_rings_y, beam_x_real, beam_y_real
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

# FROM THE FUNCTIONS make_little_buddies, make_ghosting_beam and make_cross_talk_beam_grid WE ADD SYSTEMATICS TO THE BEAM
def make_systematics_beams(N,pix_size,beam_size_fwhp, beam,bs):
    
    # intitalize the beam to zero
    B_TT=B_QQ=B_UU=B_QT=B_UT=B_QU=B_UQ=0.
    
    ## merge this into the beam
    B_TT += beam
    B_QQ += beam
    B_UU += beam
    
    beam_peak = np.max(beam)
    
    # make the little buddies
    Budy_TT,Budy_QT,Budy_UT = make_little_buddies(N,pix_size,beam_size_fwhp,bs,beam_peak)
    ## merge this into the beam
    B_TT += Budy_TT
    B_QT += Budy_QT
    B_UT += Budy_UT

    # make the ghosting shelf
    shelf = make_ghosting_beam(N,pix_size,beam_size_fwhp,bs,beam_peak)
    ## merge this into the beam
    B_TT += shelf
    B_QQ += shelf
    B_UU += shelf
    
    # make the cross talk beam
    # make a hex grid centered on the beam
    hex_grid = make_cross_talk_beam_grid(N,pix_size,beam_size_fwhp,bs)
    #convolve the hex grid with the beam
    cross_talk = convlolve(hex_grid,beam)
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
    #B_QT += TtoQ
    #B_UT += TtoU
    
    
    return(B_TT,B_QQ,B_UU,B_QT,B_UT,B_QU,B_UQ)  



def convlolve(A,B): ## the 2d convolution
    fA = np.fft.ifft2(np.fft.fftshift(A))
    fB = np.fft.ifft2(np.fft.fftshift(B))
    convAB = np.fft.fftshift(np.fft.fft2(fA*fB))
    return(np.real(convAB))