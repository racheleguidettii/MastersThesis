import numpy as np
import matplotlib
import sys, platform, os
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import camb
from scipy.interpolate import interp1d

from camb import model, initialpower
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interactive, interact, fixed
from scipy.signal import convolve

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook  # Use an alias to avoid conflicts

from PolAngle import *
from beam import * 
from map import *
from convolution import * 
'''
class BeamSystematics():
    #def __init__(self, N, pix_size, beam_size_fwhp, beam, bs):
        #self.N = N
        #self.pix_size = pix_size
        #self.beam_size_fwhp = beam_size_fwhp
        #self.beam = beam
        #self.bs = bs
    def __init__(self, N, pix_size, beam_size_fwhp, beam, bs):
        self.N = N
        self.pix_size = pix_size
        self.beam_size_fwhp = beam_size_fwhp
        self.beam = beam
        self.bs = bs
    
    def make_little_buddies(self):
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
    
    def make_ghosting_beam(self):
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
    
    def make_cross_talk_beam_grid(self):
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
    
    def convlolve(sef): ## the 2d convolution
        fA = np.fft.ifft2(np.fft.fftshift(A))
        fB = np.fft.ifft2(np.fft.fftshift(B))
        convAB = np.fft.fftshift(np.fft.fft2(fA*fB))
        return(np.real(convAB))
    
    #def make_monopole_dipole_quadrupole(self):
    #    # Implementazione della funzione make_monopole_dipole_quadrupole
    #    pass

    def make_systematics_beams(self):
    
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

#self, N, pix_size, beam_size_fwhp, beam, bs
# Utilizzo della classe BeamSystematics


'''
class BeamSystematics:
    #def __init__(self, N, pix_size, beam_size_fwhp, beam, bs):
        #self.N = N
        #self.pix_size = pix_size
        #self.beam_size_fwhp = beam_size_fwhp
        #self.beam = beam
        #self.bs = bs
    def __init__(self):
        pass
    
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
    
    def convlolve(A,B): ## the 2d convolution
        fA = np.fft.ifft2(np.fft.fftshift(A))
        fB = np.fft.ifft2(np.fft.fftshift(B))
        convAB = np.fft.fftshift(np.fft.fft2(fA*fB))
        return(np.real(convAB))
    
    #def make_monopole_dipole_quadrupole(self):
    #    # Implementazione della funzione make_monopole_dipole_quadrupole
    #    pass

    def make_systematics_beams(self,N,pix_size,beam_size_fwhp, beam,bs):
    
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
