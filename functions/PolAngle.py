import numpy as np
import matplotlib
import sys
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from beam import * 
from map import *
from convolution import *

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