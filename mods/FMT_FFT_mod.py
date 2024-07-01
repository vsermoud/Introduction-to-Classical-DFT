# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 09:54:24 2024

@author: vserm
"""

import numpy as np
from scipy.fft import fft, ifft


def W_FMT_calc(H,Nump,dhs):
        
    kf = np.fft.fftfreq(Nump,d=H/(Nump-1))
    rhs = 0.5*dhs
    w_fmt = np.zeros((Nump,6),dtype=np.complex128)
        
    for i in range(Nump):
        if (i>0):       
            w_fmt[i,0] = 2.0*rhs*np.sin(2.0*np.pi*rhs*kf[i])/(kf[i]*4.*np.pi*rhs**2)
            w_fmt[i,1] = 2.0*rhs*np.sin(2.0*np.pi*rhs*kf[i])/(kf[i]*4.*np.pi*rhs)
            w_fmt[i,2] = 2.0*rhs*np.sin(2.0*np.pi*rhs*kf[i])/kf[i]
            w_fmt[i,3] = (np.sin(2.0*rhs*kf[i]*np.pi)-
                        2.0*rhs*np.pi*kf[i]*np.cos(2.0*rhs*kf[i]*np.pi))/(2.*kf[i]**3*np.pi**2)
            w_fmt[i,4] = -(-2j*rhs*kf[i]*np.pi*np.cos(2*rhs*kf[i]*np.pi) +
                          1j*np.sin(2*rhs*kf[i]*np.pi))/(np.pi*kf[i]**2)/(4.*np.pi*rhs)
            w_fmt[i,5] = -(-2j*rhs*kf[i]*np.pi*np.cos(2*rhs*kf[i]*np.pi) +
                          1j*np.sin(2*rhs*kf[i]*np.pi))/(np.pi*kf[i]**2)
        else:
            w_fmt[i,0] = 1.0
            w_fmt[i,1] = rhs
            w_fmt[i,2] = 4.0*rhs**2*np.pi
            w_fmt[i,3] = 4.0*np.pi*rhs**3/3.0
            w_fmt[i,4] = 0.0
            w_fmt[i,5] = 0.0
                
    return w_fmt


def calc_N(w_fmt, rho, Nump):
    rho_fft = fft(rho) 
        
    N_fft = np.zeros((Nump, 6), dtype=np.complex128)
    N_real = np.zeros((Nump, 6))
    
    for i in range(6):
        N_fft[:, i] = w_fmt[:, i]*rho_fft
        
    for i in range(6):
        N_real[:, i] = ifft(N_fft[:, i]).real
          
    return N_real


def d_Phi(H,w_fmt,Nump,rho_vec):
    N_fmt = calc_N(w_fmt,rho_vec,Nump)
    dPhi = np.zeros((Nump,6))
    for i in range(Nump):
        if N_fmt[i,3]> 1.e-8:
            Temp = 1.-N_fmt[i,3]
            dPhi[i,0] = -np.log(Temp)
            dPhi[i,1] = N_fmt[i,2]/Temp
            dPhi[i,2] = (N_fmt[i,1]/Temp)+(np.log(Temp)/N_fmt[i,3]+
                        1.0/Temp**2)*(N_fmt[i,2]**2-
                        N_fmt[i,5]**2)/(12.0*np.pi*N_fmt[i,3])
            dPhi[i,3] = N_fmt[i,0]/Temp+(N_fmt[i,1]*N_fmt[i,2]-N_fmt[i,4]*
                    N_fmt[i,5])/Temp**2-(np.log(Temp)/(18.0*np.pi*
                    N_fmt[i,3]**3)+1.0/(36.0*np.pi*N_fmt[i,3]**2*Temp)+
                    (1.0-3.0*N_fmt[i,3])/(36.0*np.pi*N_fmt[i,3]**2*
                    Temp**3))*(N_fmt[i,2]**3-3.0*N_fmt[i,2]*N_fmt[i,5]**2)
            dPhi[i,4] = -N_fmt[i,5]/Temp
            dPhi[i,5] = -N_fmt[i,4]/Temp-(np.log(Temp)/N_fmt[i,3]+1.0
                    /Temp**2)*N_fmt[i,2]*N_fmt[i,5]/(6.0*np.pi*N_fmt[i,3])
    return dPhi



def calc_dFHS(H,w_fmt,rho_vec,Nump):
    dPhi = d_Phi(H,w_fmt,Nump,rho_vec)
    
    dPhi_fft = np.zeros((Nump,6),dtype=complex)
    dFHS_fft = np.zeros((Nump,6),dtype=complex)
    dFHS_real = np.zeros((Nump,6))
    

    for i in range(6):
        dPhi_fft[:,i] = fft(dPhi[:,i])
        if i == 4 or i == 5:
            w_fmt[:,i] *= -1
        dFHS_fft[:,i] = w_fmt[:,i]*dPhi_fft[:,i]
        dFHS_real[:,i] =  np.real(np.fft.ifftn(dFHS_fft[:,i]))
        
    dFHS = np.zeros((Nump))
    
    for i in range(Nump):    
        dFHS[i] = np.sum(dFHS_real[i,:])
    
    return dFHS




