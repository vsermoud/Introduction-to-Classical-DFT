# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:28:22 2024

@author: vserm
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 15:12:43 2023

@author: vdmsermoud
"""

import numpy as np
from numba import jit
import scipy.integrate as integrate


@jit(nopython=True)
def Phi_wca(dist,rc,eps,Sig_ff):
    rmin = 2.**(1./6.)*Sig_ff
    if dist>rc:
        return 0
    elif (dist>rmin and dist<=rc):
        Temp1 = (2./5.)*(eps*Sig_ff**12)/(dist**10)-eps*Sig_ff**6/(dist**4)
        Temp2 = (2./5.)*(eps*Sig_ff**12)/(rc**10)-eps*Sig_ff**6/(rc**4)
        return  2.*np.pi*(Temp1 - Temp2)
    else:
        Temp0 = 0.5*eps*(dist**2-rmin**2)
        Temp1 = (2./5.)*(eps*Sig_ff**12)/(rmin**10)-eps*Sig_ff**6/(rmin**4)
        Temp2 = (2./5.)*(eps*Sig_ff**12)/(rc**10)-eps*Sig_ff**6/(rc**4)
        return 2.*np.pi*(Temp0 + Temp1 - Temp2)

    
@jit(nopython=True)    
def MF_poid(Z_vec,rc,eps,Sig_ff):
    Phi_att = np.zeros((len(Z_vec),len(Z_vec)))
    for j in range(len(Z_vec)):
        for i in range(len(Z_vec)):
           Phi_att[i,j] = Phi_wca(abs(Z_vec[i]-Z_vec[j]),rc,eps,Sig_ff)
    return Phi_att


@jit(nopython=True)    
def calc_dFAtt(H,MF,rho_vec,Nump,dz,Sigma_ff,eps,Z_vec,rc):
    conv = np.zeros((Nump))
    FAtt = np.zeros((Nump))
    for j in range(Nump):
        Zinf = max(0,Z_vec[j]-rc)
        Zmax = min(H,Z_vec[j]+rc)
        for i in range(Nump):
            if Z_vec[i]==Zinf or Z_vec[i]==Zmax:
                conv[j] += 0.5*MF[i,j]*rho_vec[i]
            else:
                conv[j] += MF[i,j]*rho_vec[i]
    for i in range(Nump):
        FAtt[i] = conv[i]*dz
    return FAtt


@jit(nopython=True)
def calc_FAtt(H,MF,rho_vec,Nump,dz,Sigma_ff,eps,Z_vec,rc):
    rho_mat = np.outer(rho_vec, rho_vec)
    rho2MF = rho_mat*MF
    Fatt = 0.5*np.sum(rho2MF)*dz*dz
    return Fatt



#@jit(nopython=True)    
def calc_dFAtt_F(H,MF,rho_vec,Nump,dz,Sigma_ff,eps,Z_vec,rc):
    FAtt = np.zeros((Nump))
    
    Npad = Nump+int(4.*rc/dz)

    MF_padding = np.zeros((Npad))
    MF_padding[0:Nump] = MF[:,0]
    MF_padding[Npad-Nump:Npad] = MF[:,-2]

    #Paddin sem condição de contorno periódica
    Rho_padding = np.zeros((Nump+int(4*rc/dz)))#+rho_H
    Rho_padding[int(2*rc/dz)-1:int(2*rc/dz)-1+Nump] = rho_vec[0:Nump]

    rho_fft = np.fft.fftn(Rho_padding)
    MF_fft = np.fft.fftn(MF_padding)
    # Perform element-wise multiplication in the frequency domain
    result_fft = rho_fft * MF_fft

    # Perform inverse FFT to obtain the convolution in the spatial domain
    result_convolution = np.fft.ifftn(result_fft)

    # Get the real part of the convolution (imaginary part should be close to zero)
    A_padding =  np.real(result_convolution)*dz
    
    FAtt[:] = A_padding[int(2*rc/dz)-1:int(2*rc/dz)-1+Nump]

    return FAtt
