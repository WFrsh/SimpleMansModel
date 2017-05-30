#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit
import h5py
from joblib import Parallel, delayed
import multiprocessing
from numba import jit


class _Pulse(object):
    """Class for generation of pulses"""

    def carrier_au(t,f,phi, b, chirp):
        """ create a carrier wave of the pulse
            input:  t:  time array over which the carrier wave will be calcuated
                    f:  frequency of the el-m wave in 1/au
                    phi: phase of the carrier light field
                    b:  time offset of zero/ center of pulse in au
                    chirp:  chirp of pulse
            output: carrier wave"""
        return np.cos(f*(2*np.pi)*(t-b) + phi + chirp*(t-b)**2)

    def envelope_au(t,a,b,width):
        """define a gaussian as an envelope for the laser pulse
            input: t: time
                   a: heigth of envelope
                   b: t-position
                   width: width
            output: gaussian function"""
        return a*np.exp(-(t-b)**2/(2*width**2))

    def pulse_au(t, FWHM, wavelength, intensity, CEP, b, chirp):
        """ create the electric field of a laser pulse with the given parameters
            input:  t:      time array in au
                    FWHM:   temporal FWHM duration of pulse in fs
                    wavelength: wavelength of pulse in nm
                    intensity: intensity in W/cm^2
                    CEP:    CEP in pi
            output: array of electric field shape of the pulse
                  """
        # calculate the proper output parameters from the input parameters
        FWHM = FWHM*41.35 # convert FWHM from fs to au (41.35 au/fs)
        width = FWHM/(2*np.sqrt(2*np.log(2))) # relation of FWHM to with of gaussian
        wavelength = wavelength*1E-9 # wavelength from nm to m
        f = 3E8/wavelength/4.13E16  # convert wavelength to frequency in 1/au
        a = intensity/3.51E16 # convert intensity to au
        a = np.sqrt(a) # get electric field strength from intensity sqrt(2a!)
        phi = CEP*np.pi
        b = b*41.35 # convert offset from fs to au
        return _Pulse.carrier_au(t,f,phi,b, chirp)*_Pulse.envelope_au(t,a,b,width)

    def red_waist(z,w0=11.382336229148079,z0=15.10687668789063,zR=0.8364377171114098):
        """ input:  z:          z-position along propagation direction
                    w0, z0, zR: parameters for the measured focus of the red beam
            output: radius of the beam at that z position"""
        return w0*np.sqrt(1+((z-z0)/zR)**2)

    def blue_waist(z,w0=5.570579596490828,z0=15.948960104263824,zR=0.39869374130627544):
        """ input:  z:          z-position along propagation direction
                    w0, z0, zR: parameters for the measured focus of the blue beam
            output: radius of the beam at that z position"""
        return w0*np.sqrt(1+((z-z0)/zR)**2)

    tau = 50E-15/np.sqrt(3) # measured pulse length of our pulses

    def intensities_z(position,tau):
        """ input:  position:   z-position along propagation direction
                    tau:        measured pulse length
            output: Ir: Intensity of the red beam at that position
                    Ib: Intensity of the blue beam at that position"""
        rr = _Pulse.red_waist(position)
        rb = _Pulse.blue_waist(position)
        Ir = (20E-6)/(tau*np.pi*(rr*10**-6)**2)/10000
        Ib = (20E-6)/(tau*np.pi*(rb*10**-6)**2)/10000
        return Ir,Ib

@jit # numba speeds it up immensly. Total lifesaver!
def p_verlet(E,t0,t1,dt):
    '''velocity verlet algorithm
    input:  E   : electric field
            t0  : start of integration
            t1  : end of integration
            dt  : time step
    output: p_t   : final momentum'''
    x_t = 0 # init position
    v_t = 0 # init velocity
    rescattered_t = 0 # not rescattered
    x_ = np.zeros(timesteps)
    for i in range(t0,t1):
        # velocity verlet integration
        x_t = x_t + dt*v_t + dt*dt/2/m*e*E[i]
        v_t = v_t + dt/2/m*e*(E[i+1] + E[i])
        # check weather electron exits in the positive or negative direction
        if i == t0:
            if x_t > 0:
                x_positive = True
            else:
                x_positive = False
        # rescattering for positive positions
        if x_positive == True:
            if x_t < 0:
                v_t = -v_t
                rescattered_t += 1
        # rescattering for negative positions
        elif x_positive == False:
            if x_t > 0:
                v_t = -v_t
                rescattered_t += 1
        else:
            print('catch')
        x_[i] = x_t
    p_t = m*v_t
    return p_t, rescattered_t

def p_SMM(E,t0,t1):
    '''calculated momentum via Newtons eqn of motion
    input: E: electric field
           t0: birth time
           t1: final time
           v0: initial velocity'''
    p_t = e*integrate.trapz(E[t0:t1],t[t0:t1])
    return p_t

m = 1
e = -1
timesteps = 10000
t, dt = np.linspace(-110*20,110*20,timesteps,retstep=True) # time in fs 2050au = 50fs

# create the pulse
Er = _Pulse.pulse_au(t, 35, 800, 1.17E14, 0, 0, 0)
Euv = _Pulse.pulse_au(t, 40, 266, 1.85E13, 0, 0, 0)
Ecombined_i = Er + Euv

p_final = np.zeros((timesteps))
rescattered = np.zeros((timesteps))
p_final_SMM = np.zeros((timesteps))
# for phi in range(phisteps):
for i in range(timesteps):
    print(i)
    p_final[i], rescattered[i] = p_verlet(Ecombined_i,i,timesteps-1,dt)
    p_final_SMM[i] = p_SMM(Ecombined_i,i,timesteps-1)

# p_t, x_ = p_verlet(Ecombined_i,500,timesteps,dt)
# plt.plot(t,x_)

# 1 Up = 0.218 a.u. (800nm, 1E14W/cm2)
E_final = p_final*p_final/2
# plt.plot(t, E_final, '+')
# hist = np.histogram(E_final, bins=50)
# plt.semilogy(hist[1][:-1], hist[0])
# plt.xlabel('Energy/a.u.')
# plt.ylabel('yield/arb.units')

ax1 = plt.axes()
ax1.plot(t,Ecombined_i,'b',lw=2)
plt.xlabel('time/a.u.')
plt.ylabel('electric field/a.u.(blue)')
ax1.set_xlim(-110,110)
ax2 = plt.twinx(ax1)
# ax2.plot(t,p_final_SMM,'g')
ax2.plot(t,p_final,'r')
ax2.set_xlim(-110,110)
plt.ylabel('final energy/a.u.(red)')
# p_direct = p_final.copy()
# p_direct[rescattered != 0] = np.nan
# ax2.plot(t,p_direct, 'red')
# p_rescattered = p_final.copy()
# p_rescattered[rescattered != 1] = np.nan
# ax2.plot(t,p_rescattered, 'orange')
# ax2.set_ylim(-2,2)

# plt.figure()
# plt.hist(p_final, bins=50, alpha=.5, label='0pi')
# plt.legend()

plt.show()
