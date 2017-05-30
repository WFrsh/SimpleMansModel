#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit
import h5py
from joblib import Parallel, delayed
import multiprocessing
from numba import jit

np.random.seed(10051990)


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

def ADK(Cl, I_p, Z_c, F, l, alpha):
    """ ADK rate for ionization adapted from eq (2) of Tong et Lin, 2005, Journal of Physics B, 38, 15, 2593-2600
        with the normalization factor for noble gases as explained in Zhao et Brabec, 2006, arXiv:physics/0605049v1
        input:  Cl:     C_l parameter
                I_p:    Ionization potential in au
                Z_c:    populated charge state
                F:      Field amplitude in au
                l:      angular quantum number
                alpha:  empirical fittinng factor
        output: rate of ionization"""
    # get all necessary variables for the cumputation of the ADK rate
    arraym = np.linspace(-l,+l,num=(2*l+1),endpoint=True)
    kappa = np.sqrt(2*I_p)
    F = np.abs(F) # use field strength instead of the electric field
    w_ADK = 0
    for m in arraym: # loop over all magnetic quantum numbers and sum the ADK rate over all
        first_term = Cl**2/(2**np.abs(m)*np.math.factorial(np.abs(m)))
        second_term = (2*l+1)*np.math.factorial(l+np.abs(m))/(2*np.math.factorial(l-np.abs(m)))
        third_term = 1/(kappa**(2*Z_c/kappa-1))
        fourth_term = (2*kappa**3/F)**(2*Z_c/kappa-np.abs(m)-1)
        fifth_term = np.exp(-2*kappa**3/(3*F))
        sixth_term = np.exp(-alpha*(Z_c**2/I_p)*(F/kappa**3))
        w_ADK += first_term*second_term*third_term*fourth_term*fifth_term*sixth_term
    w_ADK = w_ADK/(2*l+1)
    return w_ADK

@jit # numba speeds it up immensly. Total lifesaver!
def p_verlet(E,t0,t1,dt,rescatter_prob):
    '''velocity verlet algorithm
    input:  E   : electric field
            t0  : start of integration
            t1  : end of integration
            dt  : time step
            rescatter_prob: rescattering probability
    output: p_t   : final momentum'''
    x_t = 0 # init position
    v_t = 0 # init velocity
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
            if x_t < 0 and np.random.rand() < rescatter_prob:
                v_t = -v_t
        # rescattering for negative positions
        elif x_positive == False:
            if x_t > 0 and np.random.rand() < rescatter_prob:
                v_t = -v_t
        else:
            print('catch')
        x_[i] = x_t
    p_t = m*v_t
    return p_t

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
rescatter_prob = 1
t, dt = np.linspace(-2050,2050,timesteps,retstep=True) # time in fs 2050au = 50fs

# create the pulse
Er = _Pulse.pulse_au(t, 35, 800, 1.17E14, 0, 0, 0)
Euv = _Pulse.pulse_au(t, 40, 266, 1.85E13, 1.8, 0, 0)
Ecombined_i = Er + Euv
rate_0pi = ADK(2.44, 0.579, 1., Ecombined_i, 1, 9.)*dt #Argon

p_final_0pi = np.zeros((timesteps))
# for phi in range(phisteps):
for i in range(timesteps):
    print(i)
    p_final_0pi[i] = p_verlet(Ecombined_i,i,timesteps-1,dt,rescatter_prob)
    # p_final_0pi[i] = p_SMM(Ecombined_i,i,timesteps-1)

# p_t, x_ = p_verlet(Ecombined_i,500,timesteps,dt)
# plt.plot(t,x_)

# 1 Up = 0.218 a.u. (800nm, 1E14W/cm2)
E_final_0pi = p_final_0pi*p_final_0pi/2

# # 1 Up = 0.218 a.u. (800nm, 1E14W/cm2)
# E_final = p_final_0pi*p_final_0pi/2
# # plt.plot(t, E_final, '+')
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
ax2.plot(t,p_final_0pi,'r')
ax2.set_xlim(-110,110)
plt.ylabel('final momentum/a.u.(red)')
ax3 = plt.twiny(ax2)
plt.hist(p_final_0pi, bins=50, weights=rate_0pi, color='r', alpha=.5, label='0pi',
                        orientation='horizontal', fill=True, histtype='bar') #
ax3.invert_xaxis()
ax3.set_xlim(0.02,0)

# histogram of the two distributions
# plt.figure()
# plt.hist(p_final_0pi, bins=50, weights=rate_0pi, alpha=.5, label='0pi', orientation='horizontal', fill=False, histtype='step') #
# plt.legend()

plt.show()
