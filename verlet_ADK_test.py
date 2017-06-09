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

class _Avg(object):
    """docstring for _Avg"""

    def gaussian(x,I,sigma):
        return I*np.exp(-x**2/sigma**2)

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

def number_of_atoms(timesteps,rate):
    N0p = np.ones(timesteps) # number neutral atoms
    # calculate the number of atoms remaining based on the rate
    for j in range(timesteps-1):
        if N0p[j] < 1e-5: # check if occupation is already zero (does some crazy shit otherwise)
            N0p[j+1] = 0 # keep it zero
        else:  # otherwise, do something
            N0p[j+1] = N0p[j] - N0p[j]*rate[j]# *dt # change population of neutral atoms
    return N0p

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

def plot_(axes, phi,t,Ecombined_i,rate_0pi,p_final_0pi,N0p):
    '''just a funtion to plot it'''
    axes.plot(t,Ecombined_i,'b',lw=2)
    axes.fill(t,rate_0pi/np.max(rate_0pi)*0.05*np.sign(p_final_0pi),'k',alpha=.2)
    axes.set_title(r'{0}$\pi$'.format(phi), y=1.05)
    axes.set_xlabel('time/a.u.')
    axes.set_ylabel('electric field/a.u.(blue)')
    axes.set_xlim(-110,110)
    ax2 = plt.twinx(axes)
    # ax2.plot(t,p_final_SMM,'g')
    ax2.plot(t,p_final_0pi,'r')
    ax2.set_xlim(-110,110)
    ax2.set_ylabel('final momentum/a.u.(red)')
    ax3 = plt.twiny(ax2)
    plt.hist(p_final_0pi, bins=50, weights=rate_0pi*N0p, color='r', alpha=.5, label='0pi',
                            orientation='horizontal', fill=True, histtype='bar') #
    ax3.invert_xaxis()
    ax3.set_xlim(0.01,0)

def simulate_(phi,t,Ir,Iuv,x_uv):
    '''just a function to simulate it'''
    p_final_0pi = np.zeros((timesteps))
    # create the pulse
    Ir_x = _Avg.gaussian(0,Ir,40)
    Iuv_x = _Avg.gaussian(x_uv,Iuv,35)
    Er = _Pulse.pulse_au(t, 35, 800, Ir_x, 0, 0, 0)
    Euv = _Pulse.pulse_au(t, 40, 266, Iuv_x, phi, 0, 0)
    Ecombined_i = Er + Euv
    rate_0pi = ADK(2.44, 0.579, 1., Ecombined_i, 1, 9.)*dt #Argon
    N0p = number_of_atoms(timesteps,rate_0pi)

    for i in range(timesteps):
        if i % 100 == 0:
            print(i)
        p_final_0pi[i] = p_verlet(Ecombined_i,i,timesteps-1,dt,rescatter_prob)
        # p_final_0pi[i] = p_SMM(Ecombined_i,i,timesteps-1)
    return Ecombined_i, rate_0pi, p_final_0pi, N0p

def focus_avg(nI):
    '''focus average; do simulation for different intensities
    create weigths for the histogram by the volume of that intensity
    DOESNT WORK YET'''
    I_disc, dI = np.linspace(0.001,1,nI,retstep=True)
    hist = np.zeros((nI,50))
    x_I = np.zeros(nI)
    # create a gaussian
    x = np.linspace(-4,4,10000)
    gauss = _Avg.gaussian(x,1,1) # create a gaussian
    for j in range(nI):
        # simulate
        outputs = simulate_(phi=0, t=t, Ir=I_disc[j]*1E14, Iuv=0, x_uv=0)
        # calculate the volume where the intensity is within a certain intensity interval
        for i in range(nI):
            x_i = np.argmax(gauss >= I_disc[i]) # get the x value where the intensity is at I_i
            x_I[i] = x[x_i]
        x_I[-1] = 0 # Volume of max intensity is zero
        # calculate the gradient as a weight for the focus averaging
        dVdI = np.gradient(x_I**2)
        avg_weights = np.abs(dVdI[i])*dI
        E_final = outputs[2]*outputs[2]/2
        # print((np.histogram(E_final, bins=np.linspace(0, 3, num=50), weights=avg_weights*outputs[2]*outputs[3])[0]).shape)
        hist[j,:] = np.histogram(E_final, bins=np.linspace(0,2.5,51), weights=avg_weights*outputs[1]*outputs[3])[0]
    return hist

m = 1836*20
e = +1
timesteps = 10000
rescatter_prob = 0
t, dt = np.linspace(-4100,4100,timesteps,retstep=True) # time in fs 2050au = 50fs
phases = np.linspace(0,2,6,endpoint=False)

# plot focus average
# E = focus_avg(10)
# [plt.semilogy(np.linspace(0,2.5,50),E[i,:]) for i in range(10)]

Ecombined_i, rate_0pi, p_final_0pi, N0p = simulate_(1,t,Ir=1E14, Iuv=1E14, x_uv=0)
# Ecombined_i, rate_50pi, p_final_50pi, N50p = simulate_(1,t,Ir=1.17E14, Iuv=1.85E13, x_uv=50)

# plt.figure()
# plt.plot(t,Ecombined_i)

# # 1 Up = 0.218 a.u. (800nm, 1E14W/cm2)
# E_final = p_final_0pi*p_final_0pi/2
# plt.figure()
# hist = np.histogram(E_final, bins=50)
# plt.semilogy(hist[1][:-1], hist[0])
# plt.xlabel('Energy/a.u.')
# plt.ylabel('yield/arb.units')

# histogram of the two distributions
plt.figure()
plt.hist(p_final_0pi, bins=50, weights=rate_0pi*N0p, alpha=.5, label='0pi', orientation='horizontal', fill=False, histtype='step') #
# plt.hist(p_final_50pi, bins=50, weights=rate_50pi*N50p, alpha=.5, label='0pi', orientation='horizontal', fill=False, histtype='step') #
# plt.legend()

# plot multiple axes in one figure
# fig, axs = plt.subplots(2,3)
# fig.subplots_adjust(hspace = .5, wspace=.5)
# axs = axs.ravel()
#
# for i in range(6):
#     print(phases[i])
#     Ecombined_i, rate_0pi, p_final_0pi, N0p = simulate_(phases[i],t)
#     plot_(axs[i],phases[i],t,Ecombined_i,rate_0pi,p_final_0pi,N0p)

plt.show()
