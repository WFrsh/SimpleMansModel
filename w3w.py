import numpy as np
from scipy import integrate
from scipy.optimize import curve_fit
import h5py
from joblib import Parallel, delayed
import multiprocessing


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

class _Ionization(object):
    """docstring for _Ionization"""

    def ADK(Cl, I_p, Z_c, F, l, alpha):
        """ ADK rate for ionization adapted from eq (2) of Tong et Lin, 2005, Journal of Physics B, 38, 15, 2593-2600
        	with the normalization factor for noble gases as explained in Zhao et Brabec, 2006, arXiv:physics/0605049v1
            input:  Cl:     C_l parameter
                    I_p:    Ionization potential in au
                    Z_c:    populated charge state
                    F:      Field amplitude in au
                    l:      angular quantum number
            output: rate of ionization"""
        # get all necessary variables for the cumputation of the ADK rate
        arraym = np.linspace(-l,+l,num=(2*l+1),endpoint=True)
        kappa = np.sqrt(2*I_p)
        w_ADK = 0
        for m in arraym: # loop over all magnetic quantum numbers and sum the ADK rate over all
            first_term = Cl**2/(2**np.abs(m)*np.math.factorial(np.abs(m)))
            second_term = (2*l+1)*np.math.factorial(l+np.abs(m))/(2*np.math.factorial(l-np.abs(m)))
            third_term = 1/(kappa**(2*Z_c/kappa-1))
            fourth_term = (2*kappa**3/np.abs(F))**(2*Z_c/kappa-np.abs(m)-1)
            fifth_term = np.exp(-2*kappa**3/(3*np.abs(F)))
            sixth_term = np.exp(-alpha*(Z_c**2/I_p)*(np.abs(F)/kappa**3))
            w_ADK += first_term*second_term*third_term*fourth_term*fifth_term*sixth_term
        w_ADK = w_ADK/(2*l+1)
        return w_ADK

    def p_SMM(E,t0,t1):
        '''calculated momentum via Newtons eqn of motion
        input: E: electric field
               t0: birth time
               t1: final time
               v0: initial velocity'''
        t0 = np.where(t == t0)[0][0]
        t1 = np.where(t == t1)[0][0]
        p_t = e*integrate.trapz(E[t0:t1],t[t0:t1])
        return p_t

    def p_bins(pmax,npbins):
        """get the bins"""
        # if i == 0:
        #     binmax = np.max(np.abs(p_SMM_t_i))
        pbins = np.linspace(-pmax,pmax,npbins+1)
        return pbins

    def final_distribution_intensity_atoms(Imax,nI,pbins,npbins, phi,timesteps):
        '''Calculate the momentum distribution
        input: Imax
               nI
               pbins
               npbins
               phi
               timesteps
        output: dist_i_3: momentum distribution [50]'''
        # initializing of all the arrays
        dist_i_3 = np.zeros([nI,npbins]) # momentum distributions
        p_final = np.zeros([nI,timesteps]) # final momenta
        intensities = [] # intensities
        tmax = timesteps # number of timesteps
        R = np.zeros((nI,tmax)) # rates
        N0p = np.ones((nI,tmax)) # neutrals
        N1p = np.zeros((nI,tmax)) # single ionized

        I = np.linspace(0.001,Imax,nI) # intensities from 0 to 10*max intensity
        # loop over different peak intensity
        for i in range(nI):
            # print('{0} of {1} and phi={2:.3f} of max 2 at ratio {3}'.format(i+1,nI,phi, ratio))
            # create the pulse
            Er = _Pulse.pulse_au(t, 30, 800, 1E14, 0, 0, 0)
            # Eb = _Pulse.pulse_au(t, 30, 400, ratio*1E14, phi, 0, 0)
            Euv = _Pulse.pulse_au(t, 30, 266, .037*1E14, phi, 0, 0)
            Ecombined_i = np.sqrt(I[-i-1])*(Er + Euv)
            intensities.append(np.max(.5*Ecombined_i**2)*3.51E16) # in W/cm^2

            # calculate the ionization rate
            rate = _Ionization.ADK(2.19, 0.579, 1., Ecombined_i, 1, 9.)*dt # constants for the ionization of Argon
            R[i,:] = rate

            # calculate the number of atoms remaining based on the rate
            for j in range(tmax-1):
                if N0p[i,j] < 1e-5: # check if occupation is already zero (does some crazy shit otherwise)
                    N0p[i,j+1] = 0 # keep it zero
                    N1p[i,j+1] = 1 # keep it one
                else:  # otherwise, do something
                    N0p[i,j+1] = N0p[i,j] - N0p[i,j]*rate[j]# *dt # change population of neutral atoms
                    N1p[i,j+1] = 1-N0p[i,j+1] # change population of ionized atoms

            # calculate the final positions
            p_SMM_t_i = []
            for t0 in t[:tmax]:
                p = _Ionization.p_SMM(E=Ecombined_i, t0=t0, t1=t[-1])
                p_SMM_t_i.append(p)
            p_final[i,:] = p_SMM_t_i

            # get the momentum histogram
            # weighted with the rate and the number of atoms
            dist_i_3[i,:] = np.histogram(p_SMM_t_i,bins=pbins,weights=rate*N0p[i,:])[0]

        return dist_i_3, intensities, N0p, N1p, R, Ecombined_i, p_final

class _Avg(object):
    """docstring for _Avg"""

    def gaussian(x,I,sigma):
        return I*np.exp(-x**2/sigma**2)

    def focus_avg(distribution):
        """calculate the fucus averaged spectrum
            input:  distribution:   calculated momentum distribution [nI,nbins]
            output: S_avg:   focus averaged spectrum [nbins]"""
        nI = distribution.shape[0]
        I_disc, dI = np.linspace(0.001,1,nI,retstep=True)

        x_I = np.zeros(nI)
        # create a gaussian
        x = np.linspace(-4,4,10000)
        gauss = _Avg.gaussian(x,1,1) # create a gaussian
        # calculate the volume where the intensity is within a certain intensity interval
        for i in range(nI):
            x_i = np.argmax(gauss >= I_disc[i]) # get the x value where the intensity is at I_i
            x_I[i] = x[x_i]
        x_I[-1] = 0 # Volume of max intensity is zero
        # calculate the gradient as a weight for the focus averaging
        dVdI = np.gradient(x_I**2)

        S_avg = 0.
        for i in range(nI):
            S_i = np.abs(dVdI[i])*distribution[-i-1]*dI
            S_avg += S_i

        return S_avg


class _Asymmetry(object):
    """docstring for _Asymmetry"""

    def calculate_asymmetry(dist,phisteps, npbins):
        """calculate the asymetry of the given distribution
            input: asymetry_dist: distribution of the focus averaged spectra for each phase
            output: A: asymmetry distribution"""
        A = np.zeros(phisteps)
        half_of_npbins = int(npbins/2)
        for i in range(phisteps):
            left = np.sum(dist[i,:half_of_npbins])
            right = np.sum(dist[i,half_of_npbins:])
            A[i] = (left-right)/(left+right)
        return A

    def sin_fct(x,a,b,c):
        """a sine function for fitting the asymmetry"""
        return a*np.sin(2*np.pi/b*x + c)

    def fit_asymmetry(dist, phisteps, npbins):
        """calculate the asymmetry for each phase and fit a sine function to it"""
        A = _Asymmetry.calculate_asymmetry(dist,phisteps, npbins)
        x = np.arange(A.shape[0])
        popt, cov = curve_fit(_Asymmetry.sin_fct,x,A, p0=[1,phisteps,phisteps/4])
        a, phase = popt[0], popt[2]
        A_fitted = _Asymmetry.sin_fct(np.arange(phisteps), a, popt[1], phase)
        phase = phase/phisteps
        return dist, A, A_fitted, a, phase

class _Run(object):
    """docstring for _Run"""

    def main(phi,nI,pbins,npbins,timesteps):
        """This is the main loop which runs in parallel on all cores"""
        output = _Ionization.final_distribution_intensity_atoms(1,nI,pbins, npbins, phi,timesteps)
        S_avg = _Avg.focus_avg(output[0])
        p_avg = _Avg.focus_avg(output[6]) # not working properly yet
        output = output + (S_avg,p_avg,) # append focus averaged spectrum to the outputs
        return output

class _Save(object):
    """docstring for _Save"""

    def save_results(results):
        """save the results in a binary .npz file"""
        with open('Results/3colours/3coloursraw.npz', 'wb') as f:
            np.savez(f, distribution=results[0],intensities=results[1],N0p=results[2],N1p=results[3],rates=results[4])

    def save_inits_hdf5(savefile,t,pbins,phases):
        """save the initial variables for the simulation in an hdf5 file"""
        f = h5py.File('Results/w3w/'+savefile, 'a') # create a hdf5 file object
        grp = f.create_group('variables')
        names = ['target ion','timesteps','momentum bins', 'C_l', 'I_p', 'Z_c', 'l', 'phases']
        variables = ['Argon', t, pbins, 2.19, 0.579, 1., 1, phases]
        for i in enumerate(variables):
            dset = grp.create_dataset("{}".format(names[i[0]]), data=i[1]) #create a dataset in the hdf5file
        f.flush() # save to disk

    def save_results_hdf5(savefile,results,phi):
        """save the results in an hdf5 file
            create a subgroup for each phi step"""
        f = h5py.File('Results/w3w/'+savefile, 'a') # create a hdf5 file object
        grp = f.create_group('phi={0:.3f}'.format(phi))
        names = ['distribution','intensities','N0p','N1p','rates','E-field','final momentum',
                 'focus averaged spectrum','focus averaged final momentum','phase in pi']
        results = results + (phi,)
        for i in enumerate(results):
            dset = grp.create_dataset("{}".format(names[i[0]]), data=i[1]) #create a dataset in the hdf5file
        f.flush() # save to disk

    def save_asymmetry_hdf5(savefile,results):
        """save the asymetry distribution in an hdf5 file
            create a subgroup for it"""
        f = h5py.File('Results/w3w/'+savefile, 'a') # create a hdf5 file object
        grp = f.create_group('asymmetry')
        names = ['Distribution', 'Asymmetry', 'fitted Asymmeetry', 'Asymmetry parameter', 'Asymmetry phase','E-fields']
        for i in enumerate(results):
            dset = grp.create_dataset("{}".format(names[i[0]]), data=i[1]) #create a dataset in the hdf5file
        f.flush() # save to disk

if __name__ == '__main__':
    """code in here will be executed when running the script"""
    timesteps = 10000
    t, dt = np.linspace(-2050,2050,timesteps,retstep=True) # time in fs 2050au = 50fs
    e = -1 # charge in au
    m = 1 # mass in au

    basename = 'newADK.h5'
    npbins = 50
    pbins = _Ionization.p_bins(3,npbins)
    phisteps = 50
    phases = np.linspace(0,2,phisteps,endpoint=False)
    nI = 10



    savename = basename
    _Save.save_inits_hdf5(savename,t,pbins,phases)
    averaged_dist = np.zeros((phisteps,npbins))
    E_fields = np.zeros((phisteps,timesteps))
    inputs = (nI,pbins,npbins,timesteps)

    num_cores = multiprocessing.cpu_count() # number of cores available
    # return a list of the outputs for different phases
    outputs = Parallel(n_jobs=num_cores)(delayed(_Run.main)(phi, *inputs) for phi in phases) # parallel

    for i in range(phisteps):
        averaged_dist[i,:] = outputs[i][7]
        E_fields[i,:] = outputs[i][5]
        _Save.save_results_hdf5(savename,outputs[i],phases[i])

    results = _Asymmetry.fit_asymmetry(averaged_dist, phisteps, npbins)
    results = results + (E_fields,)
    _Save.save_asymmetry_hdf5(savename,results)
