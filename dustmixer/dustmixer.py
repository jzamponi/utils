#!/usr/bin/env python3
"""
    This python program defines the Dust object, which can be used to
    calculate dust opacities from tabulated optical constants n & k. 
    This program also allows to mix different materials either using the
    Bruggeman rule for mixing optical constants into a new material or by 
    simply averaging the resultant opacities, weighted by a given mass fraction.

    The resultant extinction, scattering and absorption opacities can be 
    returned as arrays or written to file in radmc3d compatible format. 
    This file can optionally include the mueller matrix components for full 
    scattering and polarization modelling.

    At the end of this script an example of implementation can be found. 

    Wavelengths and grain sizes should be provided in microns, however all
    internal calculations are done in cgs.
"""

import sys
import itertools
import progressbar
import numpy as np
from pathlib import Path
from astropy.io import ascii
from astropy import units as u
from time import time, strftime, gmtime
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from scipy.interpolate import interp1d, splrep, splev
import utils

class Dust():
    """ Dust object. Defines the properties of a dust component. """
    verbose = False

    def __init__(self, name=''):
        self.name = name
        self.l = []
        self.n = []
        self.k = []
        self.mf = 0
        self.vf = 0
        self.dens = 0
        self.mass = 0
        self.Qext = None
        self.Qsca = None
        self.Qabs = None
        self.kext = None
        self.ksca = None
        self.kabs = None

    def __str__(self):
        print(f'{self.name}')

    def __repr(self):
        print(f'{self.name}')

    def __add__(self, other):
        """
            This 'magic' method allows dust opacities to be summed via:
            mixture = dust1 + dust2
        """
        if self.kext is None:
            raise ValueError('Dust opacities kext, ksca and kabs are not set')

        self.kext = self.kext + other.kext
        self.ksca = self.ksca + other.ksca
        self.kabs = self.kabs + other.kabs

        return Dust(self)

    def __mul__(self, mf):
        """
            This 'magic' method allows dust opacities to be rescaled as:
            dust = dust1 * 0.67
        """
        if self.kext is None:
            raise ValueError('Dust opacities kext, ksca and kabs are not set')

        self.kext = self.kext * mf
        self.ksca = self.ksca * mf
        self.kabs = self.kabs * mf

        return Dust(self)

    def set_nk(self, path, data_start=0, meters=False, cm=False):
        """ Set n and k values by reading them from file. 
            Assumes wavelength is provided in microns unless specified otherwise
         """
        
        # Download the optical constants from the internet if path is a url
        if "http" in path:
            utils.download_file(path)
            path = path.split('/')[-1]

        # Read the table
        try:
            utils.print_(f'Reading optical constants from: {path}')
            self.datafile = ascii.read(path, data_start=data_start)
        except InconsistentTableError:
            raise(f"I can't read the table from file. Received: \n" +\
                "{path = }\n{data_start = }")

        # Override the default column names used by astropy.io.ascii.read
        column_name = {'l': 'col1', 'n': 'col2', 'k': 'col3'}
        self.n = self.datafile[column_name['n']]
        self.k = self.datafile[column_name['k']]
        
        # Parse the wavelength units to ensure they are in cm
        if meters:
            self.l = self.datafile[column_name['l']] * u.m.to(u.cm)
        elif cm:
            self.l = self.datafile[column_name['l']]
        else:
            self.l = self.datafile[column_name['l']] * u.micron.to(u.cm)
                
    def set_density(self, dens, cgs=True):
        """ Set the bulk density of the dust component. """
        self.dens = dens if cgs else dens * (u.kg/u.m**3).to(u.g/u.cm**3)
    
    def set_mass_fraction(self, mf):
        """ Set the mass fraction of the dust component. """
        self.mf = mf

    def set_volume_fraction(self, vf):
        """ Set the volume fraction of the dust component. """
        self.vf = vf

    def mix(self, other, nlam=200):
        """
            Mix two dust components using the bruggeman rule. 

            It initially creates a common wavelength grid by interpolating
            the entered n and k values within the min and max wavelenth.
            TO DO: change other for *args so it can handle multiple materials
        """

        # Creates a new Dust instance to contain the mixed optical constants
        mixture = Dust(self)
        mixture.name = " + ".join([self.name,other.name])

        # Create a wavelength grid covering the range of both input tables
        self.l_min = np.max([np.min(self.l), np.min(other.l)])
        self.l_max = np.min([np.max(self.l), np.max(other.l)])
        self.l = np.logspace(
            np.log10(self.l_min), np.log10(self.l_max), nlam)
        
        # Interpolate n and k values using cubic spline
        self.n1_interp = splev(self.l, splrep(self.l, self.n)).clip(min=0.0)
        self.k1_interp = splev(self.l, splrep(self.l, self.k)).clip(min=0.0)
        self.n2_interp = splev(self.l, splrep(other.l, other.n)).clip(min=0.0)
        self.k2_interp = splev(self.l, splrep(other.l, other.k)).clip(min=0.0)

        # Apply the Bruggeman rule for n & k mixing
        self.n_mixed, self.k_mixed = self.bruggeman_mixing([self.vf, other.vf])

        mixture.n = self.n_mixed
        mixture.k = self.k_mixed
        mixture.dens = np.dot([self.dens, other.dens], [self.vf, other.vf])

        return mixture

    def bruggeman_mixing(self, vf):
        """ This function explicity mixes the n & k indices.
            Based on section 2 from Birnstiel et al. (2018).
            Receives: list of volume fractions
            Returns: (n,k)
        """
        from mpmath import findroot

        # Let epsilon = m^2 = (n+ik)^2
        eps = np.array([ 
                [complex(n,k)**2 for (n,k) in zip(self.n1_interp, self.k1_interp)], 
                [complex(n,k)**2 for (n,k) in zip(self.n2_interp, self.k2_interp)]
        ])
        
        # Let f_i be the vol. fractions of each material
        f_i = np.array(vf)

        # Iterate over wavelenghts
        eps_mean = np.empty(np.shape(self.l)).astype('complex')
        for i, l in enumerate(self.l):
            # Define the expresion for mixing and solve for eps_mean
            expression = lambda x: sum(f_i * ((eps[:,i]-x) / (eps[:,i]+2*x)))
            eps_mean[i] = complex(findroot(expression, complex(0.5,0.5)))
        
        eps_mean = np.sqrt(eps_mean)

        return eps_mean.real.squeeze(), eps_mean.imag.squeeze()
        
    def get_efficiencies(self, a, nang=3, algorithm='bhmie', coat=None, 
            verbose=True, parallel_counter=0):
        """ 
            Compute the extinction, scattering and absorption
            efficiencies (Q) by calling bhmie or bhcoat.

            Arguments: 
              - a: Size of the dust grain in cm
              - nang: Number of angles to sample scattering between 0 and 180
              - algorithm: 'bhmie' or 'bhcoat', algorithm used to calculate Q 
              - coat: Dust Object, material used as a iced coat for the grain
        
            Returns:
              - Qext: Dust extinction efficiency
              - Qsca: Dust scattering efficiency
              - Qabs: Dust absorption efficiency
              - gsca: Assymetry parameter for Henyey-Greenstein scattering
        """

        if nang >= 2:
            self.nang = nang
        else:
            raise ValueError('nang must be greater or equal than 2')

        self.angles = np.linspace(0, 180, self.nang)
        self.mass  = (4 / 3 * np.pi) * self.dens * a**3
        self.Qext = np.zeros(self.l.size)
        self.Qsca = np.zeros(self.l.size)
        self.Qabs = np.zeros(self.l.size)
        self.Qbac = np.zeros(self.l.size)
        self.gsca = np.zeros(self.l.size)
        self.s11 = np.zeros(nang)
        self.s12 = np.zeros(nang)
        self.s33 = np.zeros(nang)
        self.s34 = np.zeros(nang)
        self.zscat = np.zeros((self.l.size, nang, 6))
        self.current_a = a

        utils.print_('Calulating dust efficiencies Q for a grain size of '+\
            f'{np.round(a*u.cm.to(u.micron), 1)} microns', verbose=verbose)

        # Calculate dust efficiencies for a bare grain
        if algorithm.lower() == 'bhmie':
            from bhmie import bhmie
            
            # Iterate over wavelength
            for i, l_ in enumerate(self.l):
                # Iterate over grain radii
                self.x = 2 * np.pi * a / l_

                # Define the complex refractive index (m)
                self.m = complex(self.n[i], self.k[i])
                
                # Call BHMie (Bohren & Huffman 1986)
                bhmie_ = bhmie(self.x, self.m, self.angles)
                
                # Store the Mueller matrix elements and dust efficiencies
                s1 = bhmie_[0]
                s2 = bhmie_[1]
                self.Qext[i] = bhmie_[2]
                self.Qsca[i] = bhmie_[3]
                self.Qabs[i] = bhmie_[4]
                self.Qbac[i] = bhmie_[5]
                self.gsca[i] = bhmie_[6]

                # Compute the muller matrix elements per angle if necessary
                # Source: Radiative Transfer Lecture Notes (P. 98) K. Dullemond 
                # and the radmc3d Manual and the radmc3d script makedustopac.py
                if nang >= 3:
                    # Equations 6.33-36 from Lecture Notes and 4.77 from BH86
                    self.s11 = 0.5 * (np.abs(s2)**2 + np.abs(s1)**2)
                    self.s12 = 0.5 * (np.abs(s2)**2 - np.abs(s1)**2)
                    self.s33 = 0.5 * np.real(s1 * np.conj(s2) + s2 * np.conj(s1))
                    self.s34 = 0.5 * np.imag(s1 * np.conj(s2) - s2 * np.conj(s1))

#                    s33_kees = np.real(s2 * np.conj(s1))
#                    s34_kees = np.imag(s2 * np.conj(s1))
#                    assert (self.s33 - s33_kees == 0).all(), 'S33 and S33_kees differ'
#                    assert (self.s34 - s34_kees == 0).all(), 'S34 and S34_kees differ'
                    
                    # "Polarized scattering off dust particles" from the manual
                    k = 2 * np.pi / l_
                    self.zscat[i, :, 0] += self.s11 / (k**2 * self.mass)
                    self.zscat[i, :, 1] += self.s12 / (k**2 * self.mass)
                    self.zscat[i, :, 2] += self.s11 / (k**2 * self.mass)
                    self.zscat[i, :, 3] += self.s33 / (k**2 * self.mass)
                    self.zscat[i, :, 4] += self.s34 / (k**2 * self.mass)
                    self.zscat[i, :, 5] += self.s33 / (k**2 * self.mass)

        # Calculate dust efficiencies for a coated grain
        elif algorithm.lower() == 'bhcoat':
            from bhcoat import bhcoat

            utils.print_('The implementation for coated grains is currently '+\
                'incomplete.', red=True)

            if coat is None:
                raise ValueError(f'In order to use bhcoat you must provide '+\
                    'a Dust object to use as a coat')

            # Set a common wavelength grid
            l_min = np.max([self.l.min(),coat.l.min()])
            l_max = np.min([self.l.max(),coat.l.max()])
            self.l = np.logspace(np.log10(l_min), np.log10(l_max), 
                self.l.size)
            
            # Interpolate core and mantle indices
            self.n_interp = splev(self.l, splrep(self.l, self.n)).clip(min=0)
            self.k_interp = splev(self.l, splrep(self.l, self.k)).clip(min=0)
            coat.n_interp = splev(self.l, splrep(coat.l, coat.n)).clip(min=0)
            coat.k_interp = splev(self.l, splrep(coat.l, coat.k)).clip(min=0)
            self.n = self.n_interp
            self.k = self.k_interp
            
            # Set the grain sizes for the coat
            a_coat = a * coat.vf

            for i, l_ in enumerate(self.l):
                # Define the size parameter
                self.x = 2 * np.pi * a / l_
                self.y = 2 * np.pi * a_coat / l_

                # Set the complex refractive index for the core
                self.m_core = complex(self.n_interp[i], self.k_interp[i])

                # Set the complex refractive index for the mantle
                self.m_mant = complex(coat.n_interp[i], coat.k_interp[i])
                
                # Calculate the efficiencies for a coated grain
                bhcoat_ = bhcoat(self.x, self.y, self.m_core, self.m_mant)
                
                self.Qext[i] = bhcoat_[0]
                self.Qsca[i] = bhcoat_[1]
                self.Qabs[i] = bhcoat_[2]
                self.Qbac[i] = bhcoat_[3]

        else:
            raise ValueError(f'Invalid value for algorithm = {algorithm}.')

        # Print a simpler progress meter if using multiprocessing
        if self.nproc > 1:
            i = parallel_counter
            counter = i * 100 / self.a.size
            endl = '\r' if i != self.a.size-1 else '\n'
            bar = (' ' * 20).replace(" ", "⣿⣿", int(counter / 100 * 20))
            sys.stdout.write(f'[get_efficiencies] Using {self.nproc} processes'+\
                f' | Progress: {counter} % |{bar}| {endl}')
            sys.stdout.flush()

        return self.Qext, self.Qsca, self.Qabs, self.gsca

        
    @utils.elapsed_time
    def get_opacities(self, a=np.logspace(-1, 2, 100), q=-3.5, 
            algorithm='bhmie', nang=2, nproc=1):
        """ 
            Convert the dust efficiencies into dust opacities by integrating 
            them over a range of grain sizes. Assumes grain sizes are given in
            microns and a power-law size distribution with slope q.

            Arguments:  for a={int(self.a)} microns for a={int(self.a)} microns
              - a: Array containing the sizes of the grain size distribution
              - q: Exponent of the power-law grain size distribution
              - algorithm: 'bhmie' or 'bhcoat', algorithm for get_efficiencies
              - nang: Number of angles used in get_efficiencies
        
            Returns:
              - kext: Dust extinction opacity (cm^2/g_dust)
              - ksca: Dust scattering opacity (cm^2/g_dust)
              - kabs: Dust absorption opacity (cm^2/g_dust)
        """

        if np.isscalar(a): self.a = np.array([a]) 
        self.a = a * u.micron.to(u.cm)
        self.amin = self.a.min()
        self.amax = self.a.max()
        self.q = q
        self.na = np.size(self.a)
        self.nang = nang
        self.angles = np.linspace(0, 180, nang)
        self.kext = np.zeros(self.l.size)
        self.ksca = np.zeros(self.l.size)
        self.kabs = np.zeros(self.l.size)
        self.gsca = np.zeros(self.l.size)
        self.Qext_a = []
        self.Qsca_a = []
        self.Qabs_a = []
        self.nproc = nproc
 
        utils.print_(f'Calculating dust efficiencies for {self.na} grain ' +\
            f'sizes between {a.min()} and {int(a.max())} microns ...')

        # Serial execution
        if self.nproc == 1:
            # Customize the progressbar
            widgets = [f'[get_opacities] ', progressbar.Timer(), ' ', 
                progressbar.GranularBar(' ⡀⡄⡆⡇⣇⣧⣷⣿')]

            pb = progressbar.ProgressBar(maxval=self.a.size, widgets=widgets)
            pb.start()

            # Calculate the efficiencies for the range of grain sizes
            for j, a_ in enumerate(self.a):
                self.get_efficiencies(a_, nang, algorithm, verbose=False)
                self.Qext_a.append(self.Qext) 
                self.Qsca_a.append(self.Qsca) 
                self.Qabs_a.append(self.Qabs) 
                pb.update(j)
            pb.finish()

        # Multiprocessing (Parallelized)
        else:
            # Calculate the efficiencies for the range of grain sizes
            with multiprocessing.Pool(processes=self.nproc) as pool:
                result = pool.starmap(
                    self.get_efficiencies, 
                    zip(
                        self.a, 
                        itertools.repeat(nang), 
                        itertools.repeat(algorithm),
                        itertools.repeat(None),
                        itertools.repeat(False),
                        range(self.a.size), 
                    )
                )
                # Reorder from (a, Q, l) to (Q, a, l)
                result = np.swapaxes(result, 1, 0)
                self.Qext_a = result[0]
                self.Qsca_a = result[1]
                self.Qabs_a = result[2]
    
        # Transpose from (a, l) to (l, a) to later integrate over l
        self.Qext_a = np.transpose(self.Qext_a)
        self.Qsca_a = np.transpose(self.Qsca_a)
        self.Qabs_a = np.transpose(self.Qabs_a)

        # Integral of (a^q * a^3) = [amax^(q+4) - amin^(q-4)]/(q-4)
        q4 = self.q + 4
        int_da = (self.amax**q4 - self.amin**q4) / q4
        C = (4 / 3 * np.pi * self.dens * int_da)

        utils.print_(f'Integrating efficiencies using a power-law slope of {q = }')

        # Compute the integral per wavelength 
        for i, l_ in enumerate(self.l):
            self.kext[i] = np.pi / C * \
                np.trapz(self.Qext_a[i] * self.a**2 * self.a**q, self.a)

            self.ksca[i] = np.pi / C * \
                np.trapz(self.Qsca_a[i] * self.a**2 * self.a**q, self.a)

            self.kabs[i] = np.pi / C * \
                np.trapz(self.Qabs_a[i] * self.a**2 * self.a**q, self.a)

        return self.kext, self.ksca, self.kabs

    def write_opacity_file(self, name=None, scatmat=False):
        """ Write the dust opacities into a file ready for radmc3d """ 

        # Parse the table filename 
        name = self.name if name is None else name
        outfile = f'dustkappa_{name.lower()}.inp'
        if scatmat: 
            outfile = outfile.replace('kappa', 'kapscatmat') 

        utils.print_(f'Writing out radmc3d opacity file: {outfile}')
        with open(outfile, 'w+') as f:
            # Write file header
            f.write('1\n' if scatmat else '3\n')
            f.write(f'{self.l.size}\n')
            if scatmat:
                f.write(f'{self.nang}\n')
            
            # Write the opacities and g parameter per wavelenght
            for i, l in enumerate(self.l):
                f.write(f'{l:.6e}\t{self.kabs[i]:13.6e}\t')
                f.write(f'{self.ksca[i]:13.6e}\t{self.gsca[i]:13.6e}\n')

            if scatmat:
                for j, ang in enumerate(self.angles):
                    # Write scattering angle sampling points in degrees
                    f.write(f'{ang}\n')

                for i, l in enumerate(self.l):
                    for j, ang in enumerate(self.angles):
                        # Write the Mueller matrix components
                        f.write(f'{self.zscat[i, j, 0]:13.6e} ')
                        f.write(f'{self.zscat[i, j, 1]:13.6e} ')
                        f.write(f'{self.zscat[i, j, 2]:13.6e} ')
                        f.write(f'{self.zscat[i, j, 3]:13.6e} ')
                        f.write(f'{self.zscat[i, j, 4]:13.6e} ')
                        f.write(f'{self.zscat[i, j, 5]:13.6e}\n')

    def plot_nk(self, show=True, savefig=None):
        """ Plot the interpolated values of the refractive index (n & k). """
        import matplotlib.pyplot as plt
        
        if len(self.n) == 0 or len(self.k) == 0: 
            raise AttributeError('Optical constants n and k have not been set.')

        plt.close()
        fig, p = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
        twin_p = p.twinx()

        l = self.l * u.cm.to(u.micron)
        n = p.semilogx(l, self.n, ls='-', color='black')
        k = twin_p.loglog(l, self.k, ls=':', color='black')
        p.text(0.10, 0.95, self.name, fontsize=13, transform=p.transAxes)
        p.legend(n+k, ['n','k'], loc='upper left')
        p.set_xlabel('Wavelength (microns)')
        p.set_ylabel('n')
        twin_p.set_ylabel('k')
        plt.tight_layout()

        return utils.plot_checkout(fig, show, savefig)

    def plot_efficiencies(self, show=True, savefig=None):
        """ Plot the extinction, scattering & absorption eficiencies.  """
        import matplotlib.pyplot as plt

        if self.Qext is None or self.Qsca is None: 
            raise AttributeError('Dust efficiencies have not been calculated.')

        plt.close()
        fig, p = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
        a = np.round(self.current_a * u.cm.to(u.micron), 3)
        utils.print_(f'Plotting dust efficiencies for {a = } microns')
        
        p.loglog(self.l*u.cm.to(u.micron), self.Qext, ls='-', c='black',)
        p.loglog(self.l*u.cm.to(u.micron), self.Qsca, ls=':', c='black',)
        p.loglog(self.l*u.cm.to(u.micron), self.Qabs, ls='--', c='black')
        p.legend([r'$Q_{\rm ext}$', r'$Q_{\rm sca}$', r'$Q_{\rm abs}$'])
        p.annotate(r'$a = $'+f' {a} '+r'$\mu$m', xy=(0.1, 0.1), 
            xycoords='axes fraction', size=20)
        p.text(0.05, 0.95, self.name, fontsize=13, transform=p.transAxes)
        p.set_xlabel('Wavelength (microns)')
        p.set_ylabel(r'$Q$')
        plt.tight_layout()

        return utils.plot_checkout(fig, show, savefig)

    def plot_opacities(self, show=True, savefig=None):
        """ Plot the extinction, scattering & absorption eficiencies.  """
        import matplotlib.pyplot as plt

        if self.kext is None or self.ksca is None: 
            raise AttributeError('Dust opacities have not been calculated.')

        plt.close()
        fig, p = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
        utils.print_(f'Plotting dust opacities')

        p.loglog(self.l*u.cm.to(u.micron), self.kext, ls='-', c='black')
        p.loglog(self.l*u.cm.to(u.micron), self.ksca, ls=':', c='black')
        p.loglog(self.l*u.cm.to(u.micron), self.kabs, ls='--', c='black')
        p.legend([r'$k_{\rm ext}$', r'$k_{\rm sca}$', r'$k_{\rm abs}$'])
        p.text(0.05, 0.95, self.name, fontsize=13, transform=p.transAxes)
        p.set_xlabel('Wavelength (microns)')
        p.set_ylabel(r'Dust opacity $k$ (cm$^2$ g$^{-1}$)')
        p.set_ylim(1e-2, 1e4)
        p.set_xlim(1e-1, 3e4)
        plt.tight_layout()

        return utils.plot_checkout(fig, show, savefig)



if __name__ == "__main__":
    """
        Below is just an example implementation of the dustmixer.
    """

    # Create Dust materials
    silicate = Dust(name='Silicate')
#    grap_per = Dust(name='Graphite Perpendicular')
#    grap_par = Dust(name='Graphite Parallel')

    # Load refractive indices n and k from files
    silicate.set_nk(path='silicate.nk', meters=True, data_start=2)
#    grap_per.set_nk(path='graphite_perpend.nk', meters=True, data_start=2)
#    grap_par.set_nk(path='graphite_parallel.nk', meters=True, data_start=2)

    # Set the mass fraction and bulk density of each component
    silicate.set_density(3.50, cgs=True)
#    grap_per.set_density(2.25, cgs=True)
#    grap_par.set_density(2.25, cgs=True)

    # Bypass mixing and simply average the opacites of coexisting materials,  
    # weighting them by their mass fraction. Make sure the fractions add up to 1
#    mixture = (0.625 * silicate) + (0.250 * grap_per) + (0.125 * grap_par) 
#    mixture = silicate + grap_per + grap_par 

    # Convert the refractive indices into dust opacities
    kext, ksca, kabs = silicate.get_opacities(a=np.logspace(-1, 1, 10), nang=4)

    silicate.plot_efficiencies()
    silicate.plot_opacities()

    # Write the opacity table of the mixed material including scattering matrix
    silicate.write_opacity_file(scatmat=True)

