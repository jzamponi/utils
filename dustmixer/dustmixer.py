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

import os
import sys
import errno
import itertools
import progressbar
import numpy as np
from pathlib import Path
from astropy.io import ascii
from astropy import units as u
import matplotlib.pyplot as plt
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
        self.mf_all = []
        self.Qext = None
        self.Qsca = None
        self.Qabs = None
        self.kext = None
        self.ksca = None
        self.kabs = None
        self.nlam = None

    def __str__(self):
        print(f'{self.name}')

    def __add__(self, other, nlam=None):
        """
            This 'magic' method allows dust opacities to be summed via:
            mixture = dust1 + dust2
            Syntax is important. Arithmetic operations with Dust objects 
            should ideally be grouped by parenthesis. 
        """
        if self.kext is None:
            raise ValueError('Dust opacities kext, ksca and kabs are not set')

        dust = Dust(self)

        # Create a common wavelength grid between both materials
        l_min = np.max([self.l.min(), other.l.min()])
        l_max = np.min([self.l.max(), other.l.max()])
        dust.l = np.logspace(np.log10(l_min), np.log10(l_max), 
            nlam if nlam is not None else np.max([self.l.size, other.l.size]))
        
        # Interpolate the quantities within the new wavelength grid
#        dust.kext = splev(dust.l, splrep(self.l, self.kext))
#        dust.ksca = splev(dust.l, splrep(self.l, self.ksca))
#        dust.kabs = splev(dust.l, splrep(self.l, self.kabs))
#        dust.gsca = splev(dust.l, splrep(self.l, self.gsca))
#        dust.zsca = splev(dust.l, splrep(self.l, self.zsca))
#        other.kext = splev(other.l, splrep(other.l, other.kext))
#        other.ksca = splev(other.l, splrep(other.l, other.ksca))
#        other.kabs = splev(other.l, splrep(other.l, other.kabs))
#        other.gsca = splev(other.l, splrep(other.l, other.gsca))
#        other.zsca = splev(other.l, splrep(other.l, other.zsca))

        # Add the quantities of both materials
        dust.kext = self.kext + other.kext
        dust.ksca = self.ksca + other.ksca
        dust.kabs = self.kabs + other.kabs
        dust.zsca = self.zsca + other.zsca
        #dust.gsca = (self.ksca * self.gsca + other.gsca * other.gsca) /\
        #    (self.ksca + other.ksca)
        dust.gsca = self.gsca + other.gsca
        dust.amin = self.amin
        dust.amax = self.amax
        dust.q = self.q
        dust.na = self.na
        dust.nang = self.nang
        dust.angles = self.angles
        dust.mf_all = self.mf_all
        dust.dens = (self.dens + other.dens) / 2
        dust.name = ' + '.join([self.name, other.name])

        return dust

    def __mul__(self, mass_fraction):
        """
            This 'magic' method allows dust opacities to be rescaled as:
            dust = dust1 * 0.67
            The order is important. The number must be on the right side. 
        """
        return self.set_mass_fraction(mass_fraction) 

    def __rmul__(self, other):
        """ Rightsided multiplication of Dust objects """
        return self * other

    def __div__(self, other):
        """ Division of Dust objects by scalars """
        return self * (1 / other)

    def __truediv__(self, other):
        """ Division of Dust objects by scalars """
        return self * (1 / other)

    def set_mass_fraction(self, mass_fraction):
        """ Set the mass fraction of the dust component. """

        if self.kext is None:
            raise ValueError('Dust opacities kext, ksca and kabs are not set')
    
        if isinstance(mass_fraction, Dust): 
            raise ValueError('Dust can only multiplied by scalars.')
        
        dust = Dust(self)
        dust.name = self.name
        dust.l = self.l
        dust.kext = self.kext * mass_fraction
        dust.ksca = self.ksca * mass_fraction
        dust.kabs = self.kabs * mass_fraction
        dust.gsca = self.gsca * mass_fraction
        dust.zsca = self.zsca * mass_fraction
        dust.nang = self.nang
        dust.amin = self.amin
        dust.amax = self.amax
        dust.q = self.q
        dust.na = self.na
        dust.angles = self.angles
        dust.mf = mass_fraction
        dust.mf_all.append(mass_fraction)
        dust.dens = self.dens * mass_fraction

        return dust

    def check_mass_fractions(self):
        if np.sum(self.mf_all) != 1:
            raise ValueError(
                f'Mass fractions should add up to 1. Values are {self.mf_all}') 
        else:
            utils.print('Mass fractions add up to 1. Values are {self.mf_all}')

    def set_density(self, dens, cgs=True):
        """ Set the bulk density of the dust component. """
        self.dens = dens if cgs else dens * (u.kg/u.m**3).to(u.g/u.cm**3)
    
    def set_volume_fraction(self, volume_fraction):
        """ Set the volume fraction of the dust component. """
        self.vf = vf

    def set_nk(self, path, skip=0, meters=False, cm=False):
        """ Set n and k values by reading them from file. 
            Assumes wavelength is provided in microns unless specified otherwise
         """
        
        # Download the optical constants from the internet if path is a url
        if "http" in path:
            utils.download_file(path)
            path = path.split('/')[-1]

        # Read the table
        utils.print_(f'Reading optical constants from: {path}')
        self.datafile = ascii.read(path, data_start=skip)

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
                
    def mix(self, other, nlam=200):
        """
            Mix two dust components using the bruggeman rule. 

            It initially creates a common wavelength grid by interpolating
            the entered n and k values within the min and max wavelenth.

            *** This feature is currently incomplete. ***
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

        self.angles = np.linspace(0, 180, self.nang)
        self.mass  = (4 / 3 * np.pi) * self.dens * a**3
        self.Qext = np.zeros(self.l.size)
        self.Qsca = np.zeros(self.l.size)
        self.Qabs = np.zeros(self.l.size)
        self.Qbac = np.zeros(self.l.size)
        self.Gsca = np.zeros(self.l.size)
        self.s11 = np.zeros(nang)
        self.s12 = np.zeros(nang)
        self.s33 = np.zeros(nang)
        self.s34 = np.zeros(nang)
        self.Zsca = np.zeros((self.l.size, nang, 6))
        self.current_a = a

        utils.print_(f'Calulating {self.name} efficiencies Q for a grain size'+\
            f' of {np.round(a*u.cm.to(u.micron), 1)} microns', verbose=verbose)

        # Calculate dust efficiencies for a bare grain
        if algorithm.lower() == 'bhmie':
            from bhmie import bhmie
            
            # Iterate over wavelength
            for i, l_ in enumerate(self.l):
                # Define the size parameter
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
                self.Gsca[i] = bhmie_[6]

                # Compute the muller matrix elements per angle if necessary
                # Source: Radiative Transfer Lecture Notes (P. 98) K. Dullemond 
                # and the radmc3d Manual and the radmc3d script makedustopac.py
                if nang >= 5:
                    # Equations 6.33-36 from Lecture Notes and 4.77 from BH86
                    self.s11 = 0.5 * (np.abs(s2)**2 + np.abs(s1)**2)
                    self.s12 = 0.5 * (np.abs(s2)**2 - np.abs(s1)**2)
                    self.s33 = 0.5 * np.real(s1 * np.conj(s2) + s2 * np.conj(s1))
                    self.s34 = 0.5 * np.imag(s1 * np.conj(s2) - s2 * np.conj(s1))
                    
                    # "Polarized scattering off dust particles" from the manual
                    k = 2 * np.pi / l_
                    factor = 1 / (k**2 * self.mass)
                    self.Zsca[i, :, 0] = self.s11 / (k**2 * self.mass)
                    self.Zsca[i, :, 1] = self.s12 / (k**2 * self.mass)
                    self.Zsca[i, :, 2] = self.s11 / (k**2 * self.mass)
                    self.Zsca[i, :, 3] = self.s33 / (k**2 * self.mass)
                    self.Zsca[i, :, 4] = self.s34 / (k**2 * self.mass)
                    self.Zsca[i, :, 5] = self.s33 / (k**2 * self.mass)

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

        return self.Qext, self.Qsca, self.Qabs, self.Gsca, self.Zsca

        
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
        self.zsca = np.zeros((self.l.size, nang, 6))
        self.Qext_a = []
        self.Qsca_a = []
        self.Qabs_a = []
        self.gsca_a = []
        self.zsca_a = []
        self.nproc = nproc
 
        utils.print_(f'Calculating efficiencies for {self.name} using ' +\
            f'{self.na} sizes between {a.min()} and {int(a.max())} microns ...')

        # In case of a single grain size, skip parallelization and integration
        if self.amin == self.amax or self.na == 1:
            qe, qs, qa, gs, zs = self.get_efficiencies(self.a[0],nang,algorithm)
            self.kext = qe * np.pi * self.a[0]**2
            self.ksca = qa * np.pi * self.a[0]**2
            self.kabs = qa * np.pi * self.a[0]**2
            self.gsca = gs
            self.zsca = zs
            
            return self.kext, self.ksca, self.kabs, self.gsca, self.zsca

        # Serial execution
        if self.nproc == 1:
            # Customize the progressbar
            widgets = [f'[get_opacities] ', progressbar.Timer(), ' ', 
                progressbar.GranularBar(' ⡀⡄⡆⡇⣇⣧⣷⣿')]

            pb = progressbar.ProgressBar(maxval=self.a.size, widgets=widgets)
            pb.start()

            # Calculate the efficiencies for the range of grain sizes
            for j, a_ in enumerate(self.a):
                qe, qs, qa, gs, zs = self.get_efficiencies(
                        a_, nang, algorithm, verbose=False)
                self.Qext_a.append(qe) 
                self.Qsca_a.append(qs) 
                self.Qabs_a.append(qa) 
                self.gsca_a.append(gs) 
                self.zsca_a.append(zs) 
                pb.update(j)
            pb.finish()

        # Multiprocessing (Parallelized)
        else:
            # Calculate the efficiencies for the range of grain sizes
            with multiprocessing.Pool(processes=self.nproc) as pool:
                params = zip(
                    self.a, 
                    itertools.repeat(nang), 
                    itertools.repeat(algorithm),
                    itertools.repeat(None),
                    itertools.repeat(False),
                    range(self.a.size), 
                )
                # Parallel map function
                qe, qs, qa, gs, zs = pool.starmap(self.get_efficiencies, params)
                
                # Reorder from (a, Q, l) to (Q, a, l)
                self.Qext_a = qe
                self.Qsca_a = qs
                self.Qabs_a = qa
                self.gsca_a = gs
                self.zsca_a = zs
    
        # Transpose from (a, l) to (l, a) to later integrate over l
        self.Qext_a = np.transpose(self.Qext_a)
        self.Qsca_a = np.transpose(self.Qsca_a)
        self.Qabs_a = np.transpose(self.Qabs_a)
        self.gsca_a = np.transpose(self.gsca_a)
        self.zsca_a = np.swapaxes(self.zsca_a, 0, 1)
        
        utils.print_(f'Integrating opacities and scattering matrix using '+\
            f'a power-law slope of {q = }')

        # Mass integral: int (a^q * a^3) da = [amax^(q+4) - amin^(q-4)]/(q-4)
        q4 = self.q + 4
        int_da = (self.amax**q4 - self.amin**q4) / q4
        
        # Total mass normalization constant
        mass_norm = 4 / 3 * np.pi * self.dens * int_da

        # Size distribution
        phi = self.a**self.q

        # Calculate mass weight for the size integration of Z11
        mass = 4/3 * np.pi * self.a**3 * self.dens 
        m_of_a = (self.a*u.cm.to(u.micron))**(self.q + 1) * mass
        mtot = np.sum(m_of_a)
        mfrac = m_of_a / mtot

        # Integrate quantities over size distribution per wavelength 
        for i, l_ in enumerate(self.l):
            sigma_geo = np.pi * self.a**2
            Cext = self.Qext_a[i] * sigma_geo
            Csca = self.Qsca_a[i] * sigma_geo
            Cabs = self.Qabs_a[i] * sigma_geo

            # Integrate Zij
            for j in range(self.nang): 
                for zij in range(6):
                    self.zsca[i][j][zij] = np.sum(self.zsca_a[i,:,j,zij]*mfrac)
            
            # Angular integral of Z11
            mu = np.cos(self.angles * np.pi / 180)
            int_Z11_dmu = -np.trapz(self.zsca[i, :, 0], mu)
            int_Z11_mu_dmu = -np.trapz(self.zsca[i, :, 0] * mu, mu)

            if self.nang < 5:
                self.ksca[i] = np.trapz(Csca * phi, self.a) / mass_norm
            else:
                self.ksca[i] = 2 * np.pi * int_Z11_dmu

            self.gsca[i] = 2 * np.pi * int_Z11_mu_dmu / self.ksca[i]
            self.kext[i] = np.trapz(Cext * phi, self.a) / mass_norm
            self.kabs[i] = np.trapz(Cabs * phi, self.a) / mass_norm

            # Calculate the relative error between kscat and int Z11 dmu
            if self.nang >= 5:
                self.compare_ksca_vs_z11(i)

        if self.nang >= 5:
            self.check_ksca_z11_error(tolerance=0.1, show=False)

        return self.kext, self.ksca, self.kabs, self.gsca, self.zsca

    def compare_ksca_vs_z11(self, lam_i):
        """ Compute the relative error between ksca and int Z11 dmu """
        self.err_i = np.zeros(self.l.size)
        mu = np.cos(self.angles * np.pi / 180)
        dmu = np.abs(mu[1:self.nang] - mu[0:self.nang-1])
        zav = 0.5 * (self.zsca[lam_i, 1: self.nang,0] + 
            self.zsca[lam_i, 0:self.nang-1, 0])
        dum = 0.5 * zav * dmu
        self.dumsum = 4 * np.pi * dum.sum()
        err = np.abs(self.dumsum / self.ksca[lam_i] - 1)
        self.err_i[lam_i] = np.max(err, 0)
    
    def check_ksca_z11_error(self, tolerance, show=False):
        """ Warn if the error between kscat and int Z11 dmu is large """
        if np.any(self.err_i > tolerance):
            maxerr = np.round(self.err_i.max(), 1)
            lam_maxerr = self.l[utils.maxpos(self.err_i)] * u.cm.to(u.micron)
            utils.print_(
                'The relative error between ksca and the ' +\
                f'angular integral of Z11 is larger than {tolerance}.',red=True)
            utils.print_(
                f'Max Error: {maxerr} at {lam_maxerr:.1f} microns', bold=True)

            if show:
                plt.semilogx(self.l*u.cm.to(u.micron), self.err_i)
                plt.xlabel('Wavelength (microns)')
                plt.xlabel('Relative error')
                plt.annotate(
                    r'$err=\frac{\kappa_{\rm sca}}{\int Z_{11}(\mu)d\mu}$',
                    xy = (0.1, 0.9), xycoords='axes fraction', size=20)
                plt.show()

    def write_opacity_file(self, name=None, scatmat=False):
        """ Write the dust opacities into a file ready for radmc3d """ 

        # Parse the table filename 
        name = self.name if name is None else name
        outfile = f'dustkappa_{name.lower()}.inp'
        if scatmat: 
            outfile = outfile.replace('kappa', 'kapscatmat') 

        utils.print_(f'Writing out radmc3d opacity file: {outfile}')
        with open(outfile, 'w+') as f:
            # Write a comment with info
            f.write(f'# Opacity table generated by Dustmixer\n')
            f.write(f'# Material = {self.name}\n')
            f.write(f'# Density = {self.dens} g/cm3\n')
            f.write(f'# Minimum grain size = {np.round(self.amin*1e4, 1)}um\n')
            f.write(f'# Maximum grain size = {np.round(self.amax*1e4, 1)}um\n')
            f.write(f'# Number of sizes = {self.na}\n')
            f.write(f'# Distribution slope = {self.q}\n')
            f.write(f'# Number of scattering angles: {self.nang}\n')

            # Write file header
            f.write('1\n' if scatmat else '3\n')
            f.write(f'{self.l.size}\n')
            if scatmat:
                f.write(f'{self.nang}\n')
            
            # Write the opacities and g parameter per wavelenght
            for i, l in enumerate(self.l):
                f.write(f'{l*u.cm.to(u.micron):.6e}\t{self.kabs[i]:13.6e}\t')
                f.write(f'{self.ksca[i]:13.6e}\t{self.gsca[i]:13.6e}\n')

            if scatmat:
                for j, ang in enumerate(self.angles):
                    # Write scattering angle sampling points in degrees
                    f.write(f'{ang}\n')

                for i, l in enumerate(self.l):
                    for j, ang in enumerate(self.angles):
                        # Write the Mueller matrix components
                        f.write(f'{self.zsca[i, j, 0]:13.6e} ')
                        f.write(f'{self.zsca[i, j, 1]:13.6e} ')
                        f.write(f'{self.zsca[i, j, 2]:13.6e} ')
                        f.write(f'{self.zsca[i, j, 3]:13.6e} ')
                        f.write(f'{self.zsca[i, j, 4]:13.6e} ')
                        f.write(f'{self.zsca[i, j, 5]:13.6e}\n')

    def write_align_factor(self, name=None):
        """ Write the dust alignment factor into a file ready for radmc3d """ 

        # Parse the table filename 
        name = self.name if name is None else name
        outfile = f'dustkapalignfact_{name.lower()}.inp'
        utils.print_(f'Writing out radmc3d align factor file: {outfile}')

        # Create a mock alignment model. src:radmc3d/examples/run_simple_1_align
        mu = np.linspace(1, 0, self.nang)
        eta = np.arccos(mu) * 180 / np.pi
        amp = 0.5
        orth = np.ones(self.nang)
        para = (1 - amp * np.cos(mu * np.pi)) / (1 + amp)

        with open(outfile, 'w+') as f:
            f.write('1\n')
            f.write(f'{self.l.size}\n')
            f.write(f'{self.nang}\n')

            for l in self.l:
                f.write(f'{l*u.cm.to(u.micron):13.6e}\n')

            for i in eta:
                f.write(f'{i:13.6e}\n')

            for j in range(self.l.size):
                for a in range(self.nang):
                    f.write(f'{orth[a]:13.6e}\t{para[a]:13.6e}\n')

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
    grap_per = Dust(name='Graphite Perpendicular')
    grap_par = Dust(name='Graphite Parallel')

    # Load refractive indices n and k from files
    silicate.set_nk(path=f'silicate.nk', meters=True, skip=2)
    grap_per.set_nk(path=f'graphite_perpend.nk', meters=True, skip=2)
    grap_par.set_nk(path=f'graphite_parallel.nk', meters=True, skip=2)

    # Set the mass fraction and bulk density of each component
    silicate.set_density(3.50, cgs=True)
    grap_per.set_density(2.25, cgs=True)
    grap_par.set_density(2.25, cgs=True)

    # Convert the refractive indices into dust opacities
    silicate.get_opacities(a=np.logspace(-1, 1, 100), nang=5)
    grap_per.get_opacities(a=np.logspace(-1, 1, 100), nang=5)
    grap_par.get_opacities(a=np.logspace(-1, 1, 100), nang=5)

    #mixture = (silicate * 0.625) + (grap_per * 0.250) + (grap_par * 0.125)
    mixture = (0.625 * silicate) + (0.250 * grap_per) + (grap_par * 0.125)

    silicate.plot_opacities(show=False, savefig='silicate_opacity.png')
    grap_per.plot_opacities(show=False, savefig='grap_per_opacity.png')
    grap_par.plot_opacities(show=False, savefig='grap_par_opacity.png')
    mixture.plot_opacities(show=False, savefig='mixture_opacity.png')

    # Write the opacity table of the mixed material including scattering matrix
    mixture.write_opacity_file(scatmat=True, name='sg-a10um')

