from time import time, strftime, gmtime
import sys
import pathlib
import numpy as np
from scipy.interpolate import interp1d, splrep, splev
from astropy.io import ascii
from astropy import units as u

class Dust():
	"""	Dust object. Defines the properties of a dust component. """
	verbose = False

	def __init__(self, name=''):
		self.name = name
		self.l = []
		self.n = []
		self.k = []
		self.dens = 0
		self.f_m = 0
		self.f_v = 0

	def set_nk(self, path, data_start=0, colnames={'l':'col1', 'n':'col2', 'k':'col3'}):
		""" Set n and k values by reading them from file """
		try:
			self.datafile = ascii.read(path, data_start=data_start)
		except InconsistentTableError:
			raise("[set_nk] Incorrect value for data_start or path. Can't read the table")
		self.l = self.datafile[colnames['l']]
		self.n = self.datafile[colnames['n']]
		self.k = self.datafile[colnames['k']]
		#
		# Rescale wavelength if provided in wavenumber
		with open(path, 'r') as f:
			if '1/cm' in f.read():
				self.l = (1/self.l)[::-1]
			else:
				self.l = self.l * u.micron.to(u.cm)
				
	def set_density(self, dens, mass_fraction=None, vol_fraction=None):
		""" Set the bulk density of the dust component. """
		self.dens = dens
		self.f_m = mass_fraction if mass_fraction is not None else 1.0
		self.f_v = vol_fraction if vol_fraction is not None else 1.0

	def mix(self, other, l_grid_size=240):
		"""
			Mix two dust components using the bruggeman rule.
			It first creates a common wavelength grid by interpolating
			n and k values within the min and max wavelenth.
			TO DO: change other for *args so it can handle multiple comps
		"""
		self.l_min = np.max([np.min(self.l), np.min(other.l)])
		self.l_max = np.min([np.max(self.l), np.max(other.l)])
		self.l_grid = np.logspace(
						np.log10(self.l_min), 
						np.log10(self.l_max), 
						l_grid_size)
		#
		# Interpolate n and k values using cubic spline
		self.n1_interp = splev(self.l_grid, splrep(self.l, self.n)).clip(min=0.0)
		self.k1_interp = splev(self.l_grid, splrep(self.l, self.k)).clip(min=0.0)
		self.n2_interp = splev(self.l_grid, splrep(other.l, other.n)).clip(min=0.0)
		self.k2_interp = splev(self.l_grid, splrep(other.l, other.k)).clip(min=0.0)
		#
		# Apply the Bruggeman rule for n & k mixing
		self.n_mixed, self.k_mixed = self._call_bruggeman(f_v=[self.f_v, other.f_v])
		#
		# Return a new Dust instance containing mixed n and k 
		mixture = Dust()
		mixture.name = " + ".join([self.name,other.name])
		mixture.l = self.l_grid
		mixture.n = self.n_mixed
		mixture.k = self.k_mixed
		mixture.dens = np.dot([self.dens, other.dens], [self.f_v, other.f_v])

		return mixture

	def __add__(self, comp):
		"""
			This 'magic' method allows dust objects to be mixed via:
			mixture = dust1 + dust2
		"""
		return self.mix(comp) 

	def _call_bruggeman(self, f_v):
		""" This function explicity mixes the n & k indices.
			Based on section 2 from Birnstiel et al. (2018).
			Returns: (n,k)
		 """
		from mpmath import findroot
		
		# Let epsilon = m^2 = (n+ik)^2
		eps = np.array([ 
				[complex(n,k)**2 for (n,k) in zip(self.n1_interp, self.k1_interp)], 
				[complex(n,k)**2 for (n,k) in zip(self.n2_interp, self.k2_interp)]
		])
		#
		# Let f_i be the vol. fractions of each material
		f_i = np.array(f_v)
		#
		# Iterate over wavelenghts
		eps_mean = np.empty(np.shape(self.l_grid)).astype('complex')
		for i, l in enumerate(self.l_grid):
			# Define the expresion for mixing and solve for eps_mean
			expression = lambda x: sum(f_i * ((eps[:,i]-x) / (eps[:,i]+2*x)))
			eps_mean[i] = complex(findroot(expression, complex(0.5,0.5)))
		#
		eps_mean = np.sqrt(eps_mean)

		return eps_mean.real.squeeze(), eps_mean.imag.squeeze()
		
	def get_efficiencies(self, a, algorithm='bhmie', nang=2, coat=None):
		""" Compute the extinction, scattering and absorption
			efficiencies (Q) by calling bhmie or bhcoat.
			Returns: NamedTuple(Q_ext,Q_sca,Q_abs)
		"""
		import progressbar
		from bhmie import bhmie as call_bhmie
		from bhcoat import bhcoat as call_bhcoat
		#
		# Create Dust instance to be returned
		dust = Dust(f'{self.name}')
		#
		if algorithm.lower() == 'bhmie':
			l_grid = self.l
			dust.Q_ext = np.zeros((l_grid.size, a.size))
			dust.Q_sca = np.zeros((l_grid.size, a.size))
			dust.Q_abs = np.zeros((l_grid.size, a.size))
			#
			# Iterate over wavelength
			pb = progressbar.ProgressBar(maxval=l_grid.size)
			pb.start()
			for i, l_ in enumerate(l_grid):
				# Iterate over grain radii
				for j, a_ in enumerate(a):
					# Define the size parameter
					dust.x = 2 * np.pi * a_ / l_
					# Define the complex refractive index (m)
					dust.m = complex(self.n[i], self.k[i])
					#
					bhmie = call_bhmie(dust.x, dust.m, nang)
					#
					dust.Q_ext[i][j] = bhmie[2]
					dust.Q_sca[i][j] = bhmie[3]
					dust.Q_abs[i][j] = dust.Q_ext[i][j] - dust.Q_sca[i][j]
				pb.update(i)

		elif algorithm.lower() == 'bhcoat':
			dust = Dust(f'{dust.name} + coat({coat.name})')
			l_min = np.max([self.l.min(),coat.l.min()])
			l_max = np.min([self.l.max(),coat.l.max()])
			l_grid = np.logspace(np.log10(l_min), np.log10(l_max), 240)
			#
			# Interpolate core and mantle indices
			dust.n_interp = splev(l_grid, splrep(self.l, self.n)).clip(min=0)
			dust.k_interp = splev(l_grid, splrep(self.l, self.k)).clip(min=0)
			coat.n_interp = splev(l_grid, splrep(coat.l, coat.n)).clip(min=0)
			coat.k_interp = splev(l_grid, splrep(coat.l, coat.k)).clip(min=0)
			dust.n = dust.n_interp
			dust.k = dust.k_interp
			#
			dust.Q_ext = np.zeros((l_grid.size, a.size))
			dust.Q_sca = np.zeros((l_grid.size, a.size))
			dust.Q_abs = np.zeros((l_grid.size, a.size))
			#
			# Set the grain sizes for the coat
			a_coat = a * coat.f_v
			# Iterate over wavelength
			pb = progressbar.ProgressBar(maxval=l_grid.size)
			pb.start()
			for i, l_ in enumerate(l_grid):
				# Iterate over grain and mantle radii
				for j, (a_g, a_c) in enumerate(zip(a, a_coat)):
					# Define the size parameter
					dust.x = 2 * np.pi * a_g / l_
					dust.y = 2 * np.pi * a_c / l_
					# Define the complex refractive index (m)
					dust.m_core = complex(dust.n_interp[i], dust.k_interp[i])
					# Interpolate the coat's n & k
					dust.m_mant = complex(coat.n_interp[i], coat.k_interp[i])
					#
					bhcoat = call_bhcoat(dust.x, dust.y, dust.m_core, dust.m_mant)
					#
					dust.Q_ext[i][j] = bhcoat[0]
					dust.Q_sca[i][j] = bhcoat[1]
					dust.Q_abs[i][j] = dust.Q_ext[i][j] - dust.Q_sca[i][j]
				pb.update(i)
		pb.finish()
		#
		dust.l = l_grid
		# Average over grain radii
		print(f'NaN values in Q_ext: {np.isnan(dust.Q_ext).any()}')
		dust.Q_ext = np.nanmean(dust.Q_ext, axis=1)
		dust.Q_sca = np.nanmean(dust.Q_sca, axis=1)
		dust.Q_abs = np.nanmean(dust.Q_abs, axis=1)
		#
		return dust

	def plot_nk(self, show=True, savefig=None):
		""" Plot the interpolated values of the refractive index (n & k). """
		import matplotlib.pyplot as plt
		plt.close()
		f, p = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
		twin_p = p.twinx()
		#
		l = self.l*u.cm.to(u.micron)
		n = p.semilogx(l, self.n, ls='-', color='black')
		k = twin_p.loglog(l, self.k, ls=':', color='black')
		p.text(0.10, 0.95, self.name, fontsize=13, transform=p.transAxes)
		p.legend(n+k, ['n','k'], loc='upper left')
		p.set_xlabel('Wavelength (microns)')
		p.set_ylabel('n')
		twin_p.set_ylabel('k')
		plt.tight_layout()
		#
		if isinstance(savefig, str):
			plt.savefig(savefig)
		if show:
			plt.show()
		
	def plot_Q(self, show=True, savefig=None):
		""" Plot the extinction, scattering & absorption eficiencies.  """
		import matplotlib.pyplot as plt
		plt.close()
		f, p = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
		#
		l = self.l*u.cm.to(micron)
		qext = p.loglog(l, self.Q_ext, ls='--', color='black', label=r'$Q_{\rm\, ext}$')
		qsca = p.loglog(l, self.Q_sca, ls=':', color='black', label=r'$Q_{\rm\, sca}$')
		qabs = p.loglog(l, self.Q_abs, ls='-', color='black', label=r'$Q_{\rm\, abs}$')
		p.text(0.05, 0.95, self.name, fontsize=13, transform=p.transAxes)
		p.legend(loc='best')
		p.set_xlabel('Wavelength (microns)')
		p.set_ylabel('Q')
		p.set_xlim(2.565, 195.694)
		#p.set_ylim(0.39,2.11)
		plt.tight_layout()
		#
		if isinstance(savefig, str):
			plt.savefig(savefig)
		if show:
			plt.show()

		return f"Data to plot is stored in {self.name}.Q_ext, {self.name}.Q_sca & {self.name}.Q_abs"



#########################################################################################

if __name__ == "__main__":
	"""
		The following is a program meant to test the effect of the  Bruggeman rule for mixing
		when creating the input for bhmie and bhcoat.
		
		Instructions (Tommaso):
			You have the following refractive index:
			silicate (S) -> for grain bulk
			ice (I) -> for mantle/coating
			carbonaceus (C) -> for impurities

			First step is to add impurities to the ice, by mixing I+C: use Bruggemann, exactly as Ossenkopf+1994 does.
			Now you have a new refractive index, let's call it IC.

			Test 1:
			Mix again with Bruggemann IC+S and use bhmie.py to compute Qabs, Qext, etc... at different wavelengths.

			Test 2:
			Use IC and S in bhcoating.py to compute Qabs, Qext, etc... at different wavelengths.

			Now compare Qabs, Qext, etc... from Test1 and Test2.
			If you get the same values you can use Bruggemann as in Test 1 for POLARIS.

			Remember that Q* is size- and wavelength-dependent, so do the comparison for different typical sizes within the dust distribution range.

			Note: the refractive index have different validity range, let's assume that they have
			S: [w1_min, w1_max]
			I: [w2_min, w2_max]
			C: [w2_min, w2_max]

			This means that your range will be from w_min = max(w1_min, w2_min, w3_min) to w_max = min(w1_max, w2_max, w3_max)
			you have two options now:
			(a) interpolate over a *fine* regular grid that is in the range [w_min, w_max].
				In this case you should check that all the features (e.g. peaks) are included.
			(b) interpolate over a grid that contains all the points of the original grids S, I, and C
				again in range [w_min, w_max], in this case you will get all the features.
			The difference is that (a) is regular, while (b) not.
	"""

if __name__ == "__main__":


	dirname = pathlib.Path('/home/jz/Documents/software/compute_qabs_tgrassi/data/')

for a_max in [-1]:
		start = time()

		# Create Dust materials
		carbon = Dust('Carbon')
		silicate = Dust('Silicate')
		waterice = Dust('Water ice')
		void = Dust('Void')

		# load n and k from files
		carbon.set_nk(dirname/'eps_carb_P93.dat')
		silicate.set_nk(dirname/'eps_Sil_Oss92.dat')
		#waterice.set_nk(dirname/'eps_H93.dat')
		#void.l = waterice.l
		#void.n = np.ones(waterice.n.shape)
		#void.k = np.zeros(waterice.k.shape)
		
		# Set the mass fraction and bulk density of each component
		carbon.set_density(2.0, vol_fraction=0.11)
		silicate.set_density(2.9, vol_fraction=0.013)
		#waterice.set_density(1.0, vol_fraction=(1-silicate.f_v))
		#void.set_density(0.0, vol_fraction=(1-silicate.f_v))

		# Add impurities to the ice
		print(f'[dustmixer] Mixing silicate and carbon')
		sc = silicate + carbon

		#print(f'[dustmixer] Mixing waterice and carbon')
		#ic = waterice + carbon

		#print(f'[dustmixer] Mixing waterice and silicate')
		#iS = waterice + silicate

		#print(f'[dustmixer] Mixing waterice-carbon and silicate')
		#ics = ic + silicate

		#print(f'[dustmixer] Mixing void and silicate')
		#vs = silicate + void

		# Size dist. for the refractory core (cm)
		a = np.logspace(5e-7, 250e-7, 1)

		sc_mie = sc.get_efficiencies(a, algorithm='bhmie')
		print(sc_mie.shape)
		print(sc_mie)
"""
		# TEST 1: Compute Q_abs with IC + S using bhmie
		print(f'[dustmixer] Computing efficiencies for waterice-carbon + silicon. Using BHMIE.')
		ics_mie = silicate.get_efficiencies(a=a, algorithm='bhmie')

		# TEST 2: Compute Q_abs with IC + S using bhcoat
		print(f'[dustmixer] Computing efficiencies for silicon coated with waterice-carbon. Using BHCOAT.')
		ics_coat = silicate.get_efficiencies(a=a, coat=void, algorithm='bhcoat')

		print(f'ics_coat / ics_mie = {ics_coat.Q_abs[-1] / ics_mie.Q_abs[-1]}')

		#ics_mie.plot_Q(savefig=f'V{V}_bhmie.png')
		#ics_coat.plot_Q(savefig=f'V{V}_bhcoat.png')

		# Plot both algorithms together
		f, p = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
		#p.loglog(ics_coat.l*cm.to(micron), ics_coat.Q_ext, ls='--', color='tab:red')
		#p.loglog(ics_coat.l*cm.to(micron), ics_coat.Q_sca, ls=':', color='tab:red')
		#p.loglog(ics_coat.l*cm.to(micron), ics_coat.Q_abs, ls='-', color='tab:red')

		p.loglog(ics_mie.l*cm.to(micron), ics_mie.Q_ext, ls='--', color='black', label=r'$Q_{\rm ext}$')
		p.loglog(ics_mie.l*cm.to(micron), ics_mie.Q_sca, ls=':', color='black', label=r'$Q_{\rm sca}$')
		p.loglog(ics_mie.l*cm.to(micron), ics_mie.Q_abs, ls='-', color='black', label=r'$Q_{\rm abs}$')

		#plt.xlim(np.min([ics_coat.l,ics_mie.l])*cm.to(micron),np.max([ics_coat.l,ics_mie.l])*cm.to(micron))
		plt.text(0.05,0.20, s='BHCOAT', color='tab:red', transform=p.transAxes, size=15)
		plt.text(0.05,0.15, s='BHMIE', color='black', transform=p.transAxes, size=15)
		plt.text(0.05,0.10, s=f'a min: {a.min()*cm.to(micron):>4} micron', transform=p.transAxes, size=15)
		plt.text(0.05,0.05, s=f'a max: {a.max()*cm.to(micron):>4} micron', transform=p.transAxes, size=15)
		plt.xlabel('Wavelength (microns)', size=15)
		plt.ylabel('Q', size=15)
		plt.legend()
		plt.tight_layout()
		#plt.savefig(f'IC+S_amax_{int(a.max()*1e4)}um.png')
		plt.show()

		print(f'[dustmixer] Total time: {strftime("%H:%M:%S", gmtime(time()-start))}\n')
"""

