"""
	Collection of useful functions for my thesis.
"""
import os, sys, time 
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from astropy.io import ascii, fits
from astropy import units as u
from astropy import constants as c

home = Path.home()

class Observation:
	""" Contains data from real observations. """
	def __init__(self, name='', lam='3mm'):
		self.name = name
		self.data, self.header = fits.getdata(home/f'phd/polaris/sourceB_{lam}.fits', header=True)
		self.data, self.header = fits.getdata(home/f'phd/polaris/sourceB_{lam}.fits', header=True)

	def rescale(self, factor):
		self.data = factor * self.data
	
	def drop_axis(self, drop=True):
		self.data = np.squeeze(self.data) if drop else self.data	

	def fliplr(self, flip=True):
		self.data = np.fliplr(self.data) if flip else self.data

	def flipud(self, flip=True):
		self.data = np.flipud(self.data) if flip else self.data

	# TO DO: bring the function  set_hdr_to_iras16293() into this object


class Bfield:
	def __init__(self):
		# Read bfield data from Fits file
		self.data = fits.getdata(home/'phd/zeusTW/scripts/bfield_faceon.fits')
		# x,y-components from B field
		self.x = B[2]
		self.y = B[3]

	def get_strength(self, normalize=False):
		# Quadrature sum of B_x and B_y
		self.strength = np.sqrt(self.x**2 + self.y**2)

		# Normalize the B field vectors if requested
		self.strength /= self.strength.max() if not normalize else self.strength

		return self.strength
	
	def get_angle(self):
		# Compute the vector angles in the same way as for polarization
		self.angle = np.arctan(self.x / self.y) * u.rad.to(u.deg)
		self.angle += 90
		
		return angle


def print_(string, verbose=None, fname=None, *args):

	# Get the name of the calling function by tracing one level up in the stack
	fname = sys._getframe(1).f_code.co_name if fname is None else fname

	# Check if verbosity state is defined as a global variable
	if verbose is None:
		if 'VERBOSE' in globals() and VERBOSE:
			verbose = True

	if verbose:
		print(f"[{fname}] {string}", *args)


def write_fits(filename, data, header, overwrite=True, verbose=False):
	caller = sys._getframe(1).f_code.co_name

	if filename != '':
		if overwrite and os.path.exists(filename):
			print_('Overwriting file ...', verbose=verbose, fname=caller)
			os.remove(filename)

		fits.HDUList(fits.PrimaryHDU(data=data, header=header)).writeto(filename) 	
		print_(f"Written file {filename}", verbose=verbose, fname=caller)


def elapsed_time(runtime, verbose=False):
	# Get the name of the calling function by tracing one level up in the stack
	caller = sys._getframe(1).f_code.co_name

	run_time = time.strftime("%H:%M:%S", time.gmtime(runtime))

	print_(f"Elapsed time: {run_time}", verbose=verbose, fname=caller)
		

def set_hdr_to_iras16293B(hdr, wcs='deg', spec_axis=False, stokes_axis=False, for_casa=False, verbose=False):
	"""
		Adapt the header to match that of the ALMA observation of IRAS16293-2422B.
		Data from Maureira et al. (2020).
	"""

	# Set the sky WCS to be in deg by default
	# and delete the extra WCSs
	if all([spec_axis, stokes_axis]):
		hdr['NAXIS'] = 4
	elif any([spec_axis, stokes_axis]):
		hdr['NAXIS'] = 3
	else:
		hdr['NAXIS'] = 3 if for_casa else 2

	keys = ['NAXIS','CDELT','CUNIT','CRPIX','CRVAL','CTYPE']

	WCS = {
		'deg':'A',
		'AU' : 'B',
		'pc' : 'C'
	}

	# TO DO: tell it to copy cdelt1A to cdelt1 only if more than one wcs exists.
	# Because, if no cdelt1A then it will set cdelt1 = None

	for n in [1,2]:
		for k in keys[1:]:
			hdr[f'{k}{n}'] = hdr.get(f'{k}{n}{WCS[wcs]}', hdr.get(f'{k}{n}'))
			for a in WCS.values():
				hdr.remove(f'{k}{n}{a}', ignore_missing=True)

	for n in [3,4]:
		for key in keys:
			hdr.remove(f'{key}{n}', ignore_missing=True)

	# Adjust the header to match obs. from IRAS16293-2422B
	hdr['CUNIT1'] = 'deg'
	hdr['CTYPE1'] = 'RA---SIN'
	hdr['CRPIX1'] = 1 + hdr.get('NAXIS1') / 2
	hdr['CDELT1'] = hdr.get('CDELT1')
	hdr['CRVAL1'] = np.float64(248.0942916667)
	hdr['CUNIT2'] = 'deg'
	hdr['CTYPE2'] = 'DEC--SIN'
	hdr['CRPIX2'] = 1 + hdr.get('NAXIS2') / 2
	hdr['CDELT2'] = hdr.get('CDELT2')
	hdr['CRVAL2'] = np.float64(-24.47550000000)

	# Add spectral axis if required
	if spec_axis:
		# Convert the observing wavelength from the header into frequency
		wls = {
			'0.0013': np.float64(230609583076.92307),
			'0.003': np.float64(99988140037.24495),
			'0.007': np.float64(42827493999.99999)
		}
		hdr['NAXIS3'] = 1
		hdr['CTYPE3'] = 'FREQ'
		hdr['CRVAL3'] = wls[str(hdr.get('HIERARCH WAVELENGTH1', '0.0013'))]
		hdr['CRPIX3'] = np.float64(0.0)
		#hdr['CDELT3'] = np.float64(2.000144770049E+09)
		hdr['CDELT3'] = np.float64(3.515082631882E+10)
		hdr['CUNIT3'] = 'Hz'
		hdr['RESTFRQ'] = hdr.get('CRVAL3')
		hdr['SPECSYS'] = 'LSRK'

	# Add stokes axis if required
	if stokes_axis:
		hdr['NAXIS4'] = 1
		hdr['CTYPE4'] = 'STOKES'
		hdr['CRVAL4'] = np.float32(1)
		hdr['CRPIX4'] = np.float32(0)
		hdr['CDELT4'] = np.float32(1)
		hdr['CUNIT4'] = ''
						
	# Add missing keywords (src: http://www.alma.inaf.it/images/ArchiveKeyworkds.pdf)
	hdr['BTYPE'] = 'Intensity'
	hdr['BUNIT'] = 'Jy/pixel'
	hdr['BZERO'] = 0.0
	hdr['RADESYS'] = 'ICRS'

	return hdr


def create_cube(filename='polaris_detector_nr0001.fits.gz', outfile='', wcs='deg', spec_axis=False, stokes_axis=False, add_selfscat=False, for_casa=True, overwrite=False, verbose=False):
	"""
	Retrieves data and header from filename and add necessary keywords to the cube.
	NOTE: Wildcards are allowed by the infile argument. Thanks to glob.
	"""
	from glob import glob

	start_time = time.time()
	pwd = os.getcwd()

	# Set a global verbose if verbose is enabled
	if verbose:
		global VERBOSE
		VERBOSE = verbose

	# Read data
	data, hdr = fits.getdata(filename, header=True)
	
	if all([spec_axis, stokes_axis]):
		I = np.array([[data[0][0]]])
	elif any([spec_axis, stokes_axis]):
		I = np.array([data[0][0]])
	else:
		I = data[0][0]

	# Edit the header to match the observation from IRAS16293B
	hdr = set_hdr_to_iras16293B(hdr, wcs=wcs, spec_axis=spec_axis, stokes_axis=stokes_axis, for_casa=for_casa)

	# Add emission by self-scattering if required
	if add_selfscat:
		if 'scattered emission' in hdr.get('ETYPE', ''):
			print_('You are adding self-scattered flux to the self-scattered flux. Not gonna happen.')
			I_ss = np.zeros(I.shape)
		
		elif 'thermal emission' in hdr.get('ETYPE', ''):
			print_('Adding self-scattered flux to the thermal flux.')
			if 'dust_polarization' in pwd:
				pwd = pwd.replace('pa/','')
				selfscat_file = pwd.replace('dust_polarization','dust_scattering')

				# Change the polarized stokes I for the unpolarized stokes I
				I_th_unpol = pwd.replace('dust_polarization','dust_emission')
				try:
					i = fits.getdata(i_th_unpol+'/'+filename)[0][0]
				except OSError:
					raise FileNotFoundError(f'File with unpolarized thermal flux does not exist.\n\
											File: {i_th_unpol+"/"+filename}')

			else:
				selfscat_file = pwd.replace('dust_emission','dust_scattering')

			try:
				I_ss = fits.getdata(selfscat_file + '/' + filename)[0][0]
			except OSError:
				raise FileNotFoundError(f'File with self-scattered flux does not exist.\n\
										File: {selfscat_file+"/"+filename}')

	else:
		I_ss = np.zeros(I.shape)

	# Add all sources of emission
	I = I + I_ss

	# Write data to fits file
	write_fits(outfile, I, hdr, overwrite, verbose)

	# Print the time taken by the function
	elapsed_time(time.time()-start_time, verbose)

	return data


def read_sph(snapshot='snap_541.dat'):
	"""
	Notes:

	Assumes your binary file is formatted with a header stating the quantities,
	assumed to be f4 floats. May not be widely applicable.

	For these snapshots, the header is of the form... 

	# id t x y z vx vy vz mass hsml rho T u

	where:

	id = particle ID number
	t = simulation time [years]
	x, y, z = cartesian particle position [au]
	vx,vy,vz = cartesian particle velocity [need to check units]
	mass = particle mass [ignore]
	hmsl = particle smoothing length [ignore]
	rho = particle density [g/cm3]
	T = particle temperature [K]
	u = particle internal energy [ignore]
	"""

	with open(snapshot, 'rb') as f:
		names = f.readline()[1:].split()
		data = np.fromstring(f.read()).reshape(-1, len(names))
		data = data.astype('f4')

	return data


def radmc3d_data(file_, npix=300, sizeau=50, distance=3.086e18*u.m.to(u.cm)):
	"""
	Function to read image files resulting from an RT with RADMC3D.
	"""

	print(f'[radmc_data] Reading file {file_.split("/")[-1]}')
	img = ascii.read(file_, data_start=5, guess=False)['1']

	# Make a squared map
	img = img.reshape(npix,npix)

	# Rescale to Jy/sr
	img = img * (u.erg*u.s**-1*u.cm**-2*u.Hz**-1*u.sr**-1).to(u.Jy*u.sr**-1)

	# Obtain the pixel size
	pixsize = (sizeau/npix) * u.au.to(u.cm)

	# Convert sr into pixels (Jy/sr --> Jy/pixel)
	img = img * ((pixsize)**2 / (distance)**2)

	return img


def fill_gap(filename, outfile=None, x1=143, x2=158, y1=143, y2=157, threshold=1e-5, incl='0deg', savefile=True):
	"""
		Fill the central gap from polaris images with the peak flux
	"""
 
	# Read data
	d, hdr = fits.getdata(filename, header=True)
	d = d[0][0]
	full = d 
	gap = np.where(d[x1:x2,y1:y2] < threshold, d.max(), d[x1:x2,y1:y2])
	full[x1:x2, y1:y2] = gap 
	plt.imshow(full, cmap='magma');plt.colorbar();plt.show()

	if outfile is None:
		outfile = filename.split('.fits')[0]+'_nogap.fits'

	if savefile:
		fits.writeto(outfile, data=full, header=hdr, overwrite=True)


def circular_mask(shape, c, r, angle_range):
    """
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    """

    x,y = np.ogrid[:shape[0],:shape[1]]
    cx, cy = c
    a_i, a_f = np.deg2rad(angle_range)

    # Ensure stop angle > start angle
    if a_f < a_i:
            a_f += 2*np.pi

    # Convert cartesian --> polar coordinates
    r2 = (x-cx)**2 + (y-cy)**2
    theta = np.arctan2(x-cx,y-cy) - a_i

    # Wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # Circular mask
    circmask = r2 <= r*r

    # Angular mask
    anglemask = theta <= (a_f-a_i)

    return circmask * anglemask


def radial_average(data, step=1):
	"""
		Computes the radial average of a 2D array by averaging the values 
		within consecutive concentric circumferences from the border to
		the center.
	"""
	# Read data from fits file if filename is provided
	if isinstance(data, str):
		data, hdr = fits.getdata(data, header=True)

	# Drop empty axes
	data = data.squeeze()
	
	# Get the center of the array
	map_radius_x = int(data[0].size / 2)
	map_radius_y = int(data[1].size / 2)
	center = (map_radius_x, map_radius_y)

	# Masks are created from the border to the center because otherwise
	# all masks other than r=0 would already be NaN.
	averages = np.zeros(np.arange(map_radius_x).shape)
	radii = np.arange(0, map_radius_x, step)[::-1]

	for i,r in enumerate(radii):
		mask = circular_mask(data.shape, center, r, angle_range=(0,360))
		data[~mask] = float('nan')
		averages[i] = np.nanmean(data)

	# Reverse the averages to be given from the center to the border
	return averages[::-1]


def stats(filename, slice=None, verbose=False):
	"""
		Compute the statistics of a file.
	"""
	
	# Read data
	if isinstance(filename, str):
		data, hdr = fits.getdata(filename, header=True)

		if isinstance(slice, int):
			data = data[slice]
		elif isinstance(slice, list) and len(slice) == 2:
			data = data[slice[0], slice[1]] 
	else:
		data = np.array(filename)
	
	# Set the relevant quantities
	stat = {
		'max': data.max(),
		'mean': data.mean(),
		'min': data.min(),
		'std': data.std(),
		'maxpos': maxpos(data),
		'minpos': minpos(data)
	}

	# Print statistics if verbose enabled
	for label, value in stat.items():
		print_(f'{label}: {value}', verbose) 
	
	return stat


def maxpos(data):
	"""
		Return a tuple with the coordinates of a N-dimensional array.
	"""
	# Read data from fits file if data is string
	if isinstance(data, str):
		data = fits.getdata(data)

	# Remove empty axes
	data = np.squeeze(data)
	
	return np.unravel_index(data.argmax(), data.shape)
	
	
def minpos(data):
	"""
		Return a tuple with the coordinates of a N-dimensional array.
	"""
	# Read data from fits file if data is string
	if isinstance(data, str):
		data = fits.getdata(data)

	# Remove empty axes
	data = np.squeeze(data)
	
	return np.unravel_index(data.argmin(), data.shape)


def add_comment(filename, comment):
	"""
		Read in a fits file and add a new keyword to the header. 
	""" 
	data, hdr = fits.getdata(filename, header=True)

	header['NOTE'] = comment

	write_fits(filename, data=data, header=header, overwrite=True)


def plot_map(filename, savefig=None, rescale=1, cblabel='', verbose=True, *args, **kwargs):
	"""
		Plot a fits file using the APLPy library.
	"""
	from aplpy import FITSFigure	
	
	fig = FITSFigure(filename, rescale=rescale)
	fig.show_colorscale(*args, **kwargs)

	# Auto set the colorbar label if not provided
	if cblabel == '' and rescale == 1:
		try:
			hdr = fits.getheader(filename)
			cblabel = hdr.get('BTYPE')
		except Exception as e:
			print_(e)
			cblabel = ''
		
	elif cblabel == '' and rescale == 1e3:
		cblabel = 'mJy/beam'

	elif cblabel == '' and rescale == 1e6:
		cblabel = r'$\mu$Jy/beam'
	
	# Colorbar
	fig.add_colorbar()
	fig.colorbar.set_location('top')
	fig.colorbar.set_axis_label_text(cblabel)
	fig.colorbar.set_axis_label_font(size=15, weight=15)
	fig.colorbar.set_font(size=15, weight=15)
	
	# Frame and ticks
	fig.frame.set_color('black')
	fig.frame.set_linewidth(1.2)
	fig.ticks.set_color('black')
	fig.ticks.set_linewidth(1.2)
	fig.ticks.set_length(6)
	fig.ticks.set_minor_frequency(5)

	# Scalebar
	# TO DO: aplpy claims alma images have no celestial WCS, no scalebar allowed
	if 'alma' not in filename:
		D = 141 * u.pc
		scalebar = 50 * u.au
		scalebar_ = (scalebar.to(u.cm) / D.to(u.cm)) * u.rad.to(u.arcsec)
		fig.add_scalebar(scalebar_ * u.arcsec)
		fig.scalebar.set_color('white')
		fig.scalebar.set_corner('bottom right')
		fig.scalebar.set_font(size=23)
		fig.scalebar.set_linewidth(3)
		fig.scalebar.set_label(f'{int(scalebar.value)} {scalebar.unit}')

	if isinstance(savefig, str) and len(savefig) > 0:
		fig.save(savefig)

	return fig




def polarization_map(filename='polaris_detector_nr0001.fits.gz', render='intensity', wcs='deg', rotate=90, step=20, scale=50, fmin=None, fmax=None, savefig=None, show=True, vector_color='tab:purple', add_thermal=False, add_selfscat=False, add_bfield=False, const_bfield=False, const_pfrac=False, verbose=True, *args, **kwargs):
	"""
		Extract I, Q, U and V from the polaris output
		and create maps of Pfrac and Pangle using APLpy.
	"""
	from aplpy import FITSFigure
	
	start = time.time()
	
	# Enable verbosity
	global VERBOSE
	VERBOSE = verbose 
	
	# Store the current path
	pwd = os.getcwd()

	# Read the output from polaris
	data, hdr = fits.getdata(filename, header=True)
	
	I = data[0][0]
	Q = data[1][0]
	U = data[2][0]
	try:
		tau = data[4][0]
	except:
		tau = np.zeros(I.shape)

	# Add thermal emission to the self-scattered emission if required
	if add_thermal:
		if 'thermal emission' in hdr.get('ETYPE', ''):
			print_('You are adding thermal flux to the thermal flux.')
			print_('Not gonna happen.')
			I_th = np.zeros(I.shape)
		
		elif 'scattered emission' in hdr.get('ETYPE', ''):
			print_('Adding thermal flux to the self-scattered flux.')
			thermal_file = pwd.replace('dust_scattering','dust_emission')
			try:
				I_th = fits.getdata(thermal_file + '/' + filename)[0][0]
			except OSError:
				raise FileNotFoundError(f'File with thermal flux does not exist.\n\
										File: {thermal_file+"/"+filename}')

	elif isinstance(add_thermal, str):
		try:
			I_th = fits.getdata(add_thermal)
		except OSError:
			raise FileNotFoundError(f'File with thermal flux does not exist.\n'+
									'File: {add_thermal}')
	
	else:
		I_th = np.zeros(I.shape)

	# Add self-scattered emission to the thermal emission if required
	if add_selfscat:
		if 'scattered emission' in hdr.get('ETYPE', ''):
			print_('You are adding self-scattered flux to the self-scattered flux. Not gonna happen.')
			I_ss = np.zeros(I.shape)
		
		elif 'thermal emission' in hdr.get('ETYPE', ''):
			print_('Adding self-scattered flux to the thermal flux.')
			if 'dust_polarization' in pwd:
				pwd = pwd.replace('pa/','')
				selfscat_file = pwd.replace('dust_polarization','dust_scattering')

				# Change the polarized stokes I for the unpolarized stokes I
				I_th_unpol = pwd.replace('dust_polarization','dust_emission')
				try:
					I = fits.getdata(I_th_unpol+'/'+filename)[0][0]
				except OSError:
					raise FileNotFoundError(f'File with thermal flux does not exist.\n\
											File: {I_th_unpol+"/"+filename}')

			else:
				selfscat_file = pwd.replace('dust_emission','dust_scattering')

			try:
				I_ss = fits.getdata(selfscat_file + '/' + filename)[0][0]
			except OSError:
				raise FileNotFoundError(f'File with self-scattered flux does not exist.\n\
										File: {selfscat_file+"/"+filename}')

	elif isinstance(add_thermal, str):
		try:
			I_ss = fits.getdata(add_selfscat)
		except OSError:
			raise FileNotFoundError(f'File with self-scattered flux does not exist.\n\
									File: {add_selfscat}')

	else:
		I_ss = np.zeros(I.shape)

	# Add all sources of emission
	I = I + I_th + I_ss

	# Define the polarization fraction and angle
	pfrac = np.sqrt(U**2 + Q**2) / I   if not const_pfrac else np.ones(Q.shape)
	pangle = 0.5 * np.arctan(U/Q) * u.rad.to(u.deg)
	
	# Edit the header to match the observation from IRAS16293B
	hdr = set_hdr_to_iras16293B(hdr)

	# Write quantities into fits files
	quantities = {
		'I': I,
		'Q': Q,
		'U': U,
		'tau': tau,
		'pfrac': pfrac,
		'pangle': pangle
	}
	for f, d in quantities.items():
		write_fits(f+'.fits', d, hdr)

	# Select the quantity to plot
	if render.lower() in ['intensity', 'i']:
		figname = 'I.fits'
		cblabel = r'Stokes I ($\mu$Jy/pixel)'
		# Rescale to micro Jy/px
		rescale = 1e6
	
	elif render.lower() in ['q']:
		figname = 'Q.fits'
		cblabel = r'Stokes Q ($\mu$Jy/pixel)'
		# Rescale to micro Jy/px
		rescale = 1e6
	
	elif render.lower() in ['u']:
		figname = 'U.fits'
		cblabel = r'Stokes U ($\mu$Jy/pixel)'
		# Rescale to micro Jy/px
		rescale = 1e6
	
	elif render.lower() in ['tau', 'optical depth']:
		figname = 'tau.fits'
		cblabel = r'Optical depth'
		rescale = 1
	
	elif render.lower() in ['pfrac', 'p', 'pol']:
		figname = 'pfrac.fits'
		cblabel = r'Polarization fraction (%)'
		# Rescale fraction to percentage
		rescale = 100

	else:
		raise ValueError("Wrong value for render. Must be 'intensity' or 'pfrac'.")


	# Set vmin to a fraction of the max
	if fmin is not None and fmin > 0 and fmin <= 1:
		vmin = rescale * fmin * fits.getdata(figname).max()
	else:
		vmin = None

	# Set vmax to a fraction of the max 
	if fmax is not None and fmax > 0 and fmax <= 1:
		vmax = rescale * fmax * fits.getdata(figname).max()
	else:
		vmax = None

	# Plot the render quantity a colormap
	fig = plot_map(figname, rescale=rescale, cblabel=cblabel, vmax=vmax, vmin=vmin, *args, **kwargs)

	# Temporal Patch. TO DO: put it right
	if 'dust_scattering' in pwd:
		rotate = 0

	# Add polarization vectors
	fig.show_vectors(
		'pfrac.fits', \
		'pangle.fits', \
		step=step, \
		scale=scale, \
		rotate=rotate, \
		color=vector_color,\
		layer='pol_vectors'
	)
	
	# Add B-field vectors
	# TO DO: ADD A LABEL TO INDICATE WHAT VECTORS ARE WHAT 
	if add_bfield:
		print_('Adding magnetic field lines.')

		B = Bfield()

		write_fits('B.fits', data=B.get_strength(const_pfrac), header=hdr)
		write_fits('Bangle.fits', data=B.get_angle(), header=hdr)

		fig.show_vectors(
			'B.fits', \
			'Bangle.fits', \
			step=step, \
			scale=scale, \
			rotate=0, \
			color='tab:green',\
			zorder=1,\
			layer='B_vectors'
		)

	if show:
		plt.show()

	if isinstance(savefig, str) and len(savefig) > 0:
		fig.save(savefig)
	
	elapsed_time(time.time() - start, verbose)

	return fig


def plot_spectral_index(lam1, lam2): 
	""" Plot the spectral index between observations at two wavelengths."""
	pass


def horizontal_cuts(angles, add_obs=False, scale_obs=None, axis=0, lam='3mm', amax='100um', prefix='', show=True, savefig=None, *args, **kwargs):
	""" Self-explanatory.
	"""
	def angular_offset(d, hdr):
		""" Calculate the angular offset (assumes angular scale is in degrees)
		"""
		cdelt1 = hdr.get('CDELT1') * u.deg.to(u.arcsec)
		naxis1 = hdr.get('NAXIS1')
		FOV = naxis1 * cdelt1

		# Find the peak in the image and cut along the given axis
		cut = maxpos(d)[axis]
		if axis == 0:
			cut = d[cut, :]
		elif axis == 1:
			cut = d[:, cut]	

		# Offset from the center of the image
		offset = np.linspace(-FOV/2, FOV/2, naxis1)
		
		# Find the peak position along the cut
		cut_peak = np.argmax(cut)

		# Shift the angular offset to be centered on the peak
		offset = offset - offset[cut_peak]

		return offset, cut

	# Set the path prefix as a Path object, if provided
	prefix = Path(prefix)
	
	# Plot the cut from the real observation if required
	if add_obs:
		# Read data
		obs = Observation(name='IRAS16293B', lam=lam)
		
		# Drop empty axes, flip and rescale
		obs.rescale(1e3)
		obs.drop_axis()
		obs.fliplr()

		label = f'{obs.name} (x{scale_obs:.1})' if scale_obs else obs.name 
		plt.plot(*angular_offset(obs.data, obs.header), label=label, color='black', ls='-.', *args, **kwargs)

	# Plot the cuts from the simulated observations for every inclination angle
	for angle in [f'{i}deg' for i in angles]:
		# Read data
		filename = prefix/f'{angle}/data/{lam}_{angle}_a{amax}_alma.fits'
		data, hdr = fits.getdata(filename, header=True)

		# Drop empty axes. Flip and rescale
		data = 1e3 * np.fliplr(np.squeeze(data))

		plt.plot(*angular_offset(data, hdr), label=f'{angle}', *args, **kwargs)

	# Customize the plot
	plt.axvline(0, lw=1, ls='--', alpha=0.5, color='grey')
	plt.legend(ncol=1, loc='upper left')
	plt.title('Horizontal cut for different inclination angles.')
	plt.xlabel('Angular offset (arcseconds)')
	plt.ylabel('mJy/beam')
	plt.xlim(-0.35, 0.35)

	if isinstance(savefig, str) and len(savefig)>0:
		plt.savefig(savefig)

	if show:
		plt.show()


def get_polaris_temp(binfile='grid_temp.dat'):
	""" Read the binary output from a Polaris dust heating 
		simulation and return the dust temperature field.
	"""
	import struct 

	with open(binfile, 'rb') as f:
		# Read grid ID
		ID = struct.unpack("H", f.read(2))

		# Read N quantities
		n, = struct.unpack("H", f.read(2))
		for q in range(n):
			struct.unpack("H", f.read(2))

		# Read radial boundaries 
		r_in = struct.unpack("d", f.read(8))
		r_out = struct.unpack("d", f.read(8))

		# Read number of cells on each axis
		n_r, = struct.unpack("H", f.read(2))
		n_t, = struct.unpack("H", f.read(2))
		n_p, = struct.unpack("H", f.read(2))
		n_cells = n_r * n_t * n_p

		# Read shape parameters (0 if cell borders are given)
		struct.unpack("d", f.read(8))
		struct.unpack("d", f.read(8))
		struct.unpack("d", f.read(8))

		# Read r coords
		for r in range(n_r):
			struct.unpack("d", f.read(8))
		# Read n_t coords
		for t in range(n_t):
			struct.unpack("d", f.read(8))
		# Read n_p coords
		for p in range(n_p):
			struct.unpack("d", f.read(8))

		temp = np.zeros(n_cells)
		# Iterate over cells
		for c in range(n_cells):
			# Read first 6 quantities
			for q in range(6):
				struct.unpack("d", f.read(8))
			# Read temperature
			temp[c], = struct.unpack("d", f.read(8))

			# Move the pointer till the end of the row
			for q in range(3):
				struct.unpack("d", f.read(8))

		return temp
