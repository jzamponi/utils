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
		'std': data.std()
	}

	# Print statistics if verbose enabled
	for label, value in stat.items():
		print_(f'{label}: {value}', verbose) 
	
	return stat


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

	for n in [1,2]:
		for k in keys[1:]:
			hdr[f'{k}{n}'] = hdr[f'{k}{n}{WCS[wcs]}']
			for a in WCS.values():
				del hdr[f'{k}{n}{a}']

	for n in [3,4]:
		for key in keys:
			del hdr[f'{key}{n}']

	# Adjust the header to match obs. from IRAS16293-2422B
	hdr['CUNIT1'] = 'deg'
	hdr['CTYPE1'] = 'RA---SIN'
	hdr['CRPIX1'] = 1 + hdr['NAXIS1'] / 2
	hdr['CRVAL1'] = np.float64(248.0942916667)
	hdr['CUNIT2'] = 'deg'
	hdr['CTYPE2'] = 'DEC--SIN'
	hdr['CRPIX2'] = 1 + hdr['NAXIS2'] / 2
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
		hdr['CRVAL3'] = wls[str(hdr['HIERARCH WAVELENGTH1'])]
		hdr['CRPIX3'] = np.float64(0.0)
		hdr['CDELT3'] = np.float64(2000144770.0491333)
		hdr['CUNIT3'] = 'Hz'
		hdr['RESTFRQ'] = hdr['CRVAL3']
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
		if 'scattered emission' in hdr['ETYPE']:
			print_('You are adding self-scattered flux to the self-scattered flux. Not gonna happen.')
			I_ss = np.zeros(I.shape)
		
		elif 'thermal emission' in hdr['ETYPE']:
			print_('Adding self-scattered flux to the thermal flux.')
			if 'dust_polarization' in pwd:
				pwd = pwd.replace('pa/','')
				selfscat_file = pwd.replace('dust_polarization','dust_scattering')

				# Change the polarized stokes I for the unpolarized stokes I
				I_th_unpol = pwd.replace('dust_polarization','dust_emission')
				I = fits.getdata(I_th_unpol+'/'+filename)[0][0]

			else:
				selfscat_file = pwd.replace('dust_emission','dust_scattering')

			I_ss = fits.getdata(selfscat_file + '/' + filename)[0][0]

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


def radial_average():
	"""
		Computes the radial average of a 2D array by averaging the values 
		within increasing concentric circumferences.
	"""
	pass

def maxpos():
	"""
		Return a tuple with the coordinates of a N-dimensional array.
	"""

def create_a_decorator():
	"""
		Create a function decorator that wraps the function within a try catch block
		to control a KeyboardInterrupt.
	"""
	pass


def plot_map(filename, savefig=None, rescale=1, cblabel='', verbose=True, *args, **kwargs):
	"""
		Plot a fits file using the APLPy library.
	"""
	from aplpy import FITSFigure	
	
	print(f'args:')
	print(*args)

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
	tau = data[4][0]

	# Add thermal emission to the self-scattered emission if required
	if add_thermal:
		if 'thermal emission' in hdr['ETYPE']:
			print_('You are adding thermal flux to the thermal flux.')
			print_('Not gonna happen.')
			I_th = np.zeros(I.shape)
		
		elif 'scattered emission' in hdr['ETYPE']:
			print_('Adding thermal flux to the self-scattered flux.')
			thermal_file = pwd.replace('dust_scattering','dust_emission')
			I_th = fits.getdata(thermal_file + '/' + filename)[0][0]

	elif isinstance(add_thermal, str):
		I_th = fits.getdata(add_thermal)
	
	else:
		I_th = np.zeros(I.shape)

	# Add self-scattered emission to the thermal emission if required
	if add_selfscat:
		if 'scattered emission' in hdr['ETYPE']:
			print_('You are adding self-scattered flux to the self-scattered flux. Not gonna happen.')
			I_ss = np.zeros(I.shape)
		
		elif 'thermal emission' in hdr['ETYPE']:
			print_('Adding self-scattered flux to the thermal flux.')
			if 'dust_polarization' in pwd:
				pwd = pwd.replace('pa/','')
				selfscat_file = pwd.replace('dust_polarization','dust_scattering')

				# Change the polarized stokes I for the unpolarized stokes I
				I_th_unpol = pwd.replace('dust_polarization','dust_emission')
				I = fits.getdata(I_th_unpol+'/'+filename)[0][0]

			else:
				selfscat_file = pwd.replace('dust_emission','dust_scattering')

			I_ss = fits.getdata(selfscat_file + '/' + filename)[0][0]

	elif isinstance(add_thermal, str):
		I_ss = fits.getdata(add_selfscat)

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
		B = fits.getdata('/home/jz/phd/zeusTW/scripts/bfield_faceon.fits')
		B_x, B_y = B[2], B[3]

		# Quadrature sum of B_x and B_y
		B_strength = np.sqrt(B_x**2 + B_y**2)

		# Normalize the B field vectors
		B_strength /= B_strength.max() if not const_bfield else B_strength

		# Compute the vector angles in the same way as for polarization
		B_angle = np.arctan(B_x / B_y) * u.rad.to(u.deg)
		B_angle += 90

		write_fits('B.fits', data=B_strength, header=hdr)
		write_fits('Bangle.fits', data=B_angle, header=hdr)

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


def horizontal_cuts(angles, add_obs=False, scale_obs=None, prefix='', show=True, savefig=None, *args, **kwargs):
	""" Self-explanatory.
	"""
	def angular_offset(d, hdr):
		""" Calculate the angular offset (assumes angular scale is in degrees)
		"""
		cdelt1 = hdr.get('CDELT1') * u.deg.to(u.arcsec)
		naxis1 = hdr.get('NAXIS1')
		FOV = naxis1 * cdelt1

		if hdr.get('OBJECT') == 'IRAS_16293-2422B':
			cut = d[332, :] 
			cut = scale_obs * cut if scale_obs is not None else cut

		else:
			cut = d[145, :]

		# Offset from the center of the image
		offset = np.linspace(-FOV/2, FOV/2, naxis1)
		
		# Find the peak position
		peakpos = np.argmax(cut)

		# Change the offset to be from the peak
		offset = offset - offset[peakpos]

		return offset, cut

	home = Path.home()

	# Set the path prefix as a Path object, if provided
	prefix = Path(prefix)
	
	# Plot the cut from the real observation if required
	if add_obs:
		# Read data
		obs, hdr_obs = fits.getdata(home/'phd/polaris/sourceB.image.fits', header=True)
	
		# Drop empty axes, flip and rescale
		obs = 1e3 * np.fliplr(np.squeeze(obs))

		label = f'IRAS16293B (x{scale_obs:.1})' if scale_obs else 'IRAS16293B'
		plt.plot(*angular_offset(obs, hdr_obs), label=label, color='black', ls='-.', *args, **kwargs)

	# Plot the cuts from the simulated observations for every inclination angle
	for angle in [f'{i}deg' for i in angles]:
		# Read data
		filename = prefix/f'{angle}/data/3mm_{angle}_a100um_alma.fits'
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
