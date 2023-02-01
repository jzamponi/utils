"""
    Collection of useful functions for my thesis.
"""
import os
import sys
import time
from functools import wraps
from pathlib import Path, PosixPath

import numpy as np

import matplotlib.pyplot as plt

from astropy.io import ascii, fits
from astropy import units as u
from astropy import constants as c

home = Path.home()
pwd = Path(os.getcwd())


class color:
    red = "\033[91m"
    bold = "\033[1m"
    none = "\033[0m"


class Observation:
    """ Contains data from real observations. """

    from collections import namedtuple

    def __init__(self, lam, source="iras16293"):
        self.source = source.lower()

        if lam == '1.3mm': 
            self.band = 'band6'
            self.filename = 'sourceB_1.3mm.fits'
        elif lam == '3mm': 
            self.band = 'band3'
            self.filename = 'sourceB_3mm.fits'
        elif lam == '7mm': 
            self.band = 'bandQ'
            self.filename = 'stokes_I_zoom.fits'
        elif lam == '18mm': 
            self.band = 'bandKu'
            self.filename = 'stokes_I_zoom_rob0.fits'
        elif lam == 'alpha':
            self.band = ''
            self.filename = 'sourceB_spectral_index_band3-6.fits'
        else:
            raise ValueError(f'No observation available for lam = {lam}')

        self.data, self.header = fits.getdata(
            home/"phd/observations"/self.source/self.band/self.filename, 
            header=True
        )
        
    def rescale(self, factor):
        self.data = factor * self.data

    def drop_axis(self, drop=True):
        self.data = np.squeeze(self.data) if drop else self.data

    def fliplr(self, flip=True):
        self.data = np.fliplr(self.data) if flip else self.data

    def flipud(self, flip=True):
        self.data = np.flipud(self.data) if flip else self.data


class Header:
    """ Handles header transformations. """
    # TO DO: bring the function  set_hdr_to_iras16293() into this object
    # and also the as_namedtuple feature from the Observation class
    def __init__(self, hdr):
        self.hdr = hdr


class Bfield:
    def __init__(self):
        # Read bfield data from Fits file
        self.data = fits.getdata(home / "phd/zeusTW/scripts/bfield_faceon.fits")
        # x,y-components from B field
        self.x = self.data[2]
        self.y = self.data[3]

    def get_strength(self, normalize=False):
        # Quadrature sum of B_x and B_y
        self.strength = np.sqrt(self.x ** 2 + self.y ** 2)

        # Normalize the B field vectors if requested
        self.strength /= self.strength.max() if not normalize else self.strength

        return self.strength

    def get_angle(self):
        # Compute the vector angles in the same way as for polarization
        self.angle = np.arctan(self.x / self.y) * u.rad.to(u.deg)
        self.angle += 90

        return self.angle


def print_(string, verbose=True, bold=False, red=False, fname=None, *args, **kwargs):

    # Get the name of the calling function by tracing one level up in the stack
    fname = sys._getframe(1).f_code.co_name if fname is None else fname

    # Check if verbosity state is defined as a global variable
    if verbose is None:
        if "VERBOSE" in globals() and VERBOSE:
            verbose = True

    if verbose:
        if bold:
            print(f"{color.bold}[{fname}] {string} {color.none}", flush=True, 
                *args, **kwargs)
        elif red:
            print(f"{color.red}[{fname}] {string} {color.none}", flush=True, 
                *args, **kwargs)
        else:
            print(f"[{fname}] {string}", flush=True, 
                *args, **kwargs)


def write_fits(filename, data, header=None, overwrite=True, verbose=False):
    # Get the name of the calling function by tracing one level up in the stack
    caller = sys._getframe(1).f_code.co_name

    if filename != "":
        if overwrite and os.path.exists(filename):
            os.remove(filename)

        fits.HDUList(fits.PrimaryHDU(data=data, header=header)).writeto(filename)
        print_(f"Written file {filename}", verbose=verbose, fname=caller)


def elapsed_time(caller):
    """ Decorator designed to print the time taken by a functon. """
    # TO DO: Find a way to forward verbose from the caller even when
    # is not provided explicitly, so that it takes the default value 
    # from the caller.

    # Forward docstrings to the caller function
    @wraps(caller)

    def wrapper(*args, **kwargs):
        # Measure time before it runs
        start = time.time()

        try:
            # Execute the caller function
            f = caller(*args, **kwargs)
        except KeyboardInterrupt:
            # Print the time even after sending SIGINT (Ctrl+C)
            print_('\nExecution interrupted by user.')
        except Exception as e:
            print_(e)

        # Measure time difference after it finishes
        run_time = time.time() - start

        # Print the elapsed time nicely formatted, if verbose is enabled
        print_(
            f'Elapsed time: {time.strftime("%H:%M:%S", time.gmtime(run_time))}',
            verbose = True, 
            fname = caller.__name__
        )
        return f
    return wrapper


def download_file(url, msg=None, verbose=True, *args, **kwargs):
    """ Perform an HTTP GET request to fetch files from internet """ 

    if verbose:
        print_(f'Downloading file from {url}' if msg is None else msg, 
            *args, **kwargs)
    
    import requests

    # Strip the filename from the base url
    filename = url.split('/')[-1]

    # Perform an HTTP GET request
    req = requests.get(url)

    # Raise the HTTP Error if existent
    req.raise_for_status()

    # Download the file
    download = Path(filename).write_bytes(req.content)


def ring(soundfile=None):
    """ Play a sound from system. Useful to notify when a function finishes."""
    if not isinstance(soundfile, (str,PosixPath)):
        soundfile = "/usr/share/sounds/freedesktop/stereo/service-login.oga"

    os.system(f"paplay {soundfile} >/dev/null 2>&1")


def plot_checkout(fig, show, savefig, path="", block=True):
    """
    Final step in every plotting routine:
        - Check if the figure should be showed.
        - Check if the figure should be saved.
        - Return the figure, for further editing.
    """
    # Turn path into a Path object for flexibility
    path = Path(path)

    # Save the figure if required
    savefig = "" if savefig is None else savefig
    if savefig != "":
        # Check if savefig is a global path
        plt.savefig(f"{savefig}" if '/' in savefig else f"{path}/{savefig}") 

    # Show the figure if required
    if show:
        plt.show(block=block)

    return fig


def parse(s, delimiter="%", d=None):
    """
    Parse a string containing a given delimiter and return a dictionary
    containing the key:value pairs.
    """
    # Set the delimiter character
    delimiter = d if isinstance(d, (str,PosixPath)) else delimiter

    # Store all existing global and local variables
    g = globals()
    l = locals()

    string = s.replace(d, "{")
    string = s.replace("{_", "}_")

    # TO DO: This function is incomplete.
    return string


def set_hdr_to_iras16293B(
    hdr, 
    wcs="deg", 
    keep_wcs=False,
    spec_axis=False, 
    stokes_axis=False, 
    for_casa=False, 
    verbose=False, 
):
    """
    Adapt the header to match that of the ALMA observation of IRAS16293-2422B.
    Data from Maureira et al. (2020).
    """

    # Set the sky WCS to be in deg by default
    # and delete the extra WCSs
    if all([spec_axis, stokes_axis]):
        hdr["NAXIS"] = 4
    elif any([spec_axis, stokes_axis]):
        hdr["NAXIS"] = 3
    else:
        hdr["NAXIS"] = 3 if for_casa else 2

    keys = ["NAXIS", "CDELT", "CUNIT", "CRPIX", "CRVAL", "CTYPE"]

    WCS = {"deg": "A", "AU": "B", "pc": "C"}

    # TO DO: tell it to copy cdelt1A to cdelt1 only if more than one wcs exists.
    # Because, if no cdelt1A then it will set cdelt1 = None

    for n in [1, 2]:
        for k in keys[1:]:
            hdr[f"{k}{n}"] = hdr.get(f"{k}{n}{WCS[wcs]}", hdr.get(f"{k}{n}"))
            for a in WCS.values():
                hdr.remove(f"{k}{n}{a}", ignore_missing=True)

    for n in [3, 4]:
        for key in keys:
            hdr.remove(f"{key}{n}", ignore_missing=True)

    # Remove extra keywords PC3_* & PC*_3 added by CASA tasks and associated to a 3rd dim.
    if not any([spec_axis, stokes_axis, for_casa]):
        for k in ["PC1_3", "PC2_3", "PC3_3", "PC3_1", \
                    "PC3_2", "PC4_2", "PC4_3", "PC2_4", \
                    "PC3_4", "PC4_4", "PC4_1", "PC1_4"]:
            hdr.remove(k, True) 
            hdr.remove(k.replace('PC', 'PC0').replace('_', '_0'), True) 

    # Adjust the header to match obs. from IRAS16293-2422B
    if not keep_wcs:
        hdr["CUNIT1"] = "deg"
        hdr["CTYPE1"] = "RA---SIN"
        hdr["CRPIX1"] = 1 + hdr.get("NAXIS1") / 2
        hdr["CDELT1"] = hdr.get("CDELT1")
        hdr["CRVAL1"] = np.float64(248.0942916667)
        hdr["CUNIT2"] = "deg"
        hdr["CTYPE2"] = "DEC--SIN"
        hdr["CRPIX2"] = 1 + hdr.get("NAXIS2") / 2
        hdr["CDELT2"] = hdr.get("CDELT2")
        hdr["CRVAL2"] = np.float64(-24.47550000000)

    # Add spectral axis if required
    if spec_axis:
        # Convert the observing wavelength from the header into frequency
        wls = {
            "0.0013": {
                "freq": np.float64(230609583076.92307),
                "freq_res": np.float64(3.515082631882e10),
            },
            "0.003": {
                "freq": np.float64(99988140037.24495),
                "freq_res": np.float64(2.000144770049e09),
            },
            "0.007": {
                "freq": np.float64(42827493999.99999),
                "freq_res": np.float64(3.515082631882e10),
            },
            "0.0086": {
                "freq": np.float64(35e9),
                "freq_res": np.float64(1.0),
            },
            "0.0092": {
                "freq": np.float64(32.58614e9),
                "freq_res": np.float64(1.0),
            },
        }

        wl_from_hdr = str(hdr.get("HIERARCH WAVELENGTH1"))
        hdr["NAXIS3"] = 1
        hdr["CTYPE3"] = "FREQ"
        hdr["CRVAL3"] = wls[wl_from_hdr]["freq"]
        hdr["CRPIX3"] = np.float64(0.0)
        hdr["CDELT3"] = wls[wl_from_hdr]["freq_res"]
        hdr["CUNIT3"] = "Hz"
        hdr["RESTFRQ"] = hdr.get("CRVAL3")
        hdr["SPECSYS"] = "LSRK"

    # Add stokes axis if required
    if stokes_axis:
        hdr["NAXIS4"] = 1
        hdr["CTYPE4"] = "STOKES"
        hdr["CRVAL4"] = np.float32(1)
        hdr["CRPIX4"] = np.float32(0)
        hdr["CDELT4"] = np.float32(1)
        hdr["CUNIT4"] = ""

    # Add missing keywords (src: http://www.alma.inaf.it/images/ArchiveKeyworkds.pdf)
    hdr["BTYPE"] = "Intensity"
    hdr["BUNIT"] = "Jy/beam"
    hdr["BZERO"] = 0.0
    hdr["RADESYS"] = "ICRS"

    return hdr


def plot_opacity_file(
    filename='dust_mixture_001.dat', 
    col='all', 
    add_albedo=False, 
    return_data=False, 
    show=True, 
    savefig=None,
    verbose=True,
    ):
    """ Plot the dust opacity table generated by Polaris. 
    """
    
    from astropy.io import ascii

    # Handle path prefixes and special names if provided in the filename 
    filename = str(filename)
    if 'silicate' in filename:
        if '/' in filename:
            dirname = '/'.join(filename.split('/')[:-1])
            filename = f'{dirname}/dust_mixture_001_comp_003.dat'
        else:
            filename = 'dust_mixture_001_comp_003.dat' 
        
    elif 'graphite' in filename:
        if '/' in filename:
            dirname = '/'.join(filename.split('/')[:-1])
            filename1 = f'{dirname}/dust_mixture_001_comp_001.dat' 
            filename2 = f'{dirname}/dust_mixture_001_comp_002.dat' 
        else:
            filename1 = 'dust_mixture_001_comp_001.dat' 
            filename2 = 'dust_mixture_001_comp_002.dat' 
    
    # Read opacity table
    if 'graphite' in filename:
        # Use a graphite mixture with a 1/3-2/3 ratio for para. and perp.
        print_('Mixing graphite with 1/3 parallel 2/3 perpendicular', verbose)
        par = ascii.read(filename1, data_start=26)
        per = ascii.read(filename2, data_start=26)
        lam = par['col1']*u.m.to(u.micron)
        kappa = {
            'ext': (0.33*par['col16']+0.66*per['col16']) * (u.m**2/u.kg).to(u.cm**2/u.g),
            'abs': (0.33*par['col18']+0.66*per['col18']) * (u.m**2/u.kg).to(u.cm**2/u.g), 
            'sca': (0.33*par['col20']+0.66*per['col20']) * (u.m**2/u.kg).to(u.cm**2/u.g),
        }
    else:
        data = ascii.read(filename, data_start=26 if 'comp' in filename else 10)
        lam = data['col1']*u.m.to(u.micron)
        kappa = {
            'ext': data['col16'] * (u.m**2/u.kg).to(u.cm**2/u.g),
            'abs': data['col18'] * (u.m**2/u.kg).to(u.cm**2/u.g), 
            'sca': data['col20'] * (u.m**2/u.kg).to(u.cm**2/u.g),
        }

    # Select the columns from the file to plot
    if isinstance(col, str) and col != 'all':
        print_('Data is plotted and returned in SI units.', verbose)
        if col == 'ext':
            col = 'col16'
        elif col == 'abs':
            col = 'col18'
        elif col == 'sca':
            col = 'col20'
        opacity = data[col]
        one_curve = True
        data = (lam, opacity)

    elif isinstance(col, str) and col in ['abs', 'sca', 'ext']:
        opacity = kappa[col]
        one_curve = True
        data = (lam, opacity)

    elif col == 'all':
        opacity = ['abs', 'sca', 'ext']
        one_curve = False
        data = {'lam': lam, 'abs': kappa['abs'], 'sca': kappa['sca'], 'ext': kappa['ext']}

    elif isinstance(col, list):
        opacity = [k for k in col]
        one_curve = False
        data = {'lam': lam, 'kappa': opacity}

    # Initialize the figure
    fig, p = plt.subplots(ncols=1, nrows=2 if add_albedo else 1, sharex=True)

    if add_albedo:
        # Plot opacity
        p[0].loglog(lam, opacity, color='black')
        p[0].set_ylabel(r'$\kappa$ (cm$^2$ g$^{-1}$)')
        p[0].set_xlabel('Wavelength (microns)')
        p[0].set_xlim(lam.min(), lam.max())
        p[0].set_xticks([])

        # Plot albedo
        albedo = kappa['sca'] / (kappa['abs'] + kappa['sca'])
        p[1].loglog(lam, albedo, color='black')
        p[1].set_ylabel(r'Albedo $\omega$')
        p[1].set_xlabel('Wavelength (microns)')
        p[1].set_xlim(lam.min(), lam.max())
        plt.subplots_adjust(hspace=0)

    else:
        # Plot opacity
        if one_curve: 
            p.loglog(lam, opacity, color='black')
        else:
            color = {'abs': 'tab:red', 'sca': 'tab:green', 'ext': 'black'}
            [p.loglog(lam, kappa[k], label=k, color=color[k]) for k in opacity]
            p.legend(fontsize=18)
            p.set_ylim(1e-1, 1e4)

        p.set_xlabel('Wavelength (microns)')
        p.set_ylabel(r'$\kappa$ (cm$^2$ g$^{-1}$)')
        p.set_xlim(lam.min(), lam.max())
        plt.tight_layout()

    return data if return_data else plot_checkout(fig, show, savefig)


def plot_rosseland_opacity(filename='dust_mixture_001.dat', col='col16', domain='freq', show=True, savefig=None):
    """ Plot the frequency-averaged Rosseland mean opacity 
        for a given frequency-dependent opacity table.
        
        src: http://personal.psu.edu/rbc3/A534/lec6.pdf eq. 6.1.4
    """
    from astropy.io import ascii
    
    # Read opacity table
    kappa = ascii.read(filename, data_start=10)
    lam = kappa['col1']*u.m.to(u.cm)
    k_lam = kappa[col]*(u.m**2/u.kg).to(u.cm**2/u.g)

    # Clip the wavelength range below 1 or 10 micron, otherwise dB/dT diverges
    lam = lam[50:]
    k_lam = k_lam[50:]

    # Rescale dust opacities to gas opacities
    k_lam /= 100

    # Define constants & temperature range
    h = c.h.cgs.value
    c_ = c.c.cgs.value
    k_B = c.k_B.cgs.value
    temp = np.logspace(1, 5, k_lam.size)

    # Derivative of the Planck func. w/r to T for a range of temperatures
    k_ross = np.zeros(temp.shape)
    for i, T in enumerate(temp):
        if domain == 'lam':
            # As a function of wavelength
            dB_dT = ((2 * h**2 * c_**3) / (lam**6 * k_B * T**2)) * \
                    np.exp((h * c_) / (lam * k_B * T)) / \
                    (np.exp((h * c_) / (lam * k_B * T)) - 1)**2 

            # Compute the Rosseland mean opacity
            dlam = np.full(dB_dT.shape, fill_value=(lam[1] / lam[0]))
            k_ross[i] = np.sum(dB_dT) / np.sum((1/k_lam) * dB_dT)

        elif domain == 'freq':
            # As a function of frequency
            nu = c_ / lam

            dB_dT = ((2 * h**2 * nu**4) / (c_**2 * k_B * T**2)) * \
                    np.exp((h * nu) / (k_B * T)) / \
                    (np.exp((h * nu) / (k_B * T)) - 1)**2 

            # Compute the Rosseland mean opacity
            dnu = np.full(dB_dT.shape, fill_value=(nu[1] / nu[0]))
            k_ross[i] = np.sum(dB_dT * dnu) / np.sum((1/k_lam) * dB_dT * dnu)

    if show:
        # Initiliaze the figure
        fig, p = plt.subplots(nrows=2, ncols=1, figsize=(6, 7))

        # Plot the frequency-dependent opacity
        p[0].loglog(lam*u.cm.to(u.micron), k_lam, color='black')
        p[0].set_xlabel('Wavelength (microns)')
        p[0].set_ylabel(r'$\kappa_{\lambda}$ (cm$^2$ g$_{\rm gas}^{-1}$)')

        # Plot the Rosseland opacity
        p[1].loglog(temp, k_ross, color='black')
        p[1].set_xlabel('Temperature (K)')
        p[1].set_ylabel(r'$\bar{\kappa}$ (cm$^2$ g$_{\rm gas}^{-1}$)')
        p[1].set_xlim(temp.min(), temp.max())

        plt.tight_layout()

        return plot_checkout(fig, show, savefig)

    else:
        return temp, k_ross
    

@elapsed_time
def create_cube(
    filename="polaris_detector_nr0001.fits.gz",
    outfile="",
    specmode='cont', 
    wcs="deg",
    spec_axis=False,
    stokes_axis=False,
    add_scattered=False,
    for_casa=True,
    overwrite=False,
    verbose=False,
):
    """
    Retrieves data and header from filename and add necessary keywords to the cube.
    Then adjust the header to match that of IRAS16293B ALMA observations and write out.
    NOTE: Wildcards are allowed by the infile argument. Thanks to glob.
    """
    from glob import glob

    pwd = os.getcwd()

    # Set a global verbose if verbose is enabled
    if verbose:
        global VERBOSE
        VERBOSE = verbose

    # Read data
    if specmode in ['cont', 'mfs']:
        data, hdr = fits.getdata(filename, header=True)
        # Takes care if single channel is used but also extra axes are wanted
        if all([spec_axis, stokes_axis]):
            I = np.array([[data[0][0]]])
        elif any([spec_axis, stokes_axis]):
            I = np.array([data[0][0]])
        else:
            I = data[0][0]

    elif specmode in ['line', 'cube']:
        spec_axis = True
        # Let filename expand wildcards
        filename = glob(filename)
        hdr = fits.getheader(filename[0])
        map_shape = fits.getdata(filename[0])[0].shape
        I = np.zeros(shape=(len(filename), *map_shape))
        # If multiple channels, append each of 'em to a new cube
        for i, f in enumerate(filename):
            I[i] = fits.getdata(f)[0]


    # Edit the header to match the observation from IRAS16293B
    hdr = set_hdr_to_iras16293B(
        hdr, wcs=wcs, spec_axis=spec_axis, stokes_axis=stokes_axis, for_casa=for_casa
    )

    # Add emission by self-scattering if required
    if add_scattered:
        if "scattered emission" in hdr.get("ETYPE", ""):
            print_(
                "You are adding self-scattered flux to the self-scattered flux. Not gonna happen."
            )
            I_ss = np.zeros(I.shape)

        elif "thermal emission" in hdr.get("ETYPE", ""):
            print_("Adding self-scattered flux to the thermal flux.")
            if "dust_polarization" in pwd:
                pwd = pwd.replace("pa/", "")
                selfscat_file = pwd.replace("dust_polarization", "dust_scattering")

                # Change the polarized stokes I for the unpolarized stokes I
                I_th_unpol = pwd.replace("dust_polarization", "dust_emission")
                try:
                    i = fits.getdata(i_th_unpol + "/" + filename)[0][0]
                except OSError:
                    raise FileNotFoundError(
                        f'File with unpolarized thermal flux does not exist.\n\
											File: {i_th_unpol+"/"+filename}'
                    )

            else:
                selfscat_file = pwd.replace("dust_emission", "dust_scattering")

            try:
                I_ss = fits.getdata(selfscat_file + "/" + filename)[0][0]
            except OSError:
                raise FileNotFoundError(
                    f'File with self-scattered flux does not exist.\n\
										File: {selfscat_file+"/"+filename}'
                )

    else:
        I_ss = np.zeros(I.shape)

    # Add all sources of emission
    I = I + I_ss

    # Write data to fits file
    write_fits(outfile, I, hdr, overwrite, verbose)

    return data


def read_sph(snapshot="snap_541.dat", write_hdf5=False, remove_sink=True, cgs=True, verbose=False):
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
    
    outlier_index = 31330
    """
    import h5py

    # Read file in binary format
    with open(snapshot, "rb") as f:
        print_(f'Reading file: {snapshot}', verbose)
        names = f.readline()[1:].split()
        data = np.frombuffer(f.read()).reshape(-1, len(names))
        data = data.astype("f4")

    # Turn data into CGS 
    if cgs:
        data[:, 2:5] *= u.au.to(u.cm)
        data[:, 8] *= u.M_sun.to(u.g)

    # Remove the cell data of the outlier, likely associated to a sink particle
    if remove_sink:
        sink_id = 31330
        data[sink_id, 5] = 0
        data[sink_id, 6] = 0
        data[sink_id, 7] = 0
        data[sink_id, 8] = data[:,8].min()
        data[sink_id, 9] = data[:,9].min()
        data[sink_id, 10] = 3e-11 
        data[sink_id, 11] = 900

    if write_hdf5:
        # Follow the GADGET/GIZMO convention
        print_(f'Creating HDF5 file: {snapshot.replace("dat","h5")}', verbose)
        file = h5py.File(snapshot.replace('dat','h5'), 'w')
        h = file.create_group('Header')
        g = file.create_group('PartType0')
        g.create_dataset('ParticleIDs', data=data[:, 0], dtype='f')
        g.create_dataset('Time', data=data[:, 1], dtype='f')
        g.create_dataset('Coordinates', data=data[:, 2:5], dtype='f')
        g.create_dataset('Velocities', data=data[:, 5:8], dtype='f')
        g.create_dataset('Masses', data=data[:, 8], dtype='f')
        g.create_dataset('SmoothingLength', data=data[:, 9], dtype='f')
        g.create_dataset('Density', data=data[:, 10], dtype='f')
        g.create_dataset('Temperature', data=data[:, 11], dtype='f')
        g.create_dataset('InternalEnergy', data=data[:, 12], dtype='f')
        file.close()
        
    return data 


def read_zeusTW(frame):
    """
    Read in raw snapshots from ZeusTW and return a numpy array.
    This function is simply an adapted copy of Zeus2Polaris._read_data().
    """
    
    class Data:
        def read(self, filename, coord=False):
            """ Read the binary files from zeusTW.
                Reshape to 3D only if it is a physical quantity and not a coordinate.
            """
            # Load binary file
            with open(filename, "rb") as binfile:
                data = np.fromfile(file=binfile, dtype=np.double, count=-1)

            if coord:
                return data
            else:
                shape = (self.r.size, self.th.size, self.ph.size)
                return data.reshape(shape, order='F')

        def generate_coords(self, r, th, ph):
            self.r = self.read(r, coord=True)
            self.th = self.read(th, coord=True)
            self.ph = self.read(ph, coord=True)

        def trim_ghost_cells(self, field_type, ng=3):
            if field_type == 'coords':
                # Trim ghost zones for the coordinate fields
                self.r = self.r[ng:-ng]
                self.th = self.th[ng:-ng]
                self.ph = self.ph[ng:-ng]

            elif field_type == 'scalar':
                # Trim ghost cells for scalar fields
                self.rho = self.rho[ng:-ng, ng:-ng, ng:-ng]

            elif field_type == 'vector':
                # Trim ghost cells for vector fields
                self.Vr = 0.5 * (self.Vr[ng:-ng, ng:-ng, ng:-ng] + self.Vr[ng+1:-ng+1, ng:-ng, ng:-ng])
                self.Vth = 0.5 * (self.Vth[ng:-ng, ng:-ng, ng:-ng] + self.Vth[ng:-ng, ng+1:-ng+1, ng:-ng])
                self.Vph = 0.5 *  (self.Vph[ng:-ng, ng:-ng, ng:-ng] + self.Vph[ng:-ng, ng:-ng, ng+1:-ng+1])

                self.Br = 0.5 * (self.Br[ng:-ng, ng:-ng, ng:-ng] + self.Br[ng+1:-ng+1, ng:-ng, ng:-ng])
                self.Bth = 0.5 * (self.Bth[ng:-ng, ng:-ng, ng:-ng] + self.Bth[ng:-ng, ng+1:-ng+1, ng:-ng])
                self.Bph = 0.5 * (self.Bph[ng:-ng, ng:-ng, ng:-ng] + self.Bph[ng:-ng, ng:-ng, ng+1:-ng+1])
        
        def generate_temperature(self):
            """ Formula taken from Appendix A Zhao et al. (2018). """
            rho_cr = 1e-13
            csound = 1.88e-4
            mu = 2.36
            T0 = csound**2 * mu * c.m_p.cgs.value / c.k_B.cgs.value
            T1 = T0 + 1.5 * self.rho/rho_cr
            T2 = np.where(self.rho >= 10*rho_cr, (T0+15) * (self.rho/rho_cr/10)**0.6, T1)
            T3 = np.where(self.rho >= 100*rho_cr, 10**0.6 * (T0+15) * (self.rho/rho_cr/100)**0.44, T2)
            self.temp = T3
    
        def LH_to_Gaussian(self):
            self.Br *= np.sqrt(4 * np.pi)
            self.Bth *= np.sqrt(4 * np.pi)
            self.Bph *= np.sqrt(4 * np.pi)
        
        def generate_cartesian(self):
            """ Convert spherical coordinates and vector components from spherical to cartesian. """
            # Create a coordinate grid.
            r, th, ph = np.meshgrid(self.r, self.th, self.ph, indexing='ij')

            # Convert coordinates to cartesian
            self.x = r * np.cos(ph) * np.sin(th)
            self.y = r * np.sin(ph) * np.sin(th)
            self.z = r * np.cos(th)

            # Transform vector components to cartesian
            self.Vx = self.Vr * np.sin(th) * np.cos(ph) + self.Vth * np.cos(th) * np.cos(ph) - self.Vph * np.sin(ph)
            self.Vy = self.Vr * np.sin(th) * np.sin(ph) + self.Vth * np.cos(th) * np.sin(ph) + self.Vph * np.cos(ph)
            self.Vz = self.Vr * np.cos(th) - self.Vth * np.sin(th)

            self.Bx = self.Br * np.sin(th) * np.cos(ph) + self.Bth * np.cos(th) * np.cos(ph) - self.Bph * np.sin(ph)
            self.By = self.Br * np.sin(th) * np.sin(ph) + self.Bth * np.cos(th) * np.sin(ph) + self.Bph * np.cos(ph)
            self.Bz = self.Br * np.cos(th) - self.Bth * np.sin(th)

    # Generate a Data instance
    data = Data()

    # Read coordinates: x?a are cell edges and x?b are cell centers
    data.generate_coords(r="z_x1ap", th="z_x2ap", ph="z_x3ap")

    # Read Data
    frame = str(frame).zfill(5)
    data.rho = data.read(f"o_d__{frame}")
    data.Br = data.read(f"o_b1_{frame}")
    data.Bth = data.read(f"o_b2_{frame}")
    data.Bph = data.read(f"o_b3_{frame}")
    data.Vr = data.read(f"o_v1_{frame}")
    data.Vth = data.read(f"o_v2_{frame}")
    data.Vph = data.read(f"o_v3_{frame}")

    # Trim ghost zones for the coordinate fields
    data.trim_ghost_cells(field_type='coords')

    # Trim ghost cells for scalar fields
    data.trim_ghost_cells(field_type='scalar')

    # Trim ghost cells for vector fields
    data.trim_ghost_cells(field_type='vector')

    # Convert from Lorent-Heaviside to Gaussian system
    data.LH_to_Gaussian()

    # Generate the temperature field using a barotropic Equation of State
    data.generate_temperature()

    # Generate cartesian coordinates
    data.generate_cartesian()

    return data


def radmc3d_data(file_='image.out', npix=300, sizeau=50, distance=3.086e18 * u.m.to(u.cm)):
    """
    Function to read image files resulting from an RT with RADMC3D.
    """

    img = ascii.read(file_, data_start=5, guess=False)["1"]

    # Make a squared map
    img = img.reshape(npix, npix)

    # Rescale to Jy/sr
    img = img * (u.erg * u.s ** -1 * u.cm ** -2 * u.Hz ** -1 * u.sr ** -1).to(
        u.Jy * u.sr ** -1
    )

    # Obtain the pixel size
    pixsize = (sizeau / npix) * u.au.to(u.cm)

    # Convert sr into pixels (Jy/sr --> Jy/pixel)
    img = img * ((pixsize) ** 2 / (distance) ** 2)

    return img


def convert_opacity_file(
    infile='dust_mixture_001.dat', 
    outfile='dustkappa_polaris.inp', 
    verbose=True,
    show=True, 
):
    """
    Convert dust opacity files from POLARIS to RADMC3D format.
    """

    # Read polaris file with dust info
    print_(f'Reading in polaris dust file: {infile} in SI units', verbose)
    d = ascii.read(infile, data_start=10)
    
    # Store the wavelenght, absorption and scattering opacities and assymetry g
    lam = d['col1'] * u.m.to(u.micron)
    kabs = d['col18'] * (u.m**2/u.kg).to(u.cm**2/u.g)
    ksca = d['col20'] * (u.m**2/u.kg).to(u.cm**2/u.g)
    g_HG = d['col9']

    print_(f'Writing out radmc3d opacity file: {outfile} in CGS', verbose)
    with open(outfile, 'w+') as f:
        f.write('3\n')
        f.write(f'{len(lam)}\n')
        for i,l in enumerate(lam):
            f.write(f'{l:.6e}\t{kabs[i]:.6e}\t{ksca[i]:.6e}\t{g_HG[i]:.6e}\n')
    
    if show:
        print_(f'Plotting opacities ...', verbose)
        plt.loglog(lam, kabs, '--', c='black', label=r'$\kappa_{\rm abs}$')
        plt.loglog(lam, ksca, ':', c='black', label=r'$\kappa_{\rm sca}$')
        plt.loglog(lam, kabs+ksca, '-', c='black', label=r'$\kappa_{\rm ext}$')
        plt.legend()
        plt.xlabel('Wavelength (microns)')
        plt.ylabel(r'Dust opacity $\kappa$ (cm$^2$ g$^{-1}$)')
        plt.xlim(1e-1, 1e5)
        plt.ylim(1e-2, 1e4)
        plt.tight_layout()
    

def radmc3d_casafits(fitsfile='radmc3d_I.fits', radmcimage='image.out',
        stokes='I'):
    """ Read in an image.out file created by RADMC3D and generate a
        FITS file with a CASA-compatible header, ready for a 
        synthetic observation.
    """
    from radmc3dPy import image

    im = image.readImage(radmcimage)
    
    im.writeFits(
        fitsfile,
        dpc=141, 
        coord='16h32m22.63s -24d28m31.8s', 
        casa=False,
        stokes=stokes,
    )


def fill_gap(
    filename,
    outfile=None,
    x1=123,
    x2=168,
    y1=123,
    y2=167,
    threshold=1e-5,
    fill_value=None, 
    show=True, 
    savefile=True,
    return_data=False,
):
    """
    Fill the central gap from polaris images with the peak flux
    """

    # Read data
    d, hdr = fits.getdata(filename, header=True)
    d = d.squeeze()
    
    # Copy the original array
    filled = d
    fill_value = d.max() if fill_value is None else fill_value
    gap = np.where(d[x1:x2, y1:y2] > threshold, fill_value, d[x1:x2, y1:y2])
    filled[x1:x2, y1:y2] = gap

    # Show the filled image
    if show:
        plt.imshow(filled, cmap="magma")
        plt.colorbar()
        plt.show()

    # Append suffix to output filename if not provided
    if outfile is None:
        outfile = filename.split(".fits")[0] + "_filled.fits"

    # Save the output file
    if savefile:
        fits.writeto(outfile, data=filled, header=hdr, overwrite=True)

    if return_data:
        return filled


def circular_mask(shape, c, r, angle_range=(0, 360), ring=False):
    """
    Return a boolean mask for a circular sector. The start/stop angles in
    `angle_range` should be given in clockwise order.
    """

    x, y = np.ogrid[: shape[0], : shape[1]]
    cx, cy = c
    a_i, a_f = np.deg2rad(angle_range)

    # Ensure stop angle > start angle
    if a_f < a_i:
        a_f += 2 * np.pi

    # Convert cartesian --> polar coordinates
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    theta = np.arctan2(x - cx, y - cy) - a_i

    # Wrap angles between 0 and 2*pi
    theta %= 2 * np.pi

    # Circular mask
    circular_mask = r2 <= r * r

    # Angular mask
    angular_mask = theta <= (a_f - a_i)

    # Subtract an inner circle to get a ring
    ring_mask = r2 < 0.9 * r * r
    ring_mask = ring_mask ^ circular_mask

    return (ring_mask * angular_mask) if ring else (circular_mask * angular_mask)


@elapsed_time
def radial_profile(
    fitsfile,
    average='vertical', 
    func=np.nanmean,
    step=1,
    bin_factor=1, 
    nthreads=1,
    return_radii=False,
    figsize=(8, 4.5), 
    show=False,
    savefig=None,
	verbose=True,
    *args,
    **kwargs,
):
    """
    Computes the radial profile (average by default) of a 2D array
    by averaging the values within consecutive concentric circumferences
    from the border to the center.
    """
    from astropy.nddata import block_reduce
    from matplotlib.colors import LogNorm
    from concurrent.futures import ThreadPoolExecutor

    def annular_average(image):
        def parallel_masking(i, r):
            """Function created to parallelize the for loop
            by means of a parallell map(func, args).
            """
            # Copy data to avoid propagating NaNs in the original array
            d_copy = np.copy(image)
            mask = circular_mask(d_copy.shape, center, r, ring=True)
            d_copy[~mask] = np.NaN
            averages[i] = func(d_copy)
            del d_copy

        # Get the center of the array
        map_radius_x = int(image[0].size / 2)
        map_radius_y = int(image[1].size / 2)
        center = (map_radius_x, map_radius_y)

        # Masks are created from the border to the center because otherwise
        # all masks other than r=0 would already be NaN.
        radii = np.arange(0, map_radius_x, step)[::-1]
        averages = np.zeros(radii.shape)

        # Parallel iteration over radii
        with ThreadPoolExecutor(max_workers = nthreads) as pool:
            pool.map(parallel_masking, range(radii.size), radii)

        # Reverse the averages to be given from the center to the border
        averages = averages[::-1]
        radii = radii[::-1] * dr
    
        return radii, averages

    print_(f'The anular averages are paralelized using {nthreads} threads.', verbose)

    # Read data from fits file if filename is provided
    if isinstance(fitsfile, (str,PosixPath)):
        if average == 'midplane':
    
            # Read in fits file
            print_('Read in data files ...', True)
            data, hdr = fits.getdata(fitsfile, header=True)
            data = data.squeeze()[0, 0]
    
            # Read the delta length from header. Axis B is in AU.
            dr = hdr.get('CDELT1B')

            print_(f'Averaging image at z = 0 AU', verbose)
            # Calculate the average temp in annuli at different radii at z=0
            radii, profiles = annular_average(data)

        elif average == 'vertical':

            # Read in fits file
            print_('Read in data files ...', True)
            data, hdr = fits.getdata(fitsfile, header=True)
            data = data.squeeze()
            
            # Read the density file to add contours
            dens = fits.getdata('dust_density_3d.fits.gz')
            dens = dens.squeeze() * (u.kg/u.m**3).to(u.g/u.cm**3)

            # Read the delta length from header. Axis B is in AU.
            dr = hdr.get('CDELT1B')

            # Bin down the array if requested
            if bin_factor not in [1, [1,1,1]]:
                data = block_reduce(data, bin_factor, func=np.mean)
                dens = block_reduce(dens, bin_factor, func=np.mean)
                dr = dr * bin_factor

            # Calculate the average temp in annuli at different radii and z
            ar_size = data.shape[0]
            z_axis = np.linspace(-ar_size/2, ar_size/2, ar_size) * dr
            
            # Create a grid radii vs z-axis
            temp_z = np.zeros((int(ar_size), int(ar_size/2)))
            dens_z = np.zeros((int(ar_size), int(ar_size/2)))

            for i, z in enumerate(data):
                dz = np.round(z_axis[i], 2)
                print_(f'Averaging slice at z = {dz} AU', verbose)
                radii, temp_avg = annular_average(z)
                radii, dens_avg = annular_average(dens[i])

                # Populate the 2D temperature array
                for j, r in enumerate(radii):
                    temp_z[i][j] = temp_avg[j]
                    dens_z[i][j] = dens_avg[j]

            profiles = (radii, z_axis, temp_z)

    else:
        # If the input is not a FITS file, read the delta length from stdin
        if show:
            dr = float(input('[radial_profile] Enter dr [AU]: '))

    # Plot the radial profile if required
    if show or savefig is not None:
        fig = plt.figure(figsize=figsize)

        if average == 'midplane':
            plt.semilogx(radii, profiles, *args, **kwargs)
            plt.ylabel(r'$T_{\rm dust}$ (K)')
            plt.xlabel(f"Radius (AU)")

        elif average == 'vertical':
            plt.contourf(radii, z_axis, temp_z.clip(min=0, max=400), 
                levels=90, *args, **kwargs)
            plt.colorbar().set_label(r'$T_{\rm dust}$ (K)')
            plt.contour(radii, z_axis, np.log10(dens_z), linewidths=0.6, 
                levels=[-13.5], colors='white',)
            plt.axhline(y=0, c='black', ls='-', lw=0.5)
            plt.ylabel(r'$z$-axis (AU)')
            plt.xlabel(f"Radius (AU)")
            plt.xlim(0, 30)
            plt.ylim(-10, 10)
            plt.tight_layout()

        if isinstance(savefig, (str,PosixPath)) and len(savefig) > 0:
            plt.save(savefig)
        if show:
            plt.show()

    return (radii, profiles) if return_radii else profiles


def stats(data, verbose=False, slice=None):
    """
    Compute basic statistics of a array or a fits file.
    The functions used here ignore NaN values in the data.
    """

    # Read data
    if isinstance(data, (str,PosixPath)):
        data, hdr = fits.getdata(data, header=True)

        if isinstance(slice, int):
            data = data[slice]
        elif isinstance(slice, list) and len(slice) == 2:
            data = data[slice[0], slice[1]]
    else:
        data = np.array(data)

    # Set the relevant quantities
    stat = {
        "max": np.nanmax(data),
        "mean": np.nanmean(data),
        "min": np.nanmin(data),
        "maxpos": maxpos(data),
        "minpos": minpos(data),
        "std": np.nanstd(data),
        "S/N": np.nanmax(data) / np.nanstd(data),
    }

    # Print statistics if verbose enabled
    for label, value in stat.items():
        print_(f"{label}: {value}", verbose=verbose)

    return stat


def get_beam(filename, verbose=True):
    """ Print or return the info from the header associated to the beam """
    
    data, hdr = fits.getdata(filename, header=True)

    beam = {
        'bmaj': hdr.get('BMAJ', default=0) * u.deg.to(u.arcsec), 
        'bmin': hdr.get('BMIN', default=0) * u.deg.to(u.arcsec), 
        'bpa': hdr.get('BPA', default=0), 
    }
    
    print_(f"Bmaj: {beam['bmaj']:.2f} arcsec", verbose=verbose)
    print_(f"Bmin: {beam['bmin']:.2f} arcsec", verbose=verbose)
    print_(f"Bpa: {beam['bpa']:.2f} deg", verbose=verbose)

    return beam
        

def maxpos(data, axis=None):
    """
    Return a tuple with the coordinates of a N-dimensional array.
    """
    # Read data from fits file if data is string
    if isinstance(data, (str,PosixPath)):
        data = fits.getdata(data)

    # Remove empty axes
    data = np.squeeze(data)

    return np.unravel_index(np.nanargmax(data, axis=axis), data.shape)


def minpos(data, axis=None):
    """
    Return a tuple with the coordinates of a N-dimensional array.
    """
    # Read data from fits file if data is string
    if isinstance(data, (str,PosixPath)):
        data = fits.getdata(data)

    # Remove empty axes
    data = np.squeeze(data)

    return np.unravel_index(np.nanargmin(data, axis=None), data.shape)


def add_comment(filename, comment):
    """
    Read in a fits file and add a new keyword to the header.
    """
    data, header = fits.getdata(filename, header=True)

    header["NOTE"] = comment

    write_fits(filename, data=data, header=header, overwrite=True)


def edit_header(filename, key, value, verbose=True):
    """
    Read in a fits file and change the value of a given keyword.
    """
    data, header = fits.getdata(filename, header=True)

    # Check if the key is already in the header
    value_ = header.get(key, default=None)

    if value_ is None:
        print_(f"Adding keyword {key} = {value}", verbose=verbose)
        header[key] = value

    elif value == 'del' and value_ is not None:
        print_(f'Deleting from header: {key} = {value_}', verbose=verbose)
        del header[key]

    else:
        print_(f"Keyword {key} already exists.", verbose=verbose)
        print_(f"Updating from {value_} to {value}.", verbose=verbose)
        header[key] = value

    write_fits(filename, data=data, header=header, overwrite=True, verbose=verbose)


@elapsed_time
def plot_map(
    filename,
    header=None,
    rescale=1,
    transpose=False,
    rot90=False, 
    fliplr=False, 
    flipud=False, 
    cblabel=None,
    scalebar=20 * u.au,
    cmap="magma",
    stretch='linear', 
    verbose=True,
    bright_temp=True,
    figsize=None,
    vmin=None, 
    vmax=None, 
    contours=False, 
    show=True, 
    savefig=None, 
    checkout=True, 
    block=True, 
    *args,
    **kwargs,
):
    """
    Plot a fits file using the APLPy library.
    """
    from aplpy import FITSFigure

    # Read from FITS file if input is a filename
    if isinstance(filename, (str,PosixPath)):
        data, hdr = fits.getdata(filename, header=True)

    elif isinstance(filename, (np.ndarray, list)) and header is not None:
        data = filename
        hdr = header

    # Flip the 2D array if required
    if rot90:
        print_('Rotating the image by 90 degrees', verbose)
        data = np.rot90(data)
    if transpose:
        print_('Transposing the image', verbose)
        data = np.transpose(data)
    if flipud:
        print_('Flipping up to down', verbose)
        data = np.flipud(data)
    if fliplr:
        print_('Flipping left to right', verbose)
        data = np.fliplr(data)

    # Remove non-celestial WCS
    filename_ = filename
    if hdr.get("NAXIS") > 2:
        tempfile = '.temp_file.fits'
        # <patch>: Use the header from the observation of IRAS16293B
        # but keep the phasecenter of the original input file
        print_('Setting the header to that from IRAS16293B', verbose, bold=True)
        hdr_ = set_hdr_to_iras16293B(hdr)
        write_fits(
            tempfile, 
            data=data.squeeze(), 
            header=hdr_, 
            overwrite=True
        )
        filename = tempfile
    
    # Convert Jy/beam into Kelvin if required
    if bright_temp:
        try:
            rescale = Tb(
                data=rescale,
                freq=hdr.get("RESTFRQ") * u.Hz.to(u.GHz),
                bmin=hdr.get("bmin") * u.deg.to(u.arcsec),
                bmaj=hdr.get("bmaj") * u.deg.to(u.arcsec),
            )
        except Exception as e:
            print_('Beam or frequency keywords not available. ' +\
            'Impossible to convert into T_b.', verbose, bold=True)

    # Initialize the figure
    fig = FITSFigure(
        str(filename), rescale=rescale, figsize=figsize, *args, **kwargs)
    fig.set_auto_refresh(True)
    fig.show_colorscale(cmap=cmap, vmax=vmax, vmin=vmin, stretch=stretch)
    
    # Add contours if requested
    if contours == True:
        conts = fig.show_contour(
            colors='white', 
            levels=8, 
            returnlevels=True, 
            alpha=0.5
        )
        print_(f'contours: {conts}', verbose)

    elif isinstance(contours, (str, PosixPath)):
        conts = fig.show_contour(
            str(contours), 
            colors='white', 
            levels=8, 
            returnlevels=True, 
            alpha=0.5
        )
        print_(f'Setting contours from external file: {conts}', verbose)

    # Auto set the colorbar label if not provided
    if cblabel is None and bright_temp:
        cblabel = r"$T_{\rm b}$ (K)"

    elif cblabel is None and rescale == 1e3:
        cblabel = "mJy/beam"

    elif cblabel is None and rescale == 1e6:
        cblabel = r"$\mu$Jy/beam"

    elif cblabel is None:
        try:
            hdr = fits.getheader(filename)
            cblabel = hdr.get("BUNIT")
        except Exception as e:
            print_(e, verbose)
            cblabel = ""

    # Colorbar
    fig.add_colorbar()
    fig.colorbar.set_location("top")
    fig.colorbar.set_axis_label_text(cblabel)
    fig.colorbar.set_axis_label_font(size=20, weight=15)
    fig.colorbar.set_font(size=20, weight=15)

    # Frame and ticks
    fig.frame.set_color("black")
    fig.frame.set_linewidth(1.2)
    fig.ticks.set_color('black' if cmap=='magma' else 'black')
    fig.ticks.set_linewidth(1.2)
    fig.ticks.set_length(6)
    fig.ticks.set_minor_frequency(5)

    # Hide ticks and labels if FITS file is not a real obs.
    if "alma" not in str(filename_) or "vla" not in str(filename_):
        print_(f'File: {filename_}. Hiding axis ticks and labels', False)
        fig.axis_labels.hide()

    # Beam
    if 'BMAJ' in hdr and 'BMIN' in hdr and 'BPA' in hdr:
        fig.add_beam(facecolor='none', edgecolor='white', linewidth=1)
        bmaj = hdr.get('BMAJ') * u.deg.to(u.arcsec)
        bmin = hdr.get('BMIN') * u.deg.to(u.arcsec)
        #fig.add_label(0.28, 0.07, f'{bmaj:.1f}"x {bmin:.1f}"', 
        #    relative=True, color='white', size=13)

    # Scalebar
    if scalebar is not None:
        if scalebar.unit in ['au', 'pc']:
            try:
                D = 141 * u.pc
                print_(f'Physical scalebar created for a distance of: {D}', verbose=False)
                scalebar_ = (scalebar.to(u.cm) / D.to(u.cm)) * u.rad.to(u.arcsec)
                fig.add_scalebar(scalebar_ * u.arcsec)
                unit = f' {scalebar.unit}'
            except Exception as e:
                print_(f'Not able to add scale bar. Error: {e}', verbose=True, red=True)

        elif scalebar.unit in ['arcsec', 'deg']:
            fig.add_scalebar(scalebar)
            unit = f'"' if scalebar.unit == 'arcsec' else "'"

        fig.scalebar.set_color("white")
        fig.scalebar.set_corner("bottom right")
        fig.scalebar.set_font(size=23)
        fig.scalebar.set_linewidth(3)
        fig.scalebar.set_label(f"{int(scalebar.value)}{unit}")

    # Delete the temporary file created to get rid of extra dimensions
    if hdr.get("NAXIS") > 2 and os.path.isfile(tempfile): os.remove(tempfile)

    return plot_checkout(fig, show, savefig, block=block) if checkout else fig


def pol_angle(stokes_q, stokes_u):
    """Calculates the polarization angle from Q and U Stokes component.

    Args:
        stokes_q (float): Q-Stokes component [Jy].
        stokes_u (float): U-Stokes component [Jy].

    Returns:
        float: Polarization angle.
    
    Disclaimer: This function is directly copied from polaris-tools.
    """
    # Polarization angle from Stokes Q component
    q_angle = 0.
    if stokes_q >= 0:
        q_angle = np.pi / 2.
    # Polarization angle from Stokes U component
    u_angle = 0.
    if stokes_u >= 0:
        u_angle = np.pi / 4.
    elif stokes_u < 0:
        if stokes_q >= 0:
            u_angle = np.pi * 3. / 4.
        elif stokes_q < 0:
            u_angle = -np.pi / 4.
    # x vector components from both angles
    x = abs(stokes_q) * np.sin(q_angle)
    x += abs(stokes_u) * np.sin(u_angle)
    # y vector components from both angles
    y = abs(stokes_q) * np.cos(q_angle)
    y += abs(stokes_u) * np.cos(u_angle)
    # Define a global direction of the polarization vector 
    # since polarization vectors are ambiguous in both directions.
    if x < 0:
        x *= -1.0
        y *= -1.0
    # Polarization angle calculated from Q and U components
    pol_angle = np.arctan2(y, x)

    return pol_angle, x, y


@elapsed_time
def polarization_map(
    source = 'radmc3d', 
    render="intensity",
    polarization="linear", 
    stokes_I = None, 
    stokes_Q = None, 
    stokes_U = None, 
    wcs="deg",
    rotate=0,
    step=20,
    scale=50,
    scalebar=None, 
    mapsize=None, 
    vector_color="white",
    vector_width=1, 
    add_thermal=False,
    add_scattered=False,
    add_bfield=False,
    const_bfield=False,
    const_pfrac=False,
    min_pfrac=0, 
    rms_I=None, 
    rms_Q=None, 
    bright_temp=False, 
    rescale=1, 
    savefig=None,
    show=True,
    verbose=True,
    *args,
    **kwargs,
):
    """
    Extract I, Q, U and V from the polaris output
    and create maps of Pfrac and Pangle using APLpy.
    """
    from aplpy import FITSFigure

    # Enable verbosity
    global VERBOSE
    VERBOSE = verbose

    # Store the current path
    pwd = os.getcwd()

    # Read the Stokes components from a data cube
    if source in ['alma', 'vla', 'radmc3d']:
        hdr = fits.getheader(f'{source}_I.fits')

        I = fits.getdata(f'{source}_I.fits').squeeze()
        Q = fits.getdata(f'{source}_Q.fits').squeeze()
        U = fits.getdata(f'{source}_U.fits').squeeze()
        V = np.zeros(U.shape)
        tau = np.zeros(I.shape)
        
    elif 'polaris' in source:
        filename = 'polaris_detector_nr0001.fits.gz'
        data = fits.getdata(filename).squeeze()
        hdr = fits.getheader(filename)

        I = data[0]
        Q = data[1]
        U = data[2]
        V = data[3]

        try:
            tau = data[4]
        except:
            tau = np.zeros(I.shape)

        # Add thermal flux to the scattered flux.
        # If add_thermal is a path, then read the flux from file
        if isinstance(add_thermal, (str,PosixPath)):
            try:
                I_th = fits.getdata(add_thermal)
                if 'polaris_detector' in add_thermal:
                    I_th = I_th[0][0]
            except OSError:
                raise FileNotFoundError(
                    f"File with thermal flux does not exist.\n" + f"File: {add_thermal}"
                )

        # If add_thermal is True, assume the path to scattered flux file is similarly structured 
        elif add_thermal:
            if "thermal emission" in hdr.get("ETYPE", ""):
                print_("You are adding thermal flux to the thermal flux. Not gonna happen.", bold=True)
                I_th = np.zeros(I.shape)

            elif "scattered emission" in hdr.get("ETYPE", ""):
                print_("Adding thermal flux to the self-scattered flux.", bold=True)

                if 'dust_scattering' in pwd:
                    thermal_file = pwd.replace("dust_scattering", "dust_emission")
                    thermal_file = thermal_file.replace("data", "dust_scat/data")
                    print_(f'pwd: {pwd}', verbose=True, bold=True)
                elif 'dust_mc' in pwd:
                    thermal_file = pwd.replace("dust_mc", "dust_th")

                print_(f"File: {thermal_file + '/' + filename}", bold=True)
                try:
                    I_th = fits.getdata(thermal_file + "/" + filename)[0][0]
                except OSError:
                    raise FileNotFoundError(
                        f'File with thermal flux does not exist.\n\
                                            File: {thermal_file+"/"+filename}'
                    )
        # If not provided, set to zero
        else:
            I_th = np.zeros(I.shape)

        # Add self-scattered emission to the thermal emission if required
        if isinstance(add_scattered, (str,PosixPath)):
            try:
                I_ss = fits.getdata(add_scattered)
            except OSError:
                raise FileNotFoundError(
                    f"File with thermal flux does not exist.\n" + \
                    "File: {add_thermal}"
                )

        # If add_thermal is True, assume the path to scattered flux file is 
        # similarly structured 
        elif add_scattered:
            if "scattered emission" in hdr.get("ETYPE", ""):
                print_("You are adding scattered flux to the scattered flux. "+\
                    "Not gonna happen.", bold=True)
                I_ss = np.zeros(I.shape)

            elif "thermal emission" in hdr.get("ETYPE", ""):
                print_("Adding scattered flux to the thermal flux.", bold=True)

                try:
                    if os.path.isfile(f'{pwd}/../dust_scat/data/' + 
                        'polaris_detector_nr0001.fits.gz'):
                        # Get the full flux from results of CMD_DUST_EMISSION incl. scat.
                            I_ss = fits.getdata(f'{pwd}/../dust_scat/data/' +
                             'polaris_detector_nr0001.fits.gz')[0][0]
                            # Remove thermal flux, since this file has both.
                            I_ss = I_ss - I
                    else:
                        # Look for it in results from CMD_DUST_SCATTERING mode 
                        if 'dust_alignment' in pwd:
                            scattered_file = pwd.replace("dust_alignment", "dust_scattering")
                            scattered_file = scattered_file.replace("pa/", "")
                        elif 'dust_emission' in pwd:
                            scattered_file = pwd.replace("dust_emission", "dust_scattering")
                        elif 'dust_mc' in pwd:
                            scattered_file = pwd.replace("dust_mc", "dust_th")

                        print_(f"File: {scattered_file + '/' + filename}", bold=True)
                        I_ss = fits.getdata(scattered_file + "/" + filename)[0][0]
                except OSError:
                    raise FileNotFoundError(
                        f'File with scattered flux does not exist.\n\
                         File: {scattered_file+"/"+filename}')

        # If not provided, set scattered flux to zero
        else:
            I_ss = np.zeros(I.shape)

        # Add all sources of flux: thermal and scattered emission
        I = I + I_th + I_ss

    else:
        # Assume the name of the source files for I, Q and U is given manually
        source = 'obs'

        print_('Reading files from an external source ...', True)
        hdr = fits.getheader('obs_I.fits' if stokes_I is None else stokes_I)
        I = fits.getdata('obs_I.fits' if stokes_I is None else stokes_I).squeeze()
        Q = fits.getdata('obs_Q.fits' if stokes_Q is None else stokes_Q).squeeze()
        U = fits.getdata('obs_U.fits' if stokes_U is None else stokes_U).squeeze()
        V = np.zeros(U.shape)
        tau = np.zeros(I.shape)

    # Compute the polarizatoin angle
    pangle = 0.5 * np.arctan2(U, Q)
    pangle = pangle * u.rad.to(u.deg)
	
    # Set rms of stokes I  
    if rms_I is None and 'RMS_I' in hdr:
        rms_I = hdr.get('RMS_I')
        min_I = 5 * rms_I 
    elif rms_I is not None and rms_I > 0:
        min_I = 5 * rms_I
    else:
        rms_I = 0
        min_I = np.nanmin(I)

    # Set rms of stokes Q  
    if rms_Q is None and 'RMS_Q' in hdr:
        rms_Q = hdr.get('RMS_Q')
        min_Q = 3 * rms_Q 
    elif rms_Q is not None and rms_Q > 0:
        min_Q = 2 * rms_Q
    else:
        rms_Q = 0
        min_Q = np.nanmin(Q)
        
    if source in ['alma', 'vla']:
        print_(f'rms_I: {rms_I}', verbose, bold=True)
        print_(f'rms_Q: {rms_Q}', verbose, bold=True)

    # Compute the polarized intensity
    # Apply debias correction (Viallancourt et al. 2006)
    pi = np.sqrt(U**2 + Q**2 - rms_Q**2)

    # Compute the polarization fraction 
    if polarization in ['linear', 'l']:
        pfrac = np.divide(pi, I, where=I != 0)
    elif polarization in ['circular', 'c']:
        pfrac = V / I

    # Mask the polarization vectors for a given threshold in pol. fraction
    if min_pfrac > 0:
        pangle[pfrac < min_pfrac] = np.NaN

    # Mask the polarization vectors emission with stokes I under a given SNR
    pangle[I < min_I] = np.NaN
    pfrac[I < min_I] = np.NaN

    # Mask the polarization vectors emission with pol. intensity under a given SNR
    pangle[pi < min_Q] = np.NaN
    pfrac[pi < min_Q] = np.NaN
    pi[pi < min_Q] = np.NaN
    
    # Set the polarization fraction to 100% to plot vectors of constant length
    if const_pfrac: 
        if render.lower() in ["pf", "pfrac"]:
            print_(f'Ignoring const_pfrac = {const_pfrac}', verbose)
        else:
            pfrac = np.ones(pfrac.shape) 

    # Edit the header of the models to set the phasecenter from IRAS16293B
    if source == 'polaris':
        hdr = set_hdr_to_iras16293B(hdr, keep_wcs=True, verbose=True)
        hdr['BUNIT'] = 'Jy/pixel'

    # Write quantities into fits files
    quantities = {
        "I": I, 
        "Q": Q, 
        "U": U, 
        #"V": V, 
        #"tau": tau, 
        "pi": pi, 
        "pf": pfrac, 
        "pa": pangle
    }
    for q, d in quantities.items():
        write_fits(f'{source}_{q}.fits', data=d, header=hdr, overwrite=True)

    # Define the unit for the plot
    if bright_temp:
        unit = '(K)'
    elif 'BUNIT' in hdr:
        unit = '(' + hdr.get('BUNIT') + ')'
    else:
        unit = r'($\mu$Jy/pixel)' if rescale is None else '(Jy/pixel)'
        unit = '(Jy/beam)' if source in ['alma','vla'] else unit

    # Select the quantity to plot
    if render.lower() in ["i", "intensity"]:
        figname = f"{source}_I.fits"
        cblabel = f"Stokes I {unit}"
        # Rescale to micro Jy/px
        rescale = 1e6 if rescale is None and source == 'polaris' else rescale

    elif render.lower() in ["q"]:
        figname = f"{source}_Q.fits"
        cblabel = f"Stokes Q {unit}"
        # Rescale to micro Jy/px
        rescale = 1e6 if rescale is None and source == 'polaris' else rescale

    elif render.lower() in ["u"]:
        figname = f"{source}_U.fits"
        cblabel = f"Stokes U {unit}"
        # Rescale to micro Jy/px
        rescale = 1e6 if rescale is None and source == 'polaris' else rescale

    elif render.lower() in ["tau", "optical depth"]:
        figname = f"{source}_tau.fits"
        cblabel = "Optical depth"
        rescale = 1 if rescale is None else rescale

    elif render.lower() in ["pi", "poli", "polarized intensity"]:
        figname = f"{source}_pi.fits"
        cblabel = f"Polarized intensity {unit}"
        # Rescale to micro uJy/px
        rescale = 1e6 if rescale is None and source == 'polaris' else rescale

    elif render.lower() in ["pf", "pfrac", "polarization fraction"]:
        figname = f"{source}_pf.fits"
        cblabel = "Polarization fraction"
        rescale = 1 if rescale is None else rescale
        bright_temp = False

    elif render.lower() in ["pa", "pangle", "polarization angle"]:
        figname = f"{source}_pa.fits"
        cblabel = "Polarization angle (deg)"
        rescale = 1 if rescale is None else rescale
        bright_temp = False

    else:
        rescale = 1
        raise ValueError("Wrong value for render. Must be i, q, u, pf, pi, pa or tau.")

    # Plot the render quantity a colormap
    fig = plot_map(
        figname, 
        rescale=rescale, 
        cblabel=cblabel, 
        bright_temp=bright_temp, 
        scalebar=scalebar, 
        verbose=verbose, 
        block=False,
        *args,
        **kwargs,
    )
    
    # Set the image size
    if mapsize is not None:
        D = 141*u.pc
        if mapsize.unit == 'au':
            img_size = ((mapsize / 2).to(u.pc) / D) * u.rad.to(u.deg)

        elif mapsize.unit == 'pc':
            img_size = (mapsize / 2 / D) * u.rad.to(u.deg)

        elif mapsize.unit in ['arcsec']:
            img_size = (mapsize / 2)*u.arcsec.to(u.deg)

        else:
            img_size = mapsize / 2 

        fig.recenter(hdr.get('CRVAL1'), hdr.get('CRVAL2'), radius=img_size)

    rotate = rotate if source == 'obs' else int(rotate) - 90

    # Add polarization vectors
    fig.show_vectors(
        f"{source}_pf.fits",
        f"{source}_pa.fits",
        step=step,
        scale=scale,
        rotate=rotate,
        color=vector_color,
        linewidth=vector_width, 
        units='degrees', 
        layer="pol_vectors",
    )
    fig.refresh()

    # Add B-field vectors
    # TO DO: ADD A LABEL TO INDICATE WHAT VECTORS ARE WHAT
    if add_bfield:
        print_("Adding magnetic field lines.")

        B = Bfield()

        write_fits("B.fits", data=B.get_strength(const_pfrac), header=hdr)
        write_fits("Bangle.fits", data=B.get_angle(), header=hdr)

        fig.show_vectors(
            "B.fits",
            "Bangle.fits",
            step=step,
            scale=scale,
            rotate=0,
            color="tab:green",
            zorder=1,
            layer="B_vectors",
        )
    
    # Plot the tau = 1 contour when optical depth is plotted    
    if render.lower() in ["tau", "optical depth"]:
        fig.show_contour(
            f'{source}_tau.fits', 
            levels=[1],
            colors='green',
        )

    return plot_checkout(fig, show, savefig)


def spectral_index(
    lam1_,
    lam2_,
    use_aplpy=True,
    cmap="PuOr",
    scalebar=20*u.au,
    vmin=None,
    vmax=None,
    mask=None,
    figsize=None, 
    show=True,
    savefig=None,
    return_fig=False, 
    savefile=True, 
    verbose=False, 
    *args,
    **kwargs
):
    """Calculate the spectral index between observations at two wavelengths.
    lam1 must be shorter than lam2.
    """
    # Read data from Fits file
    lam1, hdr1 = fits.getdata(lam1_, header=True)
    lam2, hdr2 = fits.getdata(lam2_, header=True)
    lam1 = lam1.squeeze()
    lam2 = lam2.squeeze()

    # Calculate the spectral index
    nu1 = hdr1.get("restfrq")
    nu2 = hdr2.get("restfrq")
    alpha = np.log10(lam1 / lam2) / np.log10(nu1 / nu2)

    # Mask the spectral index in regions where S_1.3mm < "mask" [Jy/bm]
    # TO DO: Implement a proper error propagation to mask on the error
    # of the spectral index and not on the image's rms.
    if mask is not None:
        alpha[lam1 < mask] = np.NaN

    # Plot using APLPy if possible, otherwise fallback to Matplotlib
    if use_aplpy:
        try:
            if savefile is True:
                specindex = 'spectral_index.fits'
            elif isinstance(savefile, (str, Path)):
                specindex = str(savefile)

            write_fits(specindex, data=alpha, header=set_hdr_to_iras16293B(hdr1))
            fig = plot_map(
                specindex,
                cblabel="Spectral index",
                scalebar=scalebar,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                figsize=figsize, 
                bright_temp=False, 
                verbose=verbose,
                checkout=False, 
                *args, 
                **kwargs
            )
            # Plot the beam ony if it is an ALMA simulated observation
            if all(['alma' in [lam1_, lam2_]]) or all(['vla' in [lam1_, lam2_]]): 
                fig.show_contour(specindex, colors="black", levels=[2])
                fig.add_beam(facecolor='white', edgecolor='black', linewidth=3)
            else:
                fig.show_contour(specindex, colors="black", levels=[2])

            # Delete the temporal file after the plot is done, unless savefile is True
            if savefile in [False, None, '']: 
                os.remove(specindex)

        except Exception as e:
            plt.close()
            print_(f"Imposible to use aplpy: {e}", True, bold=False, red=True)
            print_(f"Plotting with Matplotlib...", True, bold=False)

            fig = plt.figure(figsize=figsize)
            alpha = np.flipud(alpha)
            plt.imshow(alpha, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(pad=0.01).set_label(r"$\alpha_{223-100 {\rm GHz}}$")
            plt.contour(alpha, colors='black', levels=[2])
            plt.xticks([])
            plt.yticks([])

    return plot_checkout(fig, show, savefig) if return_fig else alpha


def Tb(data, outfile="", freq=0, bmin=0, bmaj=0, overwrite=False, verbose=False):
    """
    Convert intensities [Jy/beam] into brightness temperatures [K].
    Frequencies must be in GHz and bmin and bmaj in arcseconds.
    Frequencies and beams are read from header if possible. 
    """

    # Detects whether data flux comes from a file or an array
    if isinstance(data, (str,PosixPath)):
        data, hdr = fits.getdata(data, header=True)
    else:
        hdr = {}

    # Drop empty axes
    flux = np.squeeze(data)

    # Get the frequency from the header if not provided
    if freq == 0:
        freq = hdr.get("RESTFRQ", default=freq) * u.Hz
        freq = freq.to(u.GHz)
    else:
        freq *= u.GHz

    # Get the beam minor and major axis from the header if not provided
    if bmin == 0:
        bmin = hdr.get("BMIN", default=bmin) * u.deg
        bmin = bmin.to(u.arcsec)
    else:
        bmin *= u.arcsec

    if bmaj == 0:
        bmaj = hdr.get("BMAJ", default=bmaj) * u.deg
        bmaj = bmaj.to(u.arcsec)
    else:
        bmaj *= u.arcsec

    print_(
        f"Reading BMIN and BMAJ from header: "
        + f'{bmin.to(u.arcsec):1.3f}" x '
        + f'{bmaj.to(u.arcsec):1.3f}"',
        verbose=verbose,
    )

    # Convert the beam gaussian stddev into a FWHM and obtain the beam area
    fwhm_to_sigma = 1 / np.sqrt(8 * np.log(2))
    beam = 2 * np.pi * bmaj * bmin * fwhm_to_sigma ** 2

    # Convert surface brightness (jy/beam) into brightness temperature
    to_Tb = u.brightness_temperature(freq)
    temp = flux * (u.Jy / beam).to(u.K, equivalencies=to_Tb)

    # Write flux to fits file if required
    write_fits(outfile, temp, hdr, overwrite, verbose)

    return temp.value


def horizontal_cut(
    filename=None, 
    angles=None,
    add_obs=True,
    scale_obs=None,
    axis=0,
    cut_along="max", 
    lam="3mm",
    amax=None,
    prefix="",
    align=True, 
    show=True,
    savefig=None,
    bright_temp=True,
    return_data=False, 
    verbose=False, 
    *args,
    **kwargs,
):
    """Perform horizontal cuts through the position of the peak in the image."""

    def angular_offset(d, hdr, cut_along=cut_along):
        """Calculate the angular offset (assumes angular scale is in degrees)"""
        cdelt1 = hdr.get("CDELT1") * u.deg.to(u.arcsec)
        naxis1 = hdr.get("NAXIS1")
        FOV = naxis1 * cdelt1

        if cut_along in ["max", "peak"]:
            # Find the peak in the image and cut along the given axis
            cut = maxpos(d)[axis]
            if axis == 0:
                cut = d[cut, :]
            elif axis == 1:
                cut = d[:, cut]
        elif cut_along in ["min"]:
            # Find the min in the image and cut along the given axis
            cut = minpos(d)[axis]
            if axis == 0:
                cut = d[cut, :]
            elif axis == 1:
                cut = d[:, cut]
        elif cut_along in ["c", "center"]:
            # Generate the cut along the center of the image. 
            img_center = int(d.shape[-1] / 2)
            if axis == 0:
                cut = d[img_center, :]
            elif axis == 1:
                cut = d[:, img_center]

        # Convert data into brightness temperature, if possible
        bright_temp = False if lam == 'alpha' else True
        if bright_temp:
            try:
                cut = Tb(
                    data=cut,
                    freq=hdr.get("RESTFRQ") * u.Hz.to(u.GHz),
                    bmin=hdr.get("bmin") * u.deg.to(u.arcsec),
                    bmaj=hdr.get("bmaj") * u.deg.to(u.arcsec),
                )
            except Exception as e:
                print_('Cannot convert to temperature. \
                Frequency or beam keywords not available', True, bold=True)

        # Offset from the center of the image
        offset = np.linspace(-FOV / 2, FOV / 2, naxis1)
    
        # Find the peak position along the cut
        if cut_along in ['max', 'peak']:
            cut_peak = np.nanargmax(cut)
        elif cut_along in ['min']:
            cut_peak = np.nanargmin(cut)
        elif cut_along in ['c', 'center']:
            if lam == 'alpha':
                cut_peak = np.nanargmin(cut)
            else:
                cut_peak = np.nanargmax(cut)

        # Shift the angular offset to be centered on the peak
        if align:
            offset = offset - offset[cut_peak]

        return offset, cut

    # Set the path prefix as a Path object, if provided
    prefix = Path(prefix)

    # Avoid opening a figure if show=False, in case pylab is enabled
    plt.rcParams['interactive'] = show

    # Create a figure object
    fig = plt.figure()

    plt.xlabel("Angular offset (arcseconds)")
    plt.xlim(-0.33, 0.33)
    if lam != 'alpha':
        ylabel_ = r"$T_{\rm b}$ (K)" if bright_temp else r"mJy/beam"
        plt.ylabel(ylabel_)

    # Plot the cut from the real observation if required
    if add_obs:
        # Read data
        obs = Observation(source="iras16293", lam=lam)

        # Drop empty axes, flip and rescale
        obs.drop_axis()
        obs.fliplr()
        if lam != 'alpha' and not bright_temp:
            obs.rescale(1e3)
        if scale_obs is not None and scale_obs > 0:
            obs.rescale(scale_obs)

        # Get a cut of brightness along the peak as a function of angular offset
        obs.offset, obs.cut = angular_offset(
            obs.data, obs.header, cut_along=cut_along)
        
        # Plot the observed profile
        label = f"{obs.source} (x{scale_obs:.1f})" if scale_obs else obs.source
        plt.plot(obs.offset, obs.cut, label=label, color="black", ls="-.")
    
        # If no other curve hass been added, return the obs data
        if return_data and filename is None and angles is None and amax is None:
            return (obs.offset, obs.cut)
    
    if filename is not None:
        # Plot the horizontal cut for a single file
        data, hdr = fits.getdata(filename, header=True)

        # Drop empty axes. Flip and rescale
        data = np.fliplr(np.squeeze(data))
        if lam != 'alpha' and not bright_temp:
            data *= 1e3

        offset, cut = angular_offset(data, hdr, cut_along=cut_along)
        plt.plot(
            offset,
            cut,
            *args,
            **kwargs,
        )
        plt.annotate(lam if lam != 'alpha' else 'Spectral\nindex', 
            xy=(0.08, 0.83), xycoords="axes fraction", size=18)

        if return_data:
            return (offset, cut)

    elif angles is not None:
        # Plot the cuts from the simulated observations for every inclination angle
        for angle in [f"{i}deg" for i in angles]:
            # Read data
            filename = prefix / f"amax{amax}/{lam}/{angle}/dust_scat/data/" /\
                f"{'alma' if lam in ['1.3mm','3mm'] else 'vla'}_I.fits"

            data, hdr = fits.getdata(filename, header=True)

            # Drop empty axes. Flip and rescale
            data = np.fliplr(np.squeeze(data))
            if not bright_temp:
                data *= 1e3

            offset, cut = angular_offset(data, hdr)
            plt.plot(
                offset,
                cut,
                label=f"{angle} ", 
                *args,
                **kwargs,
            )
        plt.legend() 

        if return_data:
            return {str(angle): (offset, cut) for angle in angles}

    return plot_checkout(fig, show, savefig)



def get_polaris_temp_spherical(binfile="grid_temp.dat"):
    """
    Read the binary output from a Polaris dust heating
    simulation in spherical coordinates and return the 
    dust temperature field.
    """
    import struct

    with open(binfile, "rb") as f:
        # Read grid ID
        ID = struct.unpack("H", f.read(2))

        # Read N quantities
        (n,) = struct.unpack("H", f.read(2))
        for q in range(n):
            struct.unpack("H", f.read(2))

        # Read radial boundaries
        r_in = struct.unpack("d", f.read(8))
        r_out = struct.unpack("d", f.read(8))

        # Read number of cells on each axis
        (n_r,) = struct.unpack("H", f.read(2))
        (n_t,) = struct.unpack("H", f.read(2))
        (n_p,) = struct.unpack("H", f.read(2))
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
            (temp[c],) = struct.unpack("d", f.read(8))

            # Move the pointer till the end of the row
            for q in range(3):
                struct.unpack("d", f.read(8))

        return temp.reshape(n_r, n_t, n_p)


def get_polaris_temp_voronoi(binfile="grid_temp.dat"):
    """
    Read the binary output from a Polaris dust heating
    in Voronoi grid structure and return the dust 
    temperature field.
    """
    import struct

    with open(binfile, "rb") as f:
        # Read grid ID
        ID = struct.unpack("H", f.read(2))

        # Read N quantities
        (n,) = struct.unpack("H", f.read(2))
        ids = list()
        for i in range(n):
            (q, ) = struct.unpack("H", f.read(2))
            ids.append(q)

        # Read number of cells
        (n_cells, ) = struct.unpack("d", f.read(8))
        (l_max, ) = struct.unpack("d", f.read(8))
        n_cells = int(n_cells)

        x = np.zeros(n_cells)
        y = np.zeros(n_cells)
        z = np.zeros(n_cells)
        temp = np.zeros(n_cells)

        # Iterate over cells
        for c in range(n_cells):
            # Read x coordinates
            (x[c], ) = struct.unpack("f", f.read(4))
            # Read y coordinates
            (y[c], ) = struct.unpack("f", f.read(4))
            # Read z coordinates
            (z[c], ) = struct.unpack("f", f.read(4))
            # Read cell volumes
            struct.unpack("d", f.read(8))

            # Read the first quantities
            temp_id = 2
            for i in range(ids.index(temp_id)):
                struct.unpack("f", f.read(4))

            # Read the temperature
            (temp[c],) = struct.unpack("f", f.read(4))

            # Then keep reading till the end of the row
            remaining_ids = len(ids) - ids.index(temp_id) - 1
            for q in range(remaining_ids):
                struct.unpack("f", f.read(4))

            # Read the number of neighbours
            (nn, ) = struct.unpack("f", f.read(4))
            for n in range(int(nn)):
                # To do: there's still a mismatch between the size of the
                # buffers and the original data. Double check!
                struct.unpack("f", f.read(4))

        return np.stack((x, y, z, temp))


@elapsed_time
def tau_surface(
    densfile='dust_density_3d.fits.gz', 
    tempfile='dust_temperature_3d.fits.gz', 
    prefix='', 
    tau=1, 
    los=0, 
    bin_factor=[1,1,1], 
    render='temperature', 
    plot_tau=True, 
    amax='100um', 
    convolve_map=True, 
    plot2D=False, 
    plot3D=True, 
    savefig=None, 
    verbose=True
):
    """ 
        Compute and plot the surface with optical depth = 1 within a 3D 
        density array from a FITS file.
    """

    # Prepend the prefix to the filenames
    densfile = Path(prefix/Path(densfile))
    tempfile = Path(prefix/Path(tempfile))

    from astropy.nddata.blocks import block_reduce

    print_(f'Reading density from FITS file: {densfile}', verbose)
    print_(f'Reading temperature from FITS file: {tempfile}', verbose)
    rho = fits.getdata(densfile).squeeze() * (u.kg/u.m**3).to(u.g/u.cm**3)
    temp = fits.getdata(tempfile).squeeze()
    hdr = fits.getheader(densfile)

    # Read the delta length of a given axis from the header
    dl = hdr[f'cdelt{[3, 2, 1][los]}'] * (u.m).to(u.cm)

    # Bin the array down before plotting, if required
    if bin_factor not in [1, [1,1,1]]:
        if isinstance(bin_factor, (int, float)):
            bin_factor = [bin_factor, bin_factor, bin_factor]

        print_(f'Original array shape: {temp.shape}', verbose)

        print_(f'Binning density grid ...', verbose)
        rho = block_reduce(rho, bin_factor, func=np.nanmean)

        print_(f'Binning temperature grid ...', verbose)
        temp = block_reduce(temp, bin_factor, func=np.nanmean)

        # Rescale also the delta length by the binning factor
        dl *= bin_factor[0]

        print_(f'Binned array shape: {temp.shape}', verbose)

    # Dust opacities for a mixture of silicates and graphites in units of cm2/g
    if amax == '10um':
        # Extinction opacity at 1.3 and 3 mm for amax = 10um
        kappa_1mm = 1.50
        kappa_3mm = 0.60
        kappa_7mm = 0.23
        kappa_18mm = 0.07
    elif amax == '100um':
        # Extinction opacity at 1.3 and 3 mm for amax = 1000um
        kappa_1mm = 2.30
        kappa_3mm = 0.75
        kappa_7mm = 0.31
        kappa_18mm = 0.12
    elif amax == '1000um':
        # Extinction opacity at 1.3 and 3 mm for amax = 1000um
        kappa_1mm = 12.88
        kappa_3mm = 6.120
        kappa_7mm = 1.27
        kappa_18mm = 0.09

    # In the case of grain growth, combine the optical depth of the 2 dust pops
    if amax != '100-1000um':
        print_(f'Plotting for amax = {amax}', verbose)
        sigma_3d_1mm = (rho * kappa_1mm * dl)
        sigma_3d_3mm = (rho * kappa_3mm * dl)
        sigma_3d_7mm = (rho * kappa_7mm * dl)
        sigma_3d_18mm = (rho * kappa_18mm * dl)

    else:
        print_(f'Plotting for combined amax = {amax}', verbose)
        # The following opacities are for amax100um including organics 
        # and amax1000um without organics (i.e., sublimated).
        sigma_3d_1mm = np.where(temp > 300, rho * 12.88 * dl, rho * 1.80 * dl)
        sigma_3d_3mm = np.where(temp > 300, rho * 6.120 * dl, rho * 0.55 * dl)

    # Integrate the (density * opacity) product to calculate the optical depth
    op_depth_1mm = np.cumsum(sigma_3d_1mm, axis=los)
    op_depth_3mm = np.cumsum(sigma_3d_3mm, axis=los)
    op_depth_7mm = np.cumsum(sigma_3d_7mm, axis=los)
    op_depth_18mm = np.cumsum(sigma_3d_18mm, axis=los)

    if plot2D and not plot3D:
        from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel

        # Set all tau < 1 regions to a high number
        op_thick_1mm = np.where(op_depth_1mm < 1, op_depth_1mm.max(), op_depth_1mm)
        op_thick_3mm = np.where(op_depth_3mm < 1, op_depth_3mm.max(), op_depth_3mm)
        op_thick_7mm = np.where(op_depth_7mm < 1, op_depth_7mm.max(), op_depth_7mm)
        op_thick_18mm = np.where(op_depth_18mm < 1, op_depth_18mm.max(), op_depth_18mm)

        # Find the position of the minimum for tau > 1
        min_tau_pos_1mm = np.apply_along_axis(np.argmin, 0, op_thick_1mm).squeeze()
        min_tau_pos_3mm = np.apply_along_axis(np.argmin, 0, op_thick_3mm).squeeze()
        min_tau_pos_7mm = np.apply_along_axis(np.argmin, 0, op_thick_7mm).squeeze()
        min_tau_pos_18mm = np.apply_along_axis(np.argmin, 0, op_thick_18mm).squeeze()
    
        Td_tau1_1mm = np.zeros(temp[0].shape)
        Td_tau1_3mm = np.zeros(temp[0].shape)
        Td_tau1_7mm = np.zeros(temp[0].shape)
        Td_tau1_18mm = np.zeros(temp[0].shape)

        # Fill the 2D arrays with the temp. at the position of tau=1
        for i in range(min_tau_pos_1mm.shape[0]):
            for j in range(min_tau_pos_1mm.shape[1]):
                Td_tau1_1mm[i,j] = temp[min_tau_pos_1mm[i,j], i, j]
                Td_tau1_3mm[i,j] = temp[min_tau_pos_3mm[i,j], i, j]
                Td_tau1_7mm[i,j] = temp[min_tau_pos_7mm[i,j], i, j]
                Td_tau1_18mm[i,j] = temp[min_tau_pos_18mm[i,j], i, j]
        
        # Convolve the array with the beam from the observations at 1.3 and 3 mm
        def fwhm_to_std(obs):
            scale = dl * u.cm.to(u.pc)
            bmaj = obs.header['bmaj']*u.deg.to(u.rad) * (141 / scale) 
            bmin = obs.header['bmin']*u.deg.to(u.rad) * (141 / scale) 
            bpa = obs.header['bpa']
            std_x = bmaj / np.sqrt(8 * np.log(2))
            std_y = bmin / np.sqrt(8 * np.log(2))
            return std_x, std_y, bpa
            
        if convolve_map:
            print_('Convolving 2D temperature maps', verbose=True)
            std_x, std_y, bpa = fwhm_to_std(Observation('1.3mm'))
            Td_tau1_1mm = convolve_fft(Td_tau1_1mm, Gaussian2DKernel(std_x, std_y, bpa))

            std_x, std_y, bpa = fwhm_to_std(Observation('3mm'))
            Td_tau1_3mm = convolve_fft(Td_tau1_3mm, Gaussian2DKernel(std_x, std_y, bpa))

            std_x, std_y, bpa = fwhm_to_std(Observation('7mm'))
            Td_tau1_7mm = convolve_fft(Td_tau1_7mm, Gaussian2DKernel(std_x, std_y, bpa))
        
            std_x, std_y, bpa = fwhm_to_std(Observation('18mm'))
            Td_tau1_18mm = convolve_fft(Td_tau1_18mm, Gaussian2DKernel(std_x, std_y, bpa))
        return Td_tau1_1mm.T, Td_tau1_3mm.T, Td_tau1_7mm.T, Td_tau1_18mm.T

        
    if plot3D:
        from mayavi import mlab
        from mayavi.api import Engine
        from mayavi.sources.parametric_surface import ParametricSurface
        from mayavi.modules.text import Text

        # Initialaze the Mayavi scene
        engine = Engine()
        engine.start()
        fig = mlab.figure(size=(1500,1200), bgcolor=(1,1,1), fgcolor=(0.5,0.5,0.5))

        # Select the quantity to render: density or temperature
        if render in ['d', 'dens', 'density']:
            render_quantity = rho 
            plot_label = r'log(Dust density (kg m^-3))' 
        elif render in ['t', 'temp', 'temperature']:
            render_quantity = temp 
            plot_label = r'Dust Temperature (K)' 

        # Filter the optical depth lying outside of a given temperature isosurface, 
        # e.g., at T > 100 K.
        op_depth_1mm[temp < 150] = 0
        op_depth_3mm[temp < 150] = 0
        op_depth_7mm[temp < 150] = 0
        op_depth_18mm[temp < 150] = 0

        # Plot the temperature
        rendplot = mlab.contour3d(
            render_quantity, 
            colormap='inferno', 
            opacity=0.5, 
            vmax=400, 
            contours=10, 
        )
        figcb = mlab.colorbar(
            rendplot, 
            orientation='vertical', 
            title='',
        )

        # Add the axes and outline of the box
        mlab.axes(ranges=[-100, 100] * 3, 
            xlabel='AU', ylabel='AU', zlabel='AU', nb_labels=5)

        # Plot the temperature
        densplot = mlab.contour3d(
            rho, 
            colormap='BuPu', 
            opacity=0.5, 
            contours=5, 
        )
        denscb = mlab.colorbar(
            densplot, 
            orientation='vertical', 
            title='Dust Density (g/cm^3)',
        )
        if plot_tau:
            # Plot optical depth at 1mm
            tauplot_1mm = mlab.contour3d(
                op_depth_1mm, 
                contours=[tau], 
                color=(0, 1, 0), 
                opacity=0.5, 
            )
           # Plot optical depth at 3mm
            tauplot_3mm = mlab.contour3d(
                op_depth_3mm,  
                contours=[tau], 
                color=(0, 0, 1), 
                opacity=0.7, 
            )
           # Plot optical depth at 7mm
            tauplot_7mm = mlab.contour3d(
                op_depth_7mm,  
                contours=[tau], 
                color=(0.59, 0.41, 0.27), 
                opacity=0.7, 
            )
           # Plot optical depth at 7mm
            tauplot_18mm = mlab.contour3d(
                op_depth_18mm,  
                contours=[tau], 
                color=(0.75, 0.34, 0.79), 
                opacity=0.7, 
            )

        # The following commands are meant to customize the scene and were 
        # generated with the recording option of the interactive Mayavi GUI.

        # Adjust the viewing angle for an edge-on projection
        scene = engine.scenes[0]
        scene.scene.camera.position = [114.123, -161.129, -192.886]
        scene.scene.camera.focal_point = [123.410, 130.583, 115.488]
        scene.scene.camera.view_angle = 30.0
        scene.scene.camera.view_up = [-0.999, -0.017, 0.014]
        scene.scene.camera.clipping_range = [90.347, 860.724]
        scene.scene.camera.compute_view_plane_normal()
        scene.scene.render()

        # Adjust the light source of the scene to illuminate the disk from the viewer's POV 
        #camera_light1 = engine.scenes[0].scene.light_manager.lights[0]
        #camera_light1 = scene.scene.light_manager.lights[0]
        #camera_light1.elevation = 90.0
        #camera_light1.intensity = 1.0
        ##camera_light2 = engine.scenes[0].scene_light.manager.lights[1]
        #camera_light2 = scene.scene_light.manager.lights[1]
        #camera_light2.elevation = 90.0
        #camera_light2.elevation = 0.7

        # Customize the iso-surfaces
        module_manager = engine.scenes[0].children[0].children[0]
        temp_surface = engine.scenes[0].children[0].children[0].children[0]
#        temp_surface.contour.minimum_contour = 70.0
#        temp_surface.contour.maximum_contour = 400.0
        temp_surface.actor.property.representation = 'surface'
        temp_surface.actor.property.line_width = 3.0
        if plot_tau:
            tau1_surface = engine.scenes[0].children[1].children[0].children[0]
            tau3_surface = engine.scenes[0].children[2].children[0].children[0]
            tau7_surface = engine.scenes[0].children[1].children[0].children[0]
            tau18_surface = engine.scenes[0].children[2].children[0].children[0]
            tau1_surface.actor.property.representation = 'wireframe'
            tau3_surface.actor.property.representation = 'wireframe'
            tau7_surface.actor.property.representation = 'wireframe'
            tau18_surface.actor.property.representation = 'wireframe'
            tau1_surface.actor.property.line_width = 3.0
            tau3_surface.actor.property.line_width = 4.0
            tau7_surface.actor.property.line_width = 3.0
            tau18_surface.actor.property.line_width = 4.0

        # Adjust the colorbar
        lut = module_manager.scalar_lut_manager
        lut.scalar_bar_representation.position = np.array([0.02, 0.2])
        lut.scalar_bar_representation.position2 = np.array([0.1, 0.63])
        lut.label_text_property.bold = False
        lut.label_text_property.italic = False
        lut.label_text_property.font_family = 'arial'
        lut.data_range = np.array([70., 400.])

        # Add labels as text objects to the scene
        parametric_surface = ParametricSurface()
        engine.add_source(parametric_surface, scene)        

        label1 = Text()
        engine.add_filter(label1, parametric_surface)
        label1.text = plot_label
        label1.property.font_family = 'arial'
        label1.property.shadow = True
        label1.property.color = (0.86, 0.72, 0.21)
        label1.actor.position = np.array([0.02, 0.85])
        label1.actor.width = 0.30

        if plot_tau:
            label2 = Text()
            engine.add_filter(label2, parametric_surface)
            label2.text = 'Optically thick surface at 1.3mm'
            label1.property.font_family = 'arial'
            label2.property.color = (0.31, 0.60, 0.02)
            label2.actor.position = np.array([0.02, 0.95])
            label2.actor.width = 0.38

            label3 = Text()
            engine.add_filter(label3, parametric_surface)
            label3.text = 'Optically thick surface at 3mm'
            label2.property.font_family = 'arial'
            label3.property.color = (0.20, 0.40, 0.64)
            label3.actor.position = [0.02, 0.915]
            label3.actor.width = 0.355

            label4 = Text()
            engine.add_filter(label4, parametric_surface)
            label4.text = 'Line of Sight'
            label4.property.font_family = 'times'
            label4.property.color = (0.8, 0.0, 0.0)
            label4.actor.position = np.array([0.63, 0.90])
            label4.actor.width = 0.20

        if savefig is not None:
            scene.scene.save(savefig)

        if plot_tau:
            return render_quantity, op_depth_1mm, op_depth_3mm, op_depth_7mm, op_depth_18mm
        else:
            return render_quantity


def dust_mass(temp, flux, lam='1.3mm', gdratio=100, d=141*u.pc):
    """ Calculate the disk gas mass using the observational approach, 
        e.g., assuming emission is optically thin and that temperature 
        is uniform across the disk (see Evans et al. 2017, eq. 2). 

        Example:
            utils.disk_mass(100*u.K, 8*u.mJy/u.beam, lam='3mm')
    """
    from astropy.modeling import models
    
    # Extinction opacity at 1.3 and 3 mm in cgs for amax=10um
    kappa = {
        '0.07mm': 675 * (u.cm**2 / u.g),
        '1mm': 3.32 * (u.cm**2 / u.g), 
        '2mm': 0.85 * (u.cm**2 / u.g), 
        '1.3mm': 1.49765 * (u.cm**2 / u.g), 
        '3mm': 0.58061 * (u.cm**2 / u.g)
    }

    # Turn the flux from Jy/beam to Jy/sr
    flux = flux.to(u.erg * u.s**-1 * u.cm**-2 * u.Hz**-1)
 
    # Compute the specific Black-Body function at a given temperature
    B = models.BlackBody(temp.to(u.K))
    B = B(1.3*u.mm if lam == '1.3mm' else 3*u.mm)

    # Compute the gas mass (everything is now in cgs)
    mass = (gdratio * flux.value * d**2) / (kappa[lam] * B.value)

    return mass.to(u.Msun)


def plot_web(array):
    """ Generate a web-based plot of a 3D NumPy array.
        The figure is generated with Dash (Plotly) and
        displayed in a web browser.
        Plot a Gaussian Kernel by default, just for illustration
        purposes.
    """

    import plotly.graph_objects as go

    # Create a 3D grid
    X, Y, Z = np.mgrid[
        -50:50:array.shape[0]*1j, 
        -50:50:array.shape[1]*1j, 
        -50:50:array.shape[2]*1j
    ]

    # Generate the 3D figure 
    fig = go.Figure(data=go.Isosurface(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=array.flatten(),
        opacity=0.6, 
        isomin=90, 
        isomax=400,
        surface_count=10, 
        colorscale='inferno', 
        showscale=True,
        caps=dict(x_show=True, y_show=True),
        ))  

    fig.update_layout()
    fig.show()


def plot_bfield_projection(filename='00260.hdf5', axis='z'):
    """ Plot a projection of the B field from the output of 
        zeusTW. It assumes the output from zeusTW was 
        interpolated into a Cartesian grid and dumped as an 
        HDF5 file.
        It generates the vector plot using APLPy.
    """
    import h5py

    # Read the data
    f = h5py.File(filename, 'r')
    
    # Read and generate a coordinate grid
    coor = f['coord']
    dcc = coor['dcc']
    rcc = coor['rcc']
    dx, dy, dz = np.meshgrid(dcc, dcc, dcc, sparse=True, indexing='ij')
    
    # Turn every data field into a numpy array
    data = {key: np.array(val) for key, val in f['data'].items()}
    rho = abs(data['rho_interp'])
    bx = data['Bx_interp']
    by = data['By_interp']
    bz = data['Bz_interp']

    # Compute the column density
    mu = 2.36
    m_H = c.m_p.cgs.value
    N_x = np.sum(rho * dx, axis=0).T / (m_H * mu)
    N_y = np.sum(rho * dy, axis=1).T / (m_H * mu)
    N_z = np.sum(rho * dz, axis=2).T / (m_H * mu)
    N = {'x': N_x, 'y': N_y, 'z': N_z}
    N = np.log10(N[axis])

    # Generate the figure
    fig, ax = plt.subplots(1,1)
    img = ax.imshow(N)
    fig.colorbar(img, ax=ax)

    return fig

    
def dust_temp_in_steady_state(regime='thick', alpha=0.02, r=1, between=[0, 1e3]):
    """
    Implement root finding to solve numerically for the dust temperature
    in a steady state, i.e., when radiative cooling/heating = viscous heating.
    """
    from scipy.optimize import fsolve

    T_irr = (420 * r**(-0.5))
    sigma = 100 * u.g / u.cm**2 
    kappa_0 = 5e-4 * u.cm**2 / u.g / u.K**2
    omega = 1.4e-10 / u.s
    k_B = c.k_B.cgs
    sb = c.sigma_sb.cgs
    m_H2 = 2 * c.m_p.cgs

    # Optically thick limit
    if regime == 'thick':
        a_ = (sb / sigma / kappa_0).value
        b_ = ((9/4) * alpha * sigma * omega * k_B / m_H2).value
        c_ = ((sb / sigma / kappa_0) * T_irr**4).value

        func = lambda T: a_*T**2 - b_*T - c_*T**-2

    # Optically thin limit
    elif regime == 'thin':
        a_ = (sb * sigma * kappa_0).value
        b_ = (sb * kappa_0 * sigma * T_irr**4).value
        c_ = ((9/4) * alpha * sigma * omega * k_B / m_H2).value

        func = lambda T: a_*T**6 - b_*T**2 - c_*T

    else:
        # Default to optically thick
        print_('regime not defined. Assuming optically thick.', True)
        dust_temp_in_steady_state(regime='thick')

    # Call the root finding function
    return fsolve(func, between)


@elapsed_time
def interpolate_points(f=None, npoints=4, values=None, x=None, y=None, z=None, dims=3, render='interp', show=True, savefig=None):
    """
    Interpolate a given function in 2D or 3D
    """
    from scipy.interpolate import griddata

    if dims == 3:
        a = -1
        b = 1
        if None in [x.any(), y.any(), z.any()]:
            # Construct a grid
            grid_size = 50
            x = np.linspace(a, b, grid_size)
            y = np.linspace(a, b, grid_size)
            z = np.linspace(a, b, grid_size)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Define a function if not provided (a sphere)
        if f is None:
            f = lambda x, y, z: (x*x + y*y + z*z)**0.5
            f = lambda x, y, z: (x*x + y*y - z*z)**0.5

        # Generate random points to evaluate the function 
        npoints = int(npoints)
        rng = np.random.default_rng()
        points = (b - a) * rng.random((npoints, dims)) + a
        values = f(points[:,0], points[:,1], points[:,2])


        interp = griddata(
            points, f(points[:,0], points[:,1], points[:,2]), (X,Y,Z), 'linear')

        if show:
            from mayavi import mlab
            from mayavi.api import Engine

            # Initialaze the Mayavi scene
            engine = Engine()
            engine.start()

            fig = mlab.figure(size=(1200,1100), bgcolor=(1,1,1), fgcolor=(0.5,0.5,0.5))

            # Choose the field to plot: the original function or the interpolation
            field = {'interp': interp, 'f': f(X, Y, Z)}

            i = mlab.contour3d(X, Y, Z, 
                field[render], 
                opacity=0.75, 
                contours=10, 
                colormap='viridis'
            )
            p = mlab.points3d(points[:,0], points[:,1], points[:,2], 
                np.ones(npoints), 
                color=(0,0,0),
                opacity=0.01,
            )

            # Define the representation for the points and the interpolation
            ii = engine.scenes[0].children[0].children[0].children[0]
            pp = engine.scenes[0].children[1].children[0].children[0]
            ii.actor.property.representation = 'surface'
            pp.actor.property.representation = 'points'
            pp.actor.property.point_size = 4
        
            # Add a bounding box
            mlab.outline()
            
            # Save the scene to image
            if savefig is not None and isinstance(savefig, str):
                scene = engine.scenes[0]
                scene.scene.camera.compute_view_plane_normal()
                scene.scene.render()
                scene.scene.save(savefig)
                mlab.close()

    elif dims == 2:
        from tqdm import tqdm

        # Construct a grid
        a = -1
        b = 1
        if None in [x, y]:
            grid_size = 100
            x = np.linspace(a, b, grid_size)
            y = np.linspace(a, b, grid_size)
            
        X, Y = np.meshgrid(x, y)
            
        # Define a function if not provided (a circle)
        f = lambda x, y: (x*x + y*y)**0.5

        # Generate random points to evaluate the function
        rng = np.random.default_rng()
        points = (b - a) * rng.random((int(npoints), dims)) + a
        values = f(points[:,0], points[:,1])

        if show:
            plt.figure(figsize=(10,10))

            plt.subplot(211)
            plt.title(r'$f(x,y) = \sqrt{x^2+y^2}$\\'+f'sampled at {int(n)} random points')
            plt.imshow(f(X,Y).T, extent=(a,b,a,b), origin='lower')
            plt.plot(points[:,0], points[:,1], 'k.', ms=2, color='black')
            plt.yticks([])
            plt.xticks([])

            plt.subplot(212)
            plt.title('Linear interpolation')
            plt.imshow(griddata(points, values, (X,Y), method='linear').T,
                extent=(a,b,a,b), origin='lower')
            plt.yticks([])
            plt.xticks([])

            plt.tight_layout()
            plt.show()


@elapsed_time
def interpolate_grid(ncells=100, points='all_frames/snap_541.dat', values=None, field='temp', r_out=None, show=True, verbose=True):
    """
        Interpolate a set of points in cartesian coordinates along with their
        values into a rectangular grid.
        The box can be trimmed to a given radius r_out in units of au.
    """
    from scipy.interpolate import griddata
    from mayavi import mlab
    from tqdm import tqdm

    if isinstance(points, str):
        # Read point coordinates and values from file
        print_(f'Reading point coordinates from file: {points}', verbose)
        sph = read_sph(points, remove_sink=True)
        x = sph[:, 2]
        y = sph[:, 3]
        z = sph[:, 4]
        values = {'temp': sph[:, 10], 'dens': sph[:, 11]}[field]

    else:
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

    # Convert cartesian to spherical coordinates
    r = np.sqrt(x*x + y*y + z*z)

    # Trim particles outside of a given radius
    if r_out is not None:
        print_(f'Trimming the box to a radius of {r_out} au ...', verbose)
        r_out = r_out * u.au.to(u.cm)
        
        to_remove = []
        for i in tqdm(range(x.size)):
            if r[i] > r_out:
                to_remove.append(i)

        x = np.delete(x, to_remove)
        y = np.delete(y, to_remove)
        z = np.delete(z, to_remove)
        values = np.delete(values, to_remove)
        print_(f'{r.size - x.size} particles were not included in the grid', verbose)

    # Construct the rectangular grid
    print_('Constructing the grid ...', verbose)
    r = np.linspace(x.min(), x.max(), ncells)
    t = np.linspace(y.min(), y.max(), ncells)
    p = np.linspace(z.min(), z.max(), ncells)
    R, T, P = np.meshgrid(r, t, p)

    # Interpolate the points and values at the grid points
    print_('Interpolating the point values onto the grid ...', verbose)
    interp = griddata(np.vstack([x,y,z]).T, values, (R,T,P), 'linear')

    # Render the interpolated 3D field using Mayavi
    if show:
        print_('Visualizing the interpolated field ...', verbose)
        mlab.contour3d(interp, contours=20, opacity=0.2)
    
    return interp


