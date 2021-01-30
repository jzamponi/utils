"""
    Collection of useful functions for my thesis.
"""
import os
import sys
import time
from pathlib import Path, PosixPath

import numpy as np

import matplotlib.pyplot as plt

from astropy.io import ascii, fits
from astropy import units as u
from astropy import constants as c

home = Path.home()
pwd = Path(os.getcwd())


class color:
    fail = "\033[91m"
    bold = "\033[1m"
    none = "\033[0m"


class Observation:
    """ Contains data from real observations. """

    def __init__(self, name="", lam="3mm"):
        self.name = name
        self.data, self.header = fits.getdata(
            home / f"phd/polaris/sourceB_{lam}.fits", header=True
        )
        self.data, self.header = fits.getdata(
            home / f"phd/polaris/sourceB_{lam}.fits", header=True
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
    def __init__(self, hdr):
        self.hdr = hdr


class Bfield:
    def __init__(self):
        # Read bfield data from Fits file
        self.data = fits.getdata(home / "phd/zeusTW/scripts/bfield_faceon.fits")
        # x,y-components from B field
        self.x = data[2]
        self.y = data[3]

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

        return angle


def print_(string, verbose=None, fname=None, bold=False, fail=False, *args, **kwargs):

    # Get the name of the calling function by tracing one level up in the stack
    fname = sys._getframe(1).f_code.co_name if fname is None else fname

    # Check if verbosity state is defined as a global variable
    if verbose is None:
        if "VERBOSE" in globals() and VERBOSE:
            verbose = True

    if verbose:
        if bold:
            print(f"{color.bold}[{fname}] {string} {color.none}", *args, **kwargs)
        elif fail:
            print(f"{color.fail}[{fname}] {string} {color.none}", *args, **kwargs)
        else:
            print(f"[{fname}] {string}", *args, **kwargs)


def write_fits(filename, data, header=None, overwrite=True, verbose=False):
    # Get the name of the calling function by tracing one level up in the stack
    caller = sys._getframe(1).f_code.co_name

    if filename != "":
        if overwrite and os.path.exists(filename):
            print_("Overwriting file ...", verbose=verbose, fname=caller)
            os.remove(filename)

        fits.HDUList(fits.PrimaryHDU(data=data, header=header)).writeto(filename)
        print_(f"Written file {filename}", verbose=verbose, fname=caller)


def elapsed_time(caller):
	""" Decorator designed to print the time taken by a functon. """
	# TO DO: Find a way to forward verbose from the caller even when
	# is not provided explicitly, so that it takes the default value 
	# from the caller.

	def wrapper(*args, **kwargs):
		# Measure time before it runs
		start = time.time()

		# Execute the caller function
		f = caller(*args, **kwargs)

		# Measure time difference after it finishes
		run_time = time.time() - start

		# Print the elapsed time nicely formatted, if verbose is enabled
		print_(
			f'Elapsed time: {time.strftime("%H:%M:%S", time.gmtime(run_time))}', 
			#verbose = kwargs.get('verbose'), 
			verbose = True, 
			fname = caller.__name__
		)
		return f
	return wrapper


def ring_bell(soundfile=None):
    """ Play a sound from system. Useful to notify when another function finishes."""
    if not isinstance(soundfile, (str,PosixPath)):
        soundfile = "/usr/share/sounds/freedesktop/stereo/service-login.oga"

    os.system(f"paplay {soundfile} >/dev/null 2>&1")


def plot_checkout(fig, show, savefig, path=""):
    """Final step in every plotting routine:
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
        plt.show()

    return fig


def parse(s, delimiter="%", d=None):
    """
    Parse a string containing a given delimiter and return a dictionary
    containing the key:value pairs.
    """
    # Set the delimiter character
    delimiter = d if isinstance(d, (str,PosixPath)) else d

    # Store all existing global and local variables
    g = globals()
    l = locals()

    string = s.replace(d, "{")
    string = s.replace("{_", "}_")

    # TO DO: This function is incomplete.
    return string


def set_hdr_to_iras16293B(
    hdr, wcs="deg", spec_axis=False, stokes_axis=False, for_casa=False, verbose=False
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
        [hdr.remove(k, True) for k in ["PC1_3", "PC2_3", "PC3_3", "PC3_1", "PC3_2", "PC4_2", "PC4_3", "PC2_4", "PC3_4", "PC4_4", "PC4_1", "PC1_4"]]

    # Adjust the header to match obs. from IRAS16293-2422B
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


@elapsed_time
def create_cube(
    filename="polaris_detector_nr0001.fits.gz",
    outfile="",
    wcs="deg",
    spec_axis=False,
    stokes_axis=False,
    add_selfscat=False,
    for_casa=True,
    overwrite=False,
    verbose=False,
):
    """
    Retrieves data and header from filename and add necessary keywords to the cube.
    NOTE: Wildcards are allowed by the infile argument. Thanks to glob.
    """
    from glob import glob

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
    hdr = set_hdr_to_iras16293B(
        hdr, wcs=wcs, spec_axis=spec_axis, stokes_axis=stokes_axis, for_casa=for_casa
    )

    # Add emission by self-scattering if required
    if add_selfscat:
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


def read_sph(snapshot="snap_541.dat"):
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

    with open(snapshot, "rb") as f:
        names = f.readline()[1:].split()
        data = np.fromstring(f.read()).reshape(-1, len(names))
        data = data.astype("f4")

    return data


def radmc3d_data(file_, npix=300, sizeau=50, distance=3.086e18 * u.m.to(u.cm)):
    """
    Function to read image files resulting from an RT with RADMC3D.
    """

    print(f'[radmc_data] Reading file {file_.split("/")[-1]}')
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


def fill_gap(
    filename,
    outfile=None,
    x1=143,
    x2=158,
    y1=143,
    y2=157,
    threshold=1e-5,
    incl="0deg",
    savefile=True,
):
    """
    Fill the central gap from polaris images with the peak flux
    """

    # Read data
    d, hdr = fits.getdata(filename, header=True)
    d = d[0][0]
    full = d
    gap = np.where(d[x1:x2, y1:y2] < threshold, d.max(), d[x1:x2, y1:y2])
    full[x1:x2, y1:y2] = gap
    plt.imshow(full, cmap="magma")
    plt.colorbar()
    plt.show()

    if outfile is None:
        outfile = filename.split(".fits")[0] + "_nogap.fits"

    if savefile:
        fits.writeto(outfile, data=full, header=hdr, overwrite=True)


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
    data,
    func=np.nanmean,
    step=1,
    dr=None,
    return_radii=False,
    show=False,
    savefig=None,
    nthreads=1,
	verbose=True,
    *args,
    **kwargs,
):
    """
    Computes the radial profile (average by default) of a 2D array
    by averaging the values within consecutive concentric circumferences
    from the border to the center.
    """
    from concurrent.futures import ThreadPoolExecutor

    def parallel_masking(i, r):
        """Function created to parallelize the for loop
        by means of a parallell map(func, args).
        """
        # Copy data to avoid propagating NaNs in the original array
        d_copy = np.copy(data)
        mask = circular_mask(d_copy.shape, center, r, ring=True)
        d_copy[~mask] = float("nan")
        averages[i] = func(d_copy)
        del d_copy

    # Read data from fits file if filename is provided
    if isinstance(data, (str,PosixPath)):
        data, hdr = fits.getdata(data, header=True)

    # Drop empty axes
    data = data.squeeze()

    # Get the center of the array
    map_radius_x = int(data[0].size / 2)
    map_radius_y = int(data[1].size / 2)
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

    # Generate the radial axis for plotting if required
    if return_radii or show or savefig:
        radii = radii[::-1] * dr.value

    # Plot the radial profile if required
    if show or savefig is not None:
        plt.semilogx(radii, averages, *args, **kwargs)
        plt.xlabel(f"Radius ({dr.unit})")
        # plt.xlim(radii.min(), radii.max())
        if isinstance(savefig, (str,PosixPath)) and len(savefig) > 0:
            plt.save(savefig)
        if show:
            plt.show()

    return (radii, averages) if return_radii else averages


def stats(filename, slice=None, verbose=False):
    """
    Compute the statistics of a file.
    """

    # Read data
    if isinstance(filename, (str,PosixPath)):
        data, hdr = fits.getdata(filename, header=True)

        if isinstance(slice, int):
            data = data[slice]
        elif isinstance(slice, list) and len(slice) == 2:
            data = data[slice[0], slice[1]]
    else:
        data = np.array(filename)

    # Set the relevant quantities
    stat = {
        "max": data.max(),
        "mean": data.mean(),
        "min": data.min(),
        "std": data.std(),
        "maxpos": maxpos(data),
        "minpos": minpos(data),
    }

    # Print statistics if verbose enabled
    for label, value in stat.items():
        print_(f"{label}: {value}", verbose)

    return stat


def maxpos(data):
    """
    Return a tuple with the coordinates of a N-dimensional array.
    """
    # Read data from fits file if data is string
    if isinstance(data, (str,PosixPath)):
        data = fits.getdata(data)

    # Remove empty axes
    data = np.squeeze(data)

    return np.unravel_index(data.argmax(), data.shape)


def minpos(data):
    """
    Return a tuple with the coordinates of a N-dimensional array.
    """
    # Read data from fits file if data is string
    if isinstance(data, (str,PosixPath)):
        data = fits.getdata(data)

    # Remove empty axes
    data = np.squeeze(data)

    return np.unravel_index(data.argmin(), data.shape)


def add_comment(filename, comment):
    """
    Read in a fits file and add a new keyword to the header.
    """
    data, hdr = fits.getdata(filename, header=True)

    header["NOTE"] = comment

    write_fits(filename, data=data, header=header, overwrite=True)


def edit_keyword(filename, key, value, verbose=True):
    """
    Read in a fits file and change the value of a given keyword.
    """
    data, hdr = fits.getdata(filename, header=True)

    value_ = hdr.get(key, default=None)
    if value_ is None:
        print_(f"Keyword {key} unexistent. Adding it ...", verbose=verbose)

    else:
        print_(f"Keyword {key} already exists.", verbose=verbose)
        print_(f"Changing it from {value_} to {value}.", verbose=verbose)

    hdr[key] = value

    write_fits(filename, data=data, header=hdr, overwrite=True)


def plot_map(
    filename,
    header=None,
    savefig=None,
    rescale=1,
    cblabel=None,
    scalebar=50 * u.au,
    cmap="magma",
    verbose=True,
    bright_temp=True,
    figsize=None,
    vmin=None, 
    vmax=None, 
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

    # Remove non-celestial WCS
    if hdr.get("NAXIS") > 2:
        tempfile = Path(filename).parent/'.temp_file.fits'
        write_fits(
            tempfile, 
            data=data.squeeze(), 
            header=set_hdr_to_iras16293B(hdr), 
            overwrite=True
        )
        filename = tempfile

    # Convert Jy/beam into Kelvin if required
    if bright_temp:
        rescale = Tb(
            data=rescale,
            freq=hdr.get("RESTFRQ") * u.Hz.to(u.GHz),
            bmin=hdr.get("bmin") * u.deg.to(u.arcsec),
            bmaj=hdr.get("bmaj") * u.deg.to(u.arcsec),
        )

    fig = FITSFigure(str(filename), rescale=rescale, figsize=figsize, *args, **kwargs)
    fig.show_colorscale(cmap=cmap, vmax=vmax, vmin=vmin)

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
            cblabel = hdr.get("BTYPE")
        except Exception as e:
            print_(e)
            cblabel = ""


    # Colorbar
    fig.add_colorbar()
    fig.colorbar.set_location("top")
    fig.colorbar.set_axis_label_text(cblabel)
    fig.colorbar.set_axis_label_font(size=15, weight=15)
    fig.colorbar.set_font(size=15, weight=15)

    # Frame and ticks
    fig.frame.set_color("black")
    fig.frame.set_linewidth(1.2)
    fig.ticks.set_color("black")
    fig.ticks.set_linewidth(1.2)
    fig.ticks.set_length(6)
    fig.ticks.set_minor_frequency(5)

    # Scalebar
    # TO DO: aplpy claims alma images have no celestial WCS, no scalebar allowed
    try:
        D = 141 * u.pc
        scalebar_ = (scalebar.to(u.cm) / D.to(u.cm)) * u.rad.to(u.arcsec)
        fig.add_scalebar(scalebar_ * u.arcsec)
        fig.scalebar.set_color("grey")
        fig.scalebar.set_corner("bottom right")
        fig.scalebar.set_font(size=23)
        fig.scalebar.set_linewidth(3)
        fig.scalebar.set_label(f"{int(scalebar.value)} {scalebar.unit}")
    except Exception as e:
        print_(f'Not able to add scale bar. Error: {e}', verbose=True, fail=True)

    if isinstance(savefig, (str,PosixPath)) and len(savefig) > 0:
        fig.save(savefig)

    # Delete the temporary file created to get rid of extra dimensions
    if hdr.get("NAXIS") > 2 and os.path.isfile(tempfile): os.remove(tempfile)

    return fig


@elapsed_time
def polarization_map(
    filename="polaris_detector_nr0001.fits.gz",
    render="intensity",
    wcs="deg",
    rotate=90,
    step=20,
    scale=50,
    fmin=None,
    fmax=None,
    savefig=None,
    show=True,
    vector_color="tab:purple",
    add_thermal=False,
    add_selfscat=False,
    add_bfield=False,
    const_bfield=False,
    const_pfrac=False,
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
        if "thermal emission" in hdr.get("ETYPE", ""):
            print_("You are adding thermal flux to the thermal flux.")
            print_("Not gonna happen.")
            I_th = np.zeros(I.shape)

        elif "scattered emission" in hdr.get("ETYPE", ""):
            print_("Adding thermal flux to the self-scattered flux.")
            thermal_file = pwd.replace("dust_scattering", "dust_emission")
            try:
                I_th = fits.getdata(thermal_file + "/" + filename)[0][0]
            except OSError:
                raise FileNotFoundError(
                    f'File with thermal flux does not exist.\n\
										File: {thermal_file+"/"+filename}'
                )

    elif isinstance(add_thermal, (str,PosixPath)):
        try:
            I_th = fits.getdata(add_thermal)
        except OSError:
            raise FileNotFoundError(
                f"File with thermal flux does not exist.\n" + "File: {add_thermal}"
            )

    else:
        I_th = np.zeros(I.shape)

    # Add self-scattered emission to the thermal emission if required
    if add_selfscat:
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
                    I = fits.getdata(I_th_unpol + "/" + filename)[0][0]
                except OSError:
                    raise FileNotFoundError(
                        f'File with thermal flux does not exist.\n\
											File: {I_th_unpol+"/"+filename}'
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

    elif isinstance(add_thermal, (str,PosixPath)):
        try:
            I_ss = fits.getdata(add_selfscat)
        except OSError:
            raise FileNotFoundError(
                f"File with self-scattered flux does not exist.\n\
									File: {add_selfscat}"
            )

    else:
        I_ss = np.zeros(I.shape)

    # Add all sources of emission
    I = I + I_th + I_ss

    # Define the polarization fraction and angle
    pfrac = np.sqrt(U ** 2 + Q ** 2) / I if not const_pfrac else np.ones(Q.shape)
    pangle = 0.5 * np.arctan(U / Q) * u.rad.to(u.deg)

    # Edit the header to match the observation from IRAS16293B
    hdr = set_hdr_to_iras16293B(hdr)

    # Write quantities into fits files
    quantities = {"I": I, "Q": Q, "U": U, "tau": tau, "pfrac": pfrac, "pangle": pangle}
    for f, d in quantities.items():
        write_fits(f + ".fits", d, hdr)

    # Select the quantity to plot
    if render.lower() in ["intensity", "i"]:
        figname = "I.fits"
        cblabel = r"Stokes I ($\mu$Jy/pixel)"
        # Rescale to micro Jy/px
        rescale = 1e6

    elif render.lower() in ["q"]:
        figname = "Q.fits"
        cblabel = r"Stokes Q ($\mu$Jy/pixel)"
        # Rescale to micro Jy/px
        rescale = 1e6

    elif render.lower() in ["u"]:
        figname = "U.fits"
        cblabel = r"Stokes U ($\mu$Jy/pixel)"
        # Rescale to micro Jy/px
        rescale = 1e6

    elif render.lower() in ["tau", "optical depth"]:
        figname = "tau.fits"
        cblabel = r"Optical depth"
        rescale = 1

    elif render.lower() in ["pfrac", "p", "pol"]:
        figname = "pfrac.fits"
        cblabel = r"Polarization fraction (%)"
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
    fig = plot_map(
        figname, rescale=rescale, cblabel=cblabel, vmax=vmax, vmin=vmin, *args, **kwargs
    )

    # Temporal Patch. TO DO: put it right
    if "dust_scattering" in pwd:
        rotate = 0

    # Add polarization vectors
    fig.show_vectors(
        "pfrac.fits",
        "pangle.fits",
        step=step,
        scale=scale,
        rotate=rotate,
        color=vector_color,
        layer="pol_vectors",
    )

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

    if show:
        plt.show()

    if isinstance(savefig, (str,PosixPath)) and len(savefig) > 0:
        fig.save(savefig)

    return fig


def imsmooth(filename, bmaj, bmin):
    """ Own implementation of the imsmooth task from the CASA package. """
    pass


def spectral_index(
    lam1_,
    lam2_,
    beta=False,
    use_aplpy=True,
    cmap="PuOr",
    scalebar=30*u.au,
    vmin=None,
    vmax=None,
    figsize=None, 
    show=True,
    savefig=None,
    return_fig=True, 
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
    alpha = np.log10(lam1 / lam2) / np.log10(hdr1.get("restfrq") / hdr2.get("restfrq"))

    # Derive the opacity index
    beta_ = alpha - 2

    # Determine which spectral index to work with
    index = beta_ if beta else alpha

    # Plot using APLPy if possible, else fallback to Matplotlib
    if use_aplpy:
        try:
            index2file = ".spectral_index.fits"
            write_fits(index2file, data=index, header=set_hdr_to_iras16293B(hdr1))
            cblabel_freq = r"$_{ 223-100 {\rm GHz}}$"
            fig = plot_map(
                index2file,
                cblabel=r"$\beta$"+cblabel_freq if beta else r"$\alpha$"+cblabel_freq,
                stretch='linear',
                scalebar=scalebar,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                figsize=figsize, 
                bright_temp=False, 
                verbose=True,
                *args, 
                **kwargs
            )
            fig.show_contour(index2file, colors="black", levels=[1.7, 2, 3])
            fig.add_beam(facecolor='white', edgecolor='black', linewidth=3)
            if os.path.isfile(index2file): os.remove(index2file)

        except Exception as e:
            plt.close()
            print_(f"Imposible to use aplpy: {e}", verbose=True, bold=False, fail=True)
            spectral_index(lam1_, lam2_, use_aplpy=False, show=show, savefig=savefig)
    else:
        fig = plt.figure(figsize=figsize)
        plt.imshow(index, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(pad=0.01).set_label(r"$\alpha_{223-100 {\rm GHz}}$")
        plt.xticks([])
        plt.yticks([])

    return plot_checkout(fig, show, savefig) if return_fig else index


def Tb(data, outfile="", freq=0, bmin=0, bmaj=0, overwrite=False, verbose=False):
    """
    Convert intensities [Jy/beam] into brightness temperatures [K].
    Frequencies must be in GHz and bmin and bmaj in arcseconds.
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


@elapsed_time
def horizontal_cuts(
    angles,
    add_obs=False,
    scale_obs=None,
    axis=0,
    lam="3mm",
    amax="10um",
    prefix="",
    show=True,
    savefig=None,
    bright_temp=True,
    *args,
    **kwargs,
):
    """Self-explanatory."""

    def angular_offset(d, hdr):
        """Calculate the angular offset (assumes angular scale is in degrees)"""
        cdelt1 = hdr.get("CDELT1") * u.deg.to(u.arcsec)
        naxis1 = hdr.get("NAXIS1")
        FOV = naxis1 * cdelt1

        # Find the peak in the image and cut along the given axis
        cut = maxpos(d)[axis]
        if axis == 0:
            cut = d[cut, :]
        elif axis == 1:
            cut = d[:, cut]

        # Convert data into brightness temp. to be independent of beam size
        if bright_temp:
            cut = Tb(
                data=cut,
                freq=hdr.get("RESTFRQ") * u.Hz.to(u.GHz),
                bmin=hdr.get("bmin") * u.deg.to(u.arcsec),
                bmaj=hdr.get("bmaj") * u.deg.to(u.arcsec),
            )

        # Offset from the center of the image
        offset = np.linspace(-FOV / 2, FOV / 2, naxis1)

        # Find the peak position along the cut
        cut_peak = np.argmax(cut)

        # Shift the angular offset to be centered on the peak
        offset = offset - offset[cut_peak]

        return offset, cut

    # Set the path prefix as a Path object, if provided
    prefix = Path(prefix)

    # Create a figure object
    fig = plt.figure()

    # Plot the cut from the real observation if required
    if add_obs:
        # Read data
        obs = Observation(name="IRAS16293B", lam=lam)

        # Drop empty axes, flip and rescale
        obs.drop_axis()
        obs.fliplr()
        if not bright_temp:
            obs.rescale(1e3)
        if scale_obs is not None and scale_obs > 0:
            obs.rescale(scale_obs)

        label = f"{obs.name} (x{scale_obs:.1f})" if scale_obs else obs.name
        obs.offset, obs.cut = angular_offset(obs.data, obs.header)
        plt.plot(
            obs.offset, obs.cut, label=label, color="black", ls="-.", *args, **kwargs
        )

    # Plot the cuts from the simulated observations for every inclination angle
    #for angle, color in zip([f"{i}deg" for i in angles], ["tab:blue", "tab:orange"]):
    for angle in [f"{i}deg" for i in angles]:
        # Read data
        filename = prefix / f"amax{amax}/{lam}/{angle}/data/{lam}_{angle}_a{amax}_alma.fits"
        data, hdr = fits.getdata(filename, header=True)

        # Drop empty axes. Flip and rescale
        data = np.fliplr(np.squeeze(data))
        if not bright_temp:
            data *= 1e3

        offset, cut = angular_offset(data, hdr)
        plt.plot(
            offset,
            cut,
            label=f"{angle} " + r"($T_{\rm dust}=T_{\rm gas}$)",
            *args,
            **kwargs,
        )

# <PATCH>
#        # Temporal lines to add the curves with Temp. from RT+EOS
#        prefix_ = Path(pwd.as_posix.replace("temp_comb", "temp_eos"))
#        filename = prefix_ / f"amax{amax}/{lam}/{angle}/data/{lam}_{angle}_a{amax}_alma.fits"
#        data, hdr = fits.getdata(filename, header=True)
#        # Drop empty axes. Flip and rescale
#        data = np.fliplr(np.squeeze(data))
#        if not bright_temp:
#            data *= 1e3
#        offset, cut = angular_offset(data, hdr)
#        plt.plot(
#            offset,
#            cut,
#            label=f"{angle} " + r"($T_{\rm dust}$ from $T_{\rm gas}$ only)",
#            c=color,
#            ls="--",
#        )
# </PATCH>

    # Customize the plot
    if 'lmd2.4' in str(pwd):
        hole_size = 0.014  # arcsec = 2 AU @ 141pc
        hole_size = (2 * u.au.to(u.pc) / 141) * u.rad.to(u.arcsec)
        plt.axvline(-hole_size, lw=1, ls='-', alpha=0.5, color='grey')
        plt.axvline(hole_size, lw=1, ls='-', alpha=0.5, color='grey')
        plt.text(0.003, 0, 'Central hole', rotation=90, horizontalalignment='center')

    plt.axvline(0, ls="--", lw=1, c="grey")
    plt.annotate('0.1" = 14AU', (0.75, 0.85), xycoords="axes fraction", fontsize=13)
    plt.annotate(r'$\lambda=$ %s'%lam, (0.75, 0.77), xycoords="axes fraction", fontsize=13)
    plt.legend(ncol=1, loc="upper left")
    plt.xlabel("Angular offset (arcseconds)")
    ylabel_ = r"$T_{\rm b}$ (K)" if bright_temp else r"mJy/beam"
    plt.ylabel(ylabel_)
    plt.xlim(-0.35, 0.35)

    plot_checkout(fig, show, savefig)

    return fig


def get_polaris_temp(binfile="grid_temp.dat"):
    """Read the binary output from a Polaris dust heating
    simulation and return the dust temperature field.
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

        return temp
