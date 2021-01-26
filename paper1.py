"""
	- Figure 1: Observations at 3mm, 1.3mm and spectral index.
	- Figure 2: Density xy & zx projections of Bo's disk and Ilee's disk
	- Figure 3: Dust opacities for several a_max with a silicate & graphite composition.
	- Figure 4: Dust temperature radial profiles for Tgas, Protostellar heating and combined.
	- Figure 5: Horizontal cuts for Bo's disk and Ilee's disk.
	- Figure 6: Synthetic spectral index.
    Figsize two-column: 18cm x 5.5cm = 3*2.36in x 2.17in
"""
from astropy.io import ascii, fits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import utils

home = Path.home()

def plot_observations(lam='1.3mm', show=True, savefig='', figsize=(6.90,6.00), *args, **kwargs):
    """ Figure 1 """

    fig = utils.plot_map(
        home/f'phd/polaris/sourceB_{lam}.fits',
        figsize=figsize,
        stretch='linear', 
        scalebar=30*u.au, 
        vmin=0, 
        vmax=287 if lam=='1.3mm' else 468, 
    )
    fig.scalebar.set_color('white')
    fig.ticks.set_color("white")
    fig.recenter(248.0942916667, -24.47550000000, radius=(70*u.au.to(u.pc)/141)*u.rad.to(u.deg))
    fig.add_label(0.78, 0.90, r'$\lambda = $'+lam, relative=True, layer='lambda', color='white', size=20)
    fig.add_beam(edgecolor='white', facecolor='none', linewidth=1)
    fig.axis_labels.set_xtext('Right Ascension (J2000)')
    fig.axis_labels.set_ytext('Declination (J2000)')
    plt.tight_layout()

    return utils.plot_checkout(fig, show=show, savefig=savefig)


def plot_obs_spectral_index(show=True, savefig='', figsize=(6.90, 6.00), *args, **kwargs):
    """ Figure 1 """

    filename = home/f'phd/polaris/sourceB_spectral_index.fits' 
    fig = utils.plot_map(
        filename, 
        cmap='PuOr', 
        cblabel='Spectral index', 
        figsize=figsize, 
        stretch='linear',
        scalebar=30*u.au,
        bright_temp=False, 
    )
    fig.scalebar.set_color('black')
    fig.ticks.set_color('black')
    fig.recenter(248.0942916667, -24.47550000000, radius=(70*u.au.to(u.pc)/141)*u.rad.to(u.deg))
    fig.add_label(0.78, 0.90, r'$\alpha_{223-100 {\rm GHz}}$', relative=True, layer='lambda', color='black', size=20, weight='bold')
    fig.add_beam(edgecolor='white', facecolor='none', linewidth=1)
    fig.axis_labels.set_xtext('Right Ascension (J2000)')
    fig.axis_labels.set_ytext('Declination (J2000)')
    fig.show_contour(colors="black", levels=[1.7, 2, 3])
    plt.tight_layout()

    return utils.plot_checkout(fig, show=show, savefig=savefig)


def plot_disk_bo(show=True, savefig='', figsize=(6.4,4.8)):
    """ Figure 2 """
    pass


def plot_disk_ilee(show=True, savefig='', figsize=(6.4,4.8)):
    """ Figure 2 """
    pass


def plot_opacities(show=True, savefig='', figsize=(5,3.5), composition='sg'):
    """ Figure 3 """

    def read_opac(amax, kappa):
        """Read polaris opacity file"""
        k = ascii.read(prefix/f'amax{amax}/data/dust_mixture_001.dat', data_start=10)
        kabs = k['col18'] * (u.m**2/u.kg).to(u.cm**2/u.g)
        ksca = k['col20'] * (u.m**2/u.kg).to(u.cm**2/u.g)
        return kabs if kappa == 'abs' else ksca

    # Set the global path
    prefix = home/f'phd/polaris/results/lmd2.4-1k-Slw/00260/dust_heating/{composition}/'

    # Read wavelength axis
    lam = ascii.read(prefix/f'amax1um/data/dust_mixture_001.dat', data_start=10)['col1']*u.m.to(u.micron)
    lam_l = ascii.read(prefix/f'amax1000um/data/dust_mixture_001.dat', data_start=10)['col1']*u.m.to(u.micron)

    # Create the figure object
    fig = plt.figure(figsize=figsize)
    abs_line = plt.plot([],[], ls='-',  color='black', label=r'Absorption')
    sca_line = plt.plot([],[], ls='--', color='black',  label=r'Scattering')

    abs_p10 = plt.loglog(
        lam, 
        read_opac('1um','abs'), 
        ls='-', 
        color='tab:purple', 
        label=r'$a_{\rm max}=1\mu$m'
    )
    sca_p10 = plt.loglog(
        lam, 
        read_opac('1um','sca'), 
        ls='--', color='tab:purple'
    )

    abs_p10 = plt.loglog(
        lam, 
        read_opac('10um','abs'), 
        ls='-', 
        color='tab:red', 
        label=r'$a_{\rm max}=10\mu$m'
    )
    sca_p10 = plt.loglog(
        lam, 
        read_opac('10um','sca'), 
        ls='--', color='tab:red'
    )

    abs_p100 = plt.loglog(
        lam, 
        read_opac('100um','abs'), 
        ls='-', 
        color='tab:green', 
        label=r'$a_{\rm max}=100\mu$m'
    )
    sca_p100 = plt.loglog(
        lam, 
        read_opac('100um','sca'), 
        ls='--', color='tab:green'
    )

    abs_p1000 = plt.loglog(
        lam_l, 
        read_opac('1000um','abs'), 
        ls='-', 
        color='tab:blue', 
        label=r'$a_{\rm max}=1000\mu$m'
    )
    sca_p1000 = plt.loglog(
        lam_l, 
        read_opac('1000um','sca'), 
        ls='--', color='tab:blue'
    )

    plt.axvline(1.3e3, ls=':', color='grey')
    plt.axvline(3e3, ls=':', color='grey')
    plt.text(0.9e3, 1e2, '1.3mm', rotation=90, size=13, color='grey') 
    plt.text(2.2e3, 1e2, '3mm', rotation=90, size=13, color='grey') 

    plt.tick_params(which='both', direction='in', left=True, right=True, bottom=True, top=True)
    plt.minorticks_on()
    plt.ylim(1e-3,1e5)
    plt.xlim(0.1,4e3)

    plt.legend()
    plt.xlabel('Wavelength (microns)')
    plt.ylabel(r'$\kappa_{\nu}$ (cm$^2$g$^{-1}$)')

    plt.tight_layout()

    return utils.plot_checkout(fig, show, savefig, path=home/f'phd/plots/paper1')



def dust_temperature_bo(show=True, savefig='', figsize=(6.4,4.8), nthreads=1):
    """ Figure 4 """

    prefix = home/'phd/polaris/results/lmd2.4-1k-Slw/00260/dust_heating/sg/amax10um'

    r_rt, rt = utils.radial_profile(
        fits.getdata(f'{prefix}/data/output_midplane.fits.gz')[0,0], 
        return_radii=True, 
        dr=fits.getheader(f'{prefix}/data/output_midplane.fits.gz')['cdelt1b']*u.au, 
        nthreads=nthreads, 
    )
    r_eos, eos = utils.radial_profile(
        fits.getdata(f'{prefix}/temp_offset/data/input_midplane.fits.gz')[2,0], 
        return_radii=True, 
        dr=fits.getheader(f'{prefix}/temp_offset/data/input_midplane.fits.gz')['cdelt1b']*u.au, 
        nthreads=nthreads, 
    )
    r_eos_rt, eos_rt = utils.radial_profile(
        fits.getdata(f'{prefix}/temp_offset/data/output_midplane.fits.gz')[0,0], 
        return_radii=True, 
        dr=fits.getheader(f'{prefix}/temp_offset/data/output_midplane.fits.gz')['cdelt1b']*u.au, 
        nthreads=nthreads, 
    )

    fig = plt.figure(figsize=figsize)
    plt.semilogx(r_rt, rt, ls=':', c='black', label='Protostar heating')
    plt.semilogx(r_eos, eos, ls='--', c='black', label=r'$T_{\rm gas}$')
    plt.semilogx(r_eos_rt, eos_rt, ls='-', c='black', label=r'$T_{\rm gas}$ and Protostar heating')
    plt.axvline(1.9, ls='--', c='tab:red', lw=1)
    plt.text(1.05, 220, 'Central', c='tab:red')
    plt.text(1.20, 210, 'hole', c='tab:red')
    plt.xlim(1e0,1e2)
    plt.ylim(0, 370)
    plt.xlabel('Radius (AU)')
    plt.ylabel('Dust temperature (K)')
    plt.legend()
    plt.tight_layout()

    return utils.plot_checkout(fig, show, savefig, path=home/f'phd/plots/paper1')


def dust_temperature_ilee(show=True, savefig='', figsize=(6.4,4.8), nthreads=1):
    """ Figure 4 """

    prefix = home/'phd/ilees_disk/results/dust_heating/sg/amax10um'

    r_rt, rt = utils.radial_profile(
        fits.getdata(f'{prefix}/data/output_midplane.fits.gz')[0,0], 
        return_radii=True, 
        dr=fits.getheader(f'{prefix}/data/output_midplane.fits.gz')['cdelt1b']*u.au, 
        nthreads=nthreads, 
    )
    r_eos, eos = utils.radial_profile(
        fits.getdata(f'{prefix}/temp_offset/data/input_midplane.fits.gz')[2,0], 
        return_radii=True, 
        dr=fits.getheader(f'{prefix}/temp_offset/data/input_midplane.fits.gz')['cdelt1b']*u.au, 
        nthreads=nthreads, 
    )
    r_eos_rt, eos_rt = utils.radial_profile(
        fits.getdata(f'{prefix}/temp_offset/data/output_midplane.fits.gz')[0,0], 
        return_radii=True, 
        dr=fits.getheader(f'{prefix}/temp_offset/data/output_midplane.fits.gz')['cdelt1b']*u.au, 
        nthreads=nthreads, 
    )

    fig = plt.figure(figsize=figsize)
    plt.semilogx(r_rt, rt, ls=':', c='black', label='Protostar heating')
    plt.semilogx(r_eos, eos, ls='--', c='black', label=r'$T_{\rm gas}$')
    plt.semilogx(r_eos_rt, eos_rt, ls='-', c='black', label=r'$T_{\rm gas}$ and Protostar heating')
    plt.axhline(1200, ls='--', c='tab:red', lw=1)
    plt.text(0.2, 1220, 'Silicate sublimation', c='tab:red')
    plt.xlim(0,1e2)
    plt.ylim(0, 1345)
    plt.xlabel('Radius (AU)')
    plt.ylabel('Dust temperature (K)')
    plt.legend()
    plt.tight_layout()

    return utils.plot_checkout(fig, show, savefig, path=home/f'phd/plots/paper1')


def plot_horizontal_cuts(model, lam='3mm', show=True, savefig='', figsize=(6.4,4.8)):
    """ Figure 5 
        Plot cuts for both models at 1.3 and 3mm.    

        TO DO: run the RT sim. for Bo's model with inc: 10, 20, 30 deg for 1.3 and 3mm
    """
    
    if model == 'bo':
        prefix='/home/jz/phd/polaris/results/lmd2.4-1k-Slw/00260/dust_emission/temp_comb/sg/d141pc/'
        prefix='/home/jz/phd/polaris/results/lmd2.4-1k-Slw/00260/dust_emission/temp_eos/sg/d141pc/'
    elif model == 'ilee':
        prefix='/home/jz/phd/ilees_disk/results/dust_emission/temp_eos/sg/'
    else:
        prefix=''

    fig = utils.horizontal_cuts(
        angles=[0,40], 
        bright_temp=True, 
        add_obs=True, 
        axis=0, 
        prefix=prefix, 
        amax='10um', 
        lam=lam, 
        show=False, 
    )

    return utils.plot_checkout(fig, show, savefig, path=home/f'phd/plots/paper1')


def plot_spectral_index(model, show=True, savefig='', figsize=(6.4,4.8)):
    """ Figure 6 """

    if model == 'bo':
        prefix=Path('/home/jz/phd/polaris/results/lmd2.4-1k-Slw/00260/dust_emission/temp_eos/sg/d141pc/amax10um/')
    elif model == 'ilee':
        prefix=Path('/home/jz/phd/ilees_disk/results/dust_emission/temp_eos/sg/amax10um/')
    else:
        prefix=''

    fig = utils.spectral_index(
        prefix/f'1.3mm/0deg/data/1.3mm_0deg_a10um_alma.fits', 
        prefix/f'3mm/0deg/data/3mm_0deg_a10um_alma_smoothed.fits', 
        vmin=1.7, 
        vmax=3.5, 
        figsize=figsize, 
        show=False,
    )

    return utils.plot_checkout(fig, show, savefig, path=home/f'phd/plots/paper1')

