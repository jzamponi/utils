"""
	- Figure 1: Observations at 3mm, 1.3mm and spectral index.
	- Figure 2: Density xy & zx projections of Bo's disk and Ilee's disk
	- Figure 3: Dust opacities for several a_max with a silicate & graphite composition.
	- Figure 4: Dust temperature radial profiles for Tgas, Protostellar heating and combined.
	- Figure 5: Horizontal cuts for Bo's disk and Ilee's disk.
	- Figure 6: Simulated observations at 3mm, 1.3mm and spectral index.
    Figsize two-column: 18cm x 5.5cm = 3*2.36in x 2.17in
"""
import os
from astropy.io import ascii, fits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import utils

home = Path.home()

def plot_observations(show=True, savefig=None, figsize=(17.5,6)):
    """ Figure 1 """

    fig = plt.figure(figsize=figsize)

    f1 = utils.plot_map(
        home/f'phd/polaris/sourceB_1.3mm.fits',
        figsize=None,
        stretch='linear', 
        scalebar=20*u.au, 
        vmin=0, 
        vmax=287, 
        figure=fig,
        subplot=[0.12, 0.05, 0.25, 0.9], 
    )
    f2 = utils.plot_map(
        home/f'phd/polaris/sourceB_3mm.fits',
        figsize=None,
        stretch='linear', 
        scalebar=20*u.au, 
        vmin=0, 
        vmax=468, 
        figure=fig,
        subplot=[0.40, 0.05, 0.25, 0.9], 
    )
    f3 = utils.plot_map(
        filename = home/f'phd/polaris/sourceB_spectral_index.fits', 
        cmap='PuOr', 
        cblabel='Spectral index', 
        figsize=None, 
        stretch='linear',
        scalebar=20*u.au,
        bright_temp=False, 
        figure=fig,
        subplot=[0.68, 0.05, 0.25, 0.9], 
    )
    simulation_extent = 0.709198230621132
    img_radius = (simulation_extent/2)*u.arcsec.to(u.deg)
    #img_radius = (70*u.au.to(u.pc)/141)*u.rad.to(u.deg)

    for f,lam in zip([f1,f2], ['1.3mm', '3mm']):
        f.recenter(248.0942916667, -24.47550000000, radius=img_radius)
        f.scalebar.set_color('white')
        f.ticks.set_color('white')
        f.ticks.set_length(7)
        f.ticks.set_linewidth(2)
        f.add_label(0.79, 0.90, r'$\lambda =$ '+lam, relative=True, layer='lambda', color='white', size=25)
        f.add_beam(edgecolor='white', facecolor='none', linewidth=1)
        f.axis_labels.set_xtext('Right Ascension (J2000)')
    
    for f in [f2,f3]:
        f.tick_labels.hide_y()
        f.axis_labels.hide_y()
        f.axis_labels.set_xtext('Right Ascension (J2000)')

    f1.axis_labels.set_ytext('Declination (J2000)')
    f3.recenter(248.0942916667, -24.47550000000, radius=img_radius)
    f3.scalebar.set_color('black')
    f3.ticks.set_color('black')
    f3.ticks.set_length(7)
    f3.ticks.set_linewidth(2)
    f3.add_label(0.80, 0.95, r'$\alpha_{223-100 {\rm GHz}}$', relative=True, layer='lambda', color='black', size=25, weight='bold')
    f3.add_beam(edgecolor='black', facecolor='none', linewidth=1)
    f3.show_contour(colors="black", levels=[1.7, 2, 3])

    #plt.tight_layout()

    return utils.plot_checkout(fig, show, savefig=savefig, path=home/'phd/plots/paper1')


def plot_disk_model(model='bo', show=True, savefig=None, figsize=(8,6.8), use_aplpy=False):
    """ Figure 2 """
    from matplotlib.colors import LogNorm
    
    # Set the global path
    if 'bo' in model:
        filename = home/'phd/polaris/results/lmd2.4-1k-Slw/00260/dust_emission/temp_eos/sg/d141pc'
        density_ticks = [-19, -17, -15, -13, -11]
        temperature_ticks = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190]
    elif 'ilee' in model: 
        filename = home/'phd/ilees_disk/results/dust_emission/temp_eos/sg'
        density_ticks = [-14, -13, -12, -11, -10]
        temperature_ticks = [200, 400, 600, 800, 1000, 1200, 1400]

    # Read the data
    data, hdr = fits.getdata(filename/'amax10um/3mm/0deg/data/input_midplane.fits.gz', header=True)
    temp = {'faceon': data[2,0], 'edgeon': data[2,1]}
    dens = {'faceon': data[0,0], 'edgeon': data[0,1]}

    # Convert density to cgs
    dens = {p: d*(u.kg*u.m**-3).to(u.g*u.cm**-3) for (p,d) in dens.items()}

    # Zoom in the maps
    dens['faceon'] = np.log10(dens['faceon'][300:700,300:700]).T
    temp['faceon'] = temp['faceon'][300:700,300:700].T
    dens['edgeon'] = np.log10(dens['edgeon'][450:550,300:700])
    temp['edgeon'] = temp['edgeon'][450:550,300:700]

    # Plot using APLPy
    if use_aplpy:
        from aplpy import FITSFigure
        
        # Write the zoomed-in array into temporal files
        utils.write_fits('.dens_faceon.fits', dens['faceon'], utils.set_hdr_to_iras16293B(hdr))
        utils.write_fits('.temp_faceon.fits', temp['faceon'], utils.set_hdr_to_iras16293B(hdr))
        utils.write_fits('.dens_edgeon.fits', dens['edgeon'], utils.set_hdr_to_iras16293B(hdr))
        utils.write_fits('.temp_edgeon.fits', temp['edgeon'], utils.set_hdr_to_iras16293B(hdr))

        # Initialize the multipanel figure
        fig = plt.figure(figsize=figsize)
        df = FITSFigure('.dens_faceon.fits', figure=fig, subplot=[0.1, 0.191, 0.34, 0.65])
        tf = FITSFigure('.temp_faceon.fits', figure=fig, subplot=[0.5, 0.191, 0.34, 0.65])
        de = FITSFigure('.dens_edgeon.fits', figure=fig, subplot=[0.1, 0.19, 0.34, 0.10])
        te = FITSFigure('.temp_edgeon.fits', figure=fig, subplot=[0.5, 0.19, 0.34, 0.10])

        # Customize the plot
        [p.show_colorscale(cmap='heat') for p in [tf, te]]
        [p.show_colorscale(cmap='seaborn') for p in [df, de]]
        [p.add_colorbar(location='top') for p in [df, tf]]
        df.colorbar.set_axis_label_text(r'$n_{\rm gas}$ (g cm$^{-3}$)')
        tf.colorbar.set_axis_label_text(r'$T_{\rm gas}$ (K)')
        [p.axis_labels.hide() for p in [df, tf, de, te]]
        [p.tick_labels.hide() for p in [df, tf, de, te]]

        # Show the figure
        if show:
            fig.canvas.draw()

        # Remove temporal files
        for t in ['dens', 'temp']:
            for p in ['faceon', 'edgeon']:
                if os.path.isfile(f'.{t}_{p}.fits'): os.remove(f'.{t}_{p}.fits')
    
    # Plot using matplotlib GridSpec
    else:

        # Define the pixel size in physical units
        dx = np.round(hdr['CDELT2B'], 1)
        dx_unit = hdr.get('CUNIT2B')
        
        # Params for the subplot grid
        gridspec = {
            'height_ratios': [4,1], 
            'width_ratios': [1,1],
        }

        # Setup the figure
        fig, p = plt.subplots(nrows=2, ncols=2, figsize=figsize, gridspec_kw=gridspec)

        # Plot data
        df = p[0,0].imshow(dens['faceon'], interpolation='bicubic', cmap='cividis')
        tf = p[0,1].imshow(temp['faceon'], interpolation='bicubic', cmap='Spectral_r')
        de = p[1,0].imshow(dens['edgeon'], interpolation='bicubic', cmap='cividis')
        te = p[1,1].imshow(temp['edgeon'], interpolation='bicubic', cmap='Spectral_r')
        
        # Add colorbars
        df_cb = fig.colorbar(de, ax=p[1,0], pad=0.01, orientation='horizontal', ticks=density_ticks)
        tf_cb = fig.colorbar(te, ax=p[1,1], pad=0.01, orientation='horizontal', ticks=temperature_ticks)
        df_cb.set_label(r'$\log(n_{\rm gas})$ (g cm$^{-3}$)')
        tf_cb.set_label(r'$T_{\rm gas}$ (K)')

        # Set the axes scales and labels 
        tick_pars = {
            'axis': 'both',
            'colors': 'white', 
            'direction': 'in', 
            'which': 'both', 
            'bottom': True,
            'top': True, 
            'left': True, 
            'right': True,
        }
#        [p.tick_params(**tick_pars) for p in [p[0,0], p[1,0], p[0,1], p[1,1]]]

        p[0,0].set_xticks([])
        p[0,1].set_xticks([])
        p[1,0].set_xticks([])
        p[1,1].set_xticks([])

        p[0,0].set_yticks([0, 99, 199, 299, 399])
        p[1,0].set_yticks([0, 24, 49, 74, 99])
        p[0,1].set_yticks([0, 99, 199, 299, 399])
        p[1,1].set_yticks([0, 24, 49, 74, 99])
        p[0,0].set_yticklabels(['', f'{100*dx:.0f}', '0', f'{-100*dx:.0f}', ''])
        p[1,0].set_yticklabels(['', f'{25*dx:.0f}', '0', f'{-25*dx:.0f}', ''])
        p[0,1].set_yticklabels(['', f'{100*dx:.0f}', '0', f'{-100*dx:.0f}', ''])
        p[1,1].set_yticklabels(['', f'{25*dx:.0f}', '0', f'{-25*dx:.0f}', ''])
        p[0,1].yaxis.set_label_position('right')
        p[1,1].yaxis.set_label_position('right')
        p[0,1].yaxis.set_ticks_position('right')
        p[1,1].yaxis.set_ticks_position('right')
        p[0,0].set_ylabel(f'{dx_unit}')
        p[1,0].set_ylabel(f'{dx_unit}')
        p[0,1].set_ylabel(f'{dx_unit}')
        p[1,1].set_ylabel(f'{dx_unit}')


        # Adjust the subplots
        plt.subplots_adjust(wspace=0.01, hspace=0.0)

    return utils.plot_checkout(fig, show, savefig, path=home/'phd/plots/paper1/')


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


def plot_simulated_observations(model='ilee', incl='0deg', show=True, savefig=None, figsize=(17.5,6)):
    """ Figure 6 """
    from aplpy import FITSFigure

    if model == 'bo':
        prefix=Path('/home/jz/phd/polaris/results/lmd2.4-1k-Slw/00260/dust_emission/temp_eos/sg/d141pc/amax10um/')
    elif model == 'ilee':
        prefix=Path('/home/jz/phd/ilees_disk/results/dust_emission/temp_eos/sg/amax10um/')
    else:
        prefix=''

    fig = plt.figure(figsize=figsize) 

    f1 = utils.plot_map(
        prefix/f'1.3mm/{incl}/data/1.3mm_{incl}_a10um_alma.fits', 
        figsize=None,
        stretch='linear', 
        scalebar=20*u.au, 
        vmin=0, 
        vmax=None, 
        figure=fig,
        subplot=[0.12, 0.05, 0.25, 0.9], 
    )
    f2 = utils.plot_map(
        prefix/f'3mm/{incl}/data/3mm_{incl}_a10um_alma.fits', 
        figsize=None,
        stretch='linear', 
        scalebar=20*u.au, 
        vmin=0, 
        vmax=None, 
        figure=fig,
        subplot=[0.40, 0.05, 0.25, 0.9], 
    )
    f3 = utils.spectral_index(
        prefix/f'1.3mm/0deg/data/1.3mm_0deg_a10um_alma.fits', 
        prefix/f'3mm/0deg/data/3mm_0deg_a10um_alma_smoothed.fits', 
        figsize=None, 
        vmin=1.7, 
        vmax=3.5, 
        show=False,
        scalebar=20*u.au,
        figure=fig,
        subplot=[0.68, 0.05, 0.25, 0.9], 
    )
    simulation_extent = 0.709198230621132
    img_radius = (simulation_extent/2)*u.arcsec.to(u.deg)
    #img_radius = (70*u.au.to(u.pc)/141)*u.rad.to(u.deg)

    for f,lam in zip([f1,f2], ['1.3mm', '3mm']):
        f.recenter(248.0942916667, -24.47550000000, radius=img_radius)
        f.scalebar.set_color('white')
        f.ticks.set_color('white')
        f.ticks.set_length(7)
        f.ticks.set_linewidth(2)
        f.add_label(0.79, 0.90, r'$\lambda =$ '+lam, relative=True, layer='lambda', color='white', size=25)
        f.add_label(0.26, 0.90, r'incl. = 0 deg', relative=True, layer='inclination', color='white', size=25)
        f.add_label(0.26, 0.80, r'$a_{\rm max}=10\mu$m', relative=True, layer='amax', color='white', size=25)
        f.add_beam(edgecolor='white', facecolor='none', linewidth=1)
        f.axis_labels.set_xtext('Right Ascension (J2000)')
    
    for f in [f2,f3]:
        f.tick_labels.hide_y()
        f.axis_labels.hide_y()
        f.axis_labels.set_xtext('Right Ascension (J2000)')

    f1.axis_labels.set_ytext('Declination (J2000)')
    f3.recenter(248.0942916667, -24.47550000000, radius=img_radius)
    f3.add_beam(edgecolor='white', facecolor='none', linewidth=1)
    f3.scalebar.set_color('#af5606')
    f3.ticks.set_color('black')
    f3.ticks.set_length(7)
    f3.ticks.set_linewidth(2)

    return utils.plot_checkout(fig, show, savefig, path=home/f'phd/plots/paper1')