"""
	- Figure 1: Observations at 3mm, 1.3mm and spectral index.
	- Figure 2: Density xy & zx projections of the MHD disk and RHD disk
	- Figure 3: Dust opacities for several a_max with a silicate & graphite composition.
	- Figure 4: Dust temperature radial profiles for Tgas, Protostellar heating and combined.
	- Figure 5: Horizontal cuts for MHD disk.
	- Figure 6: Horizontal cuts for RHD disk.
	- Figure 7: Simulated observations at 3mm, 1.3mm and spectral index.
    - Figure 8: Dust temperature at the tau = 1 surface
    Figsize two-column: 18cm x 5.5cm = 3*2.36in x 2.17in
"""
import os
from astropy.io import ascii, fits
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import utils

home = Path.home()

def plot_observations(show=True, savefig=None, figsize=(17.5,6)):
    """ Figure 1 """

    fig = plt.figure(figsize=figsize)

    f1 = utils.plot_map(
        home/f'phd/observations/iras16293/band6/sourceB_1.3mm.fits',
        figsize=None,
        stretch='linear', 
        scalebar=20*u.au, 
        vmin=0, 
        vmax=287, 
        contours=False, 
        figure=fig,
        subplot=[0.12, 0.05, 0.25, 0.9], 
    )
    f2 = utils.plot_map(
        home/f'phd/observations/iras16293/band3/sourceB_3mm.fits',
        figsize=None,
        stretch='linear', 
        scalebar=20*u.au, 
        vmin=0, 
        vmax=468, 
        contours=False, 
        figure=fig,
        subplot=[0.40, 0.05, 0.25, 0.9], 
    )
    f3 = utils.plot_map(
        filename = home/f'phd/observations/iras16293/sourceB_spectral_index_band3-6.fits', 
        cmap='PuOr', 
        vmin=1.7,
        vmax=3.7,
        cblabel='Spectral index', 
        figsize=None, 
        stretch='linear',
        scalebar=20*u.au,
        bright_temp=False, 
        figure=fig,
        subplot=[0.68, 0.05, 0.25, 0.9], 
    )

    # Recenter the figures
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

    
    # Add a contour for the water snow line
    #f2.show_contour(colors='tab:blue', levels=[170], linewidths=3)

    # Add each FITS-sub-figure to the main figure 
    fig.f1 = f1
    fig.f2 = f2
    fig.f3 = f3

    return utils.plot_checkout(fig, show, savefig=savefig, path=home/'phd/plots/paper1')


def plot_disk_model(model='rhd', show=True, savefig=None, figsize=(8,6.8), use_aplpy=False):
    """ Figure 2 """
    from matplotlib.colors import LogNorm
    
    # Set the global path
    if model == 'mhd':
        filename = home/'phd/polaris/results/lmd2.4-1k-Slw/00260/dust_emission/temp_eos/sg/d141pc'
    elif model == 'rhd': 
        filename = home/'phd/rhd_disk/results/snap541/dust_emission/temp_eos/sg'

    # Read the data
    data, hdr = fits.getdata(filename/'amax10um/3mm/0deg/dust_scat/data/input_midplane.fits.gz', header=True)
    temp = {'faceon': data[2,0], 'edgeon': data[2,1]}
    dens = {'faceon': data[0,0], 'edgeon': data[0,1]}

    # Convert density to cgs
    dens = {p: d*(u.kg*u.m**-3).to(u.g*u.cm**-3) for (p,d) in dens.items()}

    # Zoom in the maps
    dens['faceon'] = dens['faceon'][300:700,300:700].T
    temp['faceon'] = temp['faceon'][300:700,300:700].T
    dens['edgeon'] = dens['edgeon'][450:550,300:700]
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
        df.colorbar.set_axis_label_text(r'$\rho_{\rm gas}$ (g cm$^{-3}$)')
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

        # Set the plot scales
        min_d = 1e-15
        min_t = 8 if model == 'mhd' else 80
        max_d = 2e-10
        max_t = None if model == 'mhd' else None
        density_ticks = [1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10]
        temperature_ticks = [1e1, 1e2] if model == 'mhd' else [1e2, 1e3]

        plt.rcParams['image.interpolation'] = 'bicubic'
        cmap_d = 'BuPu'
        cmap_t = 'inferno'
        df = p[0,0].imshow(dens['faceon'], norm=LogNorm(vmin=min_d, vmax=max_d), cmap=cmap_d)
        tf = p[0,1].imshow(temp['faceon'], norm=LogNorm(vmin=min_t, vmax=max_t), cmap=cmap_t)
        de = p[1,0].imshow(dens['edgeon'], norm=LogNorm(vmin=min_d, vmax=max_d), cmap=cmap_d)
        te = p[1,1].imshow(temp['edgeon'], norm=LogNorm(vmin=min_t, vmax=max_t), cmap=cmap_t)
        
        # Add colorbars
        df_cb = fig.colorbar(de, ax=p[1,0], pad=0.01, orientation='horizontal', ticks=density_ticks)
        tf_cb = fig.colorbar(te, ax=p[1,1], pad=0.01, orientation='horizontal', ticks=temperature_ticks)
        df_cb.set_label(r'$\rho\,_{\rm gas}$ (g cm$^{-3}$)')
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


def plot_opacities(show=True, savefig='', figsize=(5,3.5), composition='sg', add_albedo=False):
    """ Figure 3 """

    def read_opac(amax, kappa):
        """Read polaris opacity file"""
        k = ascii.read(prefix/f'amax{amax}/data/dust_mixture_001.dat', data_start=10)
        lam = k['col1'] * u.m.to(u.micron)
        kabs = k['col18'] * (u.m**2/u.kg).to(u.cm**2/u.g)
        ksca = k['col20'] * (u.m**2/u.kg).to(u.cm**2/u.g)
        return (lam, kabs if kappa == 'abs' else ksca)

    def albedo(amax):
        kabs = read_opac(amax, 'abs')
        ksca = read_opac(amax, 'sca')

        return ksca / (kabs + ksca)

    # Set the global path
    prefix = home/f'phd/polaris/results/lmd2.4-1k-Slw/00260/dust_heating/temp_rt/{composition}/'

    # Read wavelength axis
    lam = ascii.read(prefix/f'amax1um/data/dust_mixture_001.dat', data_start=10)['col1']*u.m.to(u.micron)
    lam_l = ascii.read(prefix/f'amax1000um/data/dust_mixture_001.dat', data_start=10)['col1']*u.m.to(u.micron)

    # Create the figure object
    fig, p = plt.subplots(figsize=figsize, nrows=1, ncols=1, sharex=True)

    # Add two empty curves to populate the legend
    abs_line = p.plot([],[], ls='-',  color='black', label=r'Absorption')
    sca_line = p.plot([],[], ls='--', color='black',  label=r'Scattering')

    # Plot absorption and scattering opacitites
    abs_p10 = p.loglog(
        *read_opac('1um','abs'), 
        ls='-', 
        color='tab:purple', 
        label=r'$a_{\rm max}=1\mu$m'
    )
    sca_p10 = p.loglog(
        *read_opac('1um','sca'), 
        ls='--', 
        color='tab:purple'
    )

    abs_p10 = p.loglog(
        *read_opac('10um','abs'), 
        ls='-', 
        color='tab:red', 
        label=r'$a_{\rm max}=10\mu$m'
    )
    sca_p10 = p.loglog(
        *read_opac('10um','sca'), 
        ls='--', 
        color='tab:red'
    )

    abs_p100 = p.loglog(
        *read_opac('100um','abs'), 
        ls='-', 
        color='tab:green', 
        label=r'$a_{\rm max}=100\mu$m'
    )
    sca_p100 = p.loglog(
        *read_opac('100um','sca'), 
        ls='--', 
        color='tab:green'
    )

    abs_p1000 = p.loglog(
        *read_opac('1000um','abs'), 
        ls='-', 
        color='tab:blue', 
        label=r'$a_{\rm max}=1000\mu$m'
    )
    sca_p1000 = p.loglog(
        *read_opac('1000um','sca'), 
        ls='--', 
        color='tab:blue'
    )

    # Upper panel: opacities
    p.axvline(1.3e3, ls=':', color='grey')
    p.axvline(3e3, ls=':', color='grey')
    p.text(0.9e3, 1e2, '1.3 mm', rotation=90, size=13, color='grey') 
    p.text(2.05e3, 1e2, '3 mm', rotation=90, size=13, color='grey') 
    p.legend(ncol=1)
    p.set_ylim(1e-3,1e5)
    p.set_xlim(1,4e3)
    p.set_ylabel(r'$\kappa_{\nu}$ (cm$^2$g$^{-1}$)')
    p.set_xlabel(r'Wavelength (microns)')
    plt.tight_layout()

    return utils.plot_checkout(fig, show, savefig, path=home/f'phd/plots/paper1')



def plot_dust_temperature(show=True, savefig=None, figsize=(6, 7), smooth=False, nthreads=1):
    """ Figure 4 

        Plot radially averaged midplane temperatures of both models for both 
        radiative and gas temperatures.
        If smooth, it will smooth out the curves by interpolating with a spline3 or poly.
    """

    prefixes = {
        'mhd' : home/'phd/polaris/results/lmd2.4-1k-Slw/00260/dust_heating/temp_rt/sg/amax10um', 
        'rhd' : home/'phd/rhd_disk/results/snap541/dust_heating/temp_rt/sg/amax10um',
    }

    fig, p_ = plt.subplots(nrows=2, ncols=1, figsize=figsize, sharex=True)

    for model, p in zip(['mhd', 'rhd'], p_):
        prefix = prefixes[model]

        for curve in ['rt', 'eos', 'eos_rt']: 
            tempfile = home/f'phd/plots/paper1/.profile_{model}_{curve}.fits'
            # Read the results from file 
            if os.path.isfile(tempfile):
                data = fits.getdata(tempfile)
                if curve == 'rt':
                    r_rt, rt = data[0], data[1]
                elif curve == 'eos':
                    r_eos, eos = data[0], data[1]
                elif curve == 'eos_rt':
                    r_eos_rt, eos_rt = data[0], data[1]

            # If not cached, calculate the radial averages and print  to tape
            else:
                utils.print_(f'Profile "{curve}" for model {model} is not saved. Recalculating ...', True)
                if curve == 'rt':
                    r_rt, rt = utils.radial_profile(
                        f'{prefix}/data/output_midplane.fits.gz',  
                        return_radii=True, 
                        nthreads=nthreads, 
                    )
                    utils.write_fits(tempfile, np.array([r_rt, rt]), overwrite=True)
                elif curve == 'eos':
                    r_eos, eos = utils.radial_profile(
                        f'{prefix}/data/input_midplane.fits.gz', 
                        slices=[2,0], 
                        return_radii=True, 
                        nthreads=nthreads, 
                    )
                    utils.write_fits(tempfile, np.array([r_eos, eos]), overwrite=True)
                elif curve == 'eos_rt':
                    r_eos_rt, eos_rt = utils.radial_profile(
                        f'{prefix}/data/output_midplane.fits.gz', 
                        return_radii=True, 
                        nthreads=nthreads, 
                    )
                    utils.write_fits(tempfile, np.array([r_eos_rt, eos_rt]), overwrite=True)

        # TO DO: Smooth the curves with a polinomial or spline fit
        if smooth:
            pol_order = 5
            fit = lambda x, y: np.poly1d(np.polyfit(x, y, pol_order))(x)
            rt = fit(r_rt, rt)
            eos = fit(r_eos, eos)
            eos_rt = fit(r_eos_rt, eos_rt)

        # Plot the curves
        p.plot(r_rt, rt, ls=':', c='black', label=r'$T_{\rm dust} = T_{\rm rad}$ ')
        p.plot(r_eos, eos, ls='--', c='black', label=r'$T_{\rm dust} = T_{\rm gas}$')
        p.plot(r_eos_rt, eos_rt, ls='-', c='black', 
            label=r'$T_{\rm dust} = T_{\rm gas}$ and $T_{\rm rad}$'
        )

        # Customize both panels
        plt.rcParams['font.size'] = 12
        if model == 'mhd':
            p.axvline(1.7, ls='--', c='tab:red', lw=1)
            p.annotate(
                'MHD model', 
                (0.10,0.82), 
                xycoords='axes fraction', 
                weight='bold'
            ) 
            p.set_ylim(-10, 370)
            p.set_yticks(np.arange(0, 400, 50))
            p.legend(loc='upper right')

            # Annotate the Q<1.7 region (8-25au = 0.05-0.2as)
            p.annotate(r'$Q<1.7$', (15, 215), xycoords="data", c='black', fontsize=11)
            p.annotate('', xy=(8, 200), xytext=(25, 200), 
                arrowprops=dict(arrowstyle='|-|', lw=0.5, mutation_scale=1.5))

        elif model == 'rhd':
            p.axhline(1200, ls='--', c='tab:red', lw=1)
            p.text(25, 1100, 'Silicate sublimation', c='tab:red')
            p.annotate(
                'RHD model', 
                (0.10,0.82), 
                xycoords='axes fraction', 
                weight='bold'
            ) 
            p.set_ylim(-50, 1250)
            p.set_yticks(np.arange(0, 1400, 200))
            p.set_xlabel('Radius (AU)')
    
            # Annotate the Q<1.7 region (7-30au = 0.05-0.2as)
            p.annotate(r'$Q<1.7$', (17, 650), xycoords="data", c='black', fontsize=11)
            p.annotate('', xy=(7, 600), xytext=(30, 600), 
                arrowprops=dict(arrowstyle='|-|', lw=0.5, mutation_scale=1.5))

        p.set_ylabel(r'$T_{\rm dust}$ (K)')
        p.set_xlim(0.0, 40)

    plt.subplots_adjust(hspace=0)

    return utils.plot_checkout(fig, show, savefig, path=home/f'phd/plots/paper1')


def plot_horizontal_cuts(model, lam='3mm', show=True, savefig='', figsize=(6.4,4.8)):
    """ Figure 5 
        Plot cuts for both models at 1.3 and 3mm.    

        TO DO: 
            - run the RT sim. for Bo's model with inc: 10, 20, 30 deg for 1.3 and 3mm
            - Add a shaded region with the RMS to the obs.
        
    """
    
    if model == 'mhd':
        prefix=home/'phd/polaris/results/lmd2.4-1k-Slw/00260/dust_emission/temp_comb/sg/d141pc/'
        angles = [0, 10, 20, 30, 40]

    elif model == 'rhd':
        prefix=home/'phd/rhd_disk/results/snap541/dust_emission/temp_eos/sg/'
        angles = [0, 10, 20, 30, 40]

    else:
        prefix=''
        angles = [0, 10, 20, 30, 40]

    fig = utils.horizontal_cut(
        angles=angles, 
        bright_temp=True, 
        add_obs=True, 
        axis=0, 
        prefix=prefix, 
        amax='10um', 
        lam=lam, 
        show=False, 
    )

    # Customize the plot for each model
    if 'lmd2.4' in str(prefix):
        # Label the panel
        plt.annotate('MHD model', (0.70, 0.77), xycoords="axes fraction", fontsize=16)

        # Annotate the central hole
        plt.axvline(-0.014, ls="-", lw=1, alpha=0.1, c="black", zorder=2)
        plt.axvline(0.014, ls="-", lw=1, alpha=0.1, c="black", zorder=2)
        plt.fill_betweenx(range(800), x1=-0.014, x2=0.014, color='grey', 
            alpha=0.2, zorder=4, hatch='/')
        plt.ylim(0, 300 if lam=='1.3mm' else 480)

        # Annotate the Q<1.7 region (8-25au = 0.05-0.2as)
        plt.annotate(r'$Q<1.7$', (0.54, 0.05), xycoords="axes fraction", c='blue', fontsize=13)
        plt.annotate(r'$Q<1.7$', (0.33, 0.05), xycoords="axes fraction", c='blue', fontsize=13)
        plt.axvspan(-0.18, -0.06, 0, 0.03, color='blue', alpha=0.13, zorder=3)
        plt.axvspan(0.06, 0.18, 0, 0.03, color='blue', alpha=0.13, zorder=3)

        plt.xticks([])
        plt.xlabel("")

    elif 'rhd_disk' in str(prefix):
        # Label the panel
        plt.annotate('RHD model', (0.70, 0.77), xycoords="axes fraction", fontsize=16)

        # Annotate the region of aritifical viscosity
        plt.axvline(-0.07, ls="-", lw=1, alpha=0.1, c="black", zorder=2)
        plt.axvline(0.07, ls="-", lw=1, alpha=0.1, c="black", zorder=2)
        plt.fill_betweenx(range(800), x1=-0.07, x2=0.07, color='grey', alpha=0.2, 
            zorder=4, hatch='/')
        plt.ylim(0, 520 if lam=='1.3mm' else 730)

        # Annotate the Q<1.7 region (7-30au = 0.05-0.2as)
        plt.annotate(r'$Q<1.7$', (0.62, 0.05), xycoords="axes fraction", c='blue', fontsize=13)
        plt.annotate(r'$Q<1.7$', (0.20, 0.05), xycoords="axes fraction", c='blue', fontsize=13)
        plt.axvspan(-0.28, -0.05, 0, 0.03, color='blue', alpha=0.13, zorder=3)
        plt.axvspan(0.05, 0.28, 0, 0.03, color='blue', alpha=0.13, zorder=3)

        plt.xlabel("Angular offset (arcseconds)")

    plt.annotate(r'$\lambda=$ %s'%lam, (0.70, 0.85), xycoords="axes fraction", fontsize=20)
    plt.annotate('0.1" = 14AU', (0.73, 0.45), xycoords="axes fraction", fontsize=13)
    plt.legend(ncol=1, loc="upper left")
    plt.xlim(-0.33, 0.33)

    return utils.plot_checkout(fig, show, savefig, path=home/f'phd/plots/paper1')


def plot_simulated_observations(model='rhd', incl='0deg', amax='10um', show=True, savefig=None, figsize=(17.5,6)):
    """ Figure 6 """
    from aplpy import FITSFigure

    if model == 'mhd':
        prefix=Path(home/f'phd/polaris/results/lmd2.4-1k-Slw/00260/dust_emission/temp_comb/sg/d141pc/amax{amax}/')
    elif model == 'rhd':
        prefix=Path(home/f'phd/rhd_disk/results/snap541/dust_emission/temp_eos/sg/amax{amax}/')
    else:
        prefix=''

    fig = plt.figure(figsize=figsize) 

    f1 = utils.plot_map(
        prefix/f'1.3mm/{incl}/dust_scat/data/alma_I.fits', 
        figsize=None,
        stretch='linear', 
        scalebar=20*u.au, 
        vmin=0, 
        vmax=None, 
        figure=fig,
        checkout=False,
        subplot=[0.12, 0.05, 0.25, 0.9], 
    )
    f2 = utils.plot_map(
        prefix/f'3mm/{incl}/dust_scat/data/alma_I.fits', 
        figsize=None,
        stretch='linear', 
        scalebar=20*u.au, 
        vmin=0, 
        vmax=None, 
        figure=fig,
        checkout=False,
        subplot=[0.40, 0.05, 0.25, 0.9], 
    )
    f3 = utils.spectral_index(
        prefix/f'1.3mm/{incl}/dust_scat/data/alma_I.fits', 
        prefix/f'3mm/{incl}/dust_scat/data/alma_I_robust0_smoothed1.3mm.fits', 
        figsize=None, 
        vmin=1.7, 
        vmax=3.5, 
        mask=4e-3 if model == 'rhd' else 1e-4,
        show=False,
        scalebar=20*u.au,
        figure=fig,
        subplot=[0.68, 0.05, 0.25, 0.9], 
        return_fig=True, 
    )
    simulation_extent = 0.709198230621132
    img_radius = (simulation_extent/2)*u.arcsec.to(u.deg)

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
    f3.add_beam(edgecolor='black', facecolor='none', linewidth=1)
    f3.scalebar.set_color('#af5606')
    f3.ticks.set_color('black')
    f3.ticks.set_length(7)
    f3.ticks.set_linewidth(2)

    # Add each FITS-sub-figure to the main figure 
    fig.f1 = f1
    fig.f2 = f2
    fig.f3 = f3

    return utils.plot_checkout(fig, show, savefig, path=home/f'phd/plots/paper1')


def plot_tau1_surface(lam='1.3mm', tau=1, bin_factor=1, show=True, savefig=None, figsize=(10,6)):
    """ Figure 7:
        Plot the 2D temperature at the tau=1 surface using APLPy
    """

    # Compute the 2D Tdust distribution at 1.3 and 3 mm
    Td_tau1_1mm, Td_tau1_3mm = utils.tau_surface(
        tau=tau, 
        prefix=home/'phd/rhd_disk/results/snap541/grid3d/10au_height/nobin/data', 
        bin_factor=bin_factor, 
        amax='10um', 
        plot2D=True,
        plot3D=False,
        convolve_map=True,
        verbose=True
    )

    # Select which tau surface to plot
    temp2D = Td_tau1_1mm if lam == '1.3mm' else Td_tau1_3mm

    # Generate the figure with APLPy
    fig = utils.plot_map(
        np.flipud(temp2D), 
        header=utils.Observation(lam).header, 
        figsize=(8, 6),
        bright_temp=False, 
        cblabel=r'Dust temperature (K)', 
        scalebar=20*u.au if lam=='1.3mm' else 130*u.au, 
        vmin=0, 
        vmax=480 if lam == '1.3mm' else 742,
        cmap='inferno', 
    )   

    # Customize the plots        
    simulation_extent = 0.709198230621132
    img_radius = (simulation_extent/2)*u.arcsec.to(u.deg)
    fig.ticks.set_color('white')
    fig.ticks.set_length(7)
    fig.ticks.set_linewidth(2)
    fig.add_label(0.79, 0.90, r'$\lambda =$ '+lam, relative=True, color='white', size=25)
    fig.add_label(0.19, 0.90, r'Surface at', relative=True, color='white', size=25)
    fig.add_label(0.12, 0.83, r'$\tau=1$', relative=True, color='white', size=25)
    fig.scalebar.set_color('white')
    fig.scalebar.set_label('20 AU')
    fig.axis_labels.set_xtext('')
    fig.tick_labels.hide_x()
    fig.tick_labels.hide_y()
    fig.axis_labels.hide_x()
    fig.axis_labels.hide_y()
    fig.frame.set_linewidth(2)

    plt.tight_layout()
 
    utils.plot_checkout(fig, show, savefig, path=home/f'phd/plots/paper1')


def plot_toomre_parameter(show=True, savefig=None, figsize=(4.0, 3.5)):
    """ Plot the radially averaged Toomre parameter for both disk models. """
    
    from astropy.nddata import block_reduce
    
    # Read the tabulated Q values for the RHD model 
    q_hd = ascii.read(home/'phd/plots/paper1/toomre_q_rhd_model.txt')

    # Read the tabulated Q values for the MHD model 
    q_mhd = ascii.read(home/'phd/plots/paper1/toomre_q_mhd_model.txt')

    # Generate the figure
    fig = plt.figure(figsize=figsize)

    plt.plot(*q_hd.values(), color='black', ls='-', label='RHD model')
    plt.plot(*q_mhd.values(), color='black', ls='--', label='MHD model')
    plt.axhline(1.7, color='black', ls=':', alpha=0.5)
    plt.text(33, 1.57, r'$Q=1.7$', size=11)
    plt.annotate('MHD\nmodel', xy=(0.50, 0.80), xycoords='axes fraction')
    plt.annotate('RHD\nmodel', xy=(0.82, 0.70), xycoords='axes fraction')

    #plt.legend()
    plt.ylabel('Toomre Q')
    plt.xlabel('Radius (AU)')

    plt.ylim(1, 3)
    plt.xlim(5, 40)
    plt.yticks([1, 2, 3])
    plt.xticks(range(5, 45, 5))
    plt.tight_layout()

    return utils.plot_checkout(fig, show, savefig, path=home/f'phd/plots/paper1')


def plot_disk_mass(show=True, savefig=None, figsize=(4.5, 3.5)):
    """ Plot the underestimation of the observational estimate of disk
        mass from the synthetic obseravtoins of the RHD model.
    """
    
    # Define the temperature range
    temp = np.linspace(30, 100, 50)

    # Define the real mass [M_sun] of the RHD disk
    model = 0.3

    # Define the fluxes at 1.3 and 3mm
    S_1 = 1.60*u.Jy
    S_3 = 0.29*u.Jy
    
    # Compute the mass estimates using both fluxes
    mass_1 = utils.dust_mass(temp*u.K, S_1, lam='1.3mm')
    mass_3 = utils.dust_mass(temp*u.K, S_3, lam='3mm')

    # Plot the mass ratios
    fig = plt.figure(figsize=figsize)
    plt.plot(temp, mass_1/model, color='black', ls='-', label=r'$S_{\rm 1.3 mm}$')
    plt.plot(temp, mass_3/model, color='black', ls='--', label=r'$S_{\rm 3 mm}$')
    plt.axhline(1, ls=':', color='grey', alpha=0.7, lw=1.2)
    plt.annotate(r'$S_{\rm 1.3\,mm}$', xy=(0.8, 0.1), xycoords='axes fraction', size=15)
    plt.annotate(r'$S_{\rm 3\,mm}$', xy=(0.8, 0.3), xycoords='axes fraction', size=15)
    plt.annotate(r'$M_{\rm model} = 0.3\,$M$_{\odot}$', xy=(70, 1.05), xycoords='data', size=13)

    plt.ylabel(r'$M_{\rm op.\,thin} / M_{\rm model}$')
    plt.xlabel(r'Dust temperature (K)')
    plt.xlim(temp.min(), temp.max())
    plt.xticks(np.arange(30, 110, 10))
    plt.yticks(np.arange(0.25, 2, 0.25))
    plt.tight_layout()

    return utils.plot_checkout(fig, show, savefig, path=home/f'phd/plots/paper1')


def plot_spectral_index_map_unconvolved(show=True, savefig=None, *args, **kwargs):
    """ Plot the spectral index map from the polaris output, in full resolution, namely,
        with no beam convolution.
    """
    # Read data
    prefix = Path(home/'phd/rhd_disk/results/snap541/dust_emission/temp_eos/sg/amax10um')

    # Generate spectral index map
    fig = utils.spectral_index(
        prefix/'1.3mm/0deg/dust_scat/data/polaris_I.fits', 
        prefix/'3mm/0deg/dust_scat/data/polaris_I.fits',
        vmin=1.7, 
        vmax=3.4, 
        savefile='.spectral_index_polaris.fits', 
        return_fig=True,
        *args,
        **kwargs
    )
    
    # Add a dashed line to indicate the region where the cut is extracted
    fig.ax.axvline(x=150, ymin=0.05, ymax=0.95, ls='--', lw=1.5, color='tab:blue')
    plt.tight_layout()
    
    return utils.plot_checkout(fig, show, savefig, path=home/f'phd/plots/paper1')
    

def plot_spectral_index_slice(show=True, savefig=None, figsize=(6,4.5)):
    """ Plot the vertical cut of the spectral index for the ALMA observation and for 
        the face-on RHD model. Figure for the appendix. 
    """
    
    # Read data
    obs = utils.Observation('3mm')
    prefix = Path(home/'phd/rhd_disk/results/snap541/dust_emission/temp_eos/sg/amax10um/')
    
    # Read brightness temperatures 
    intensity_model = fits.getdata(prefix/'3mm/0deg/dust_scat/data/polaris_I.fits').squeeze()

    # Read spectral indices
    alpha_iras = fits.getdata(home/'phd/observations/iras16293/' / \
        'sourceB_spectral_index_band3-6.fits').squeeze()
    alpha_model = fits.getdata(prefix/'.spectral_index_polaris.fits').squeeze()

    # Generate the plot
    fig, p = plt.subplots(nrows=1, ncols=1, figsize=figsize)

    #p.plot(np.linspace(-2, 2, 400), alpha_iras[200,:], color='black', label='IRAS16293-2422 B', ls='-.')
    p.plot(np.linspace(-0.35, 0.35, 300), alpha_model[150,:], color='black', label='RHD model (Face-on)')
    p.axhline(2, color='grey', linestyle='--')

    # Add a vertical axis on the right hand to show the Stokes I of the RHD model
    pi = p.twinx()
    pi.plot(np.linspace(-0.35, 0.35, 300), intensity_model[150,:]*1e6, color='tab:blue', ls='-')

    plt.legend()
    p.set_xticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    p.set_xlim(-0.35, 0.35)
    p.set_xlabel('Angular offset (arcseconds)')
    p.set_ylabel(r'Spectral index $\alpha$')
    pi.set_ylabel(r'Intensity ($\mu$Jy/pixel)')
    pi.tick_params(axis='y', labelcolor='tab:blue')

    plt.tight_layout() 

    return utils.plot_checkout(fig, show, savefig, path=home/f'phd/plots/paper1')



