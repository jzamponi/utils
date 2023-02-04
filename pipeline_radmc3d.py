#!/usr/bin/env python3
"""
    Pipeline script to calculate synthetic ALMA images from an SPH model,
    using RADMC3D for the raytracing and CASA for the ALMA simulation. 

    Example:

    $ pipeline_radmc3d.py --sphfile snap_001.dat --ncells 100 --bbox 50
        --show-grid-2d --show-grid-3d --raytrace --lam 3000 --amax 10 
        --polarization --opacity --material p --amin 0.1 --amax 10 --na 100 
        --show-rt --synobs --show-synobs

    For details, run:
    $ pipeline_radmc3d.py --help

    Requisites:
        Software:   python3, CASA, RADMC3D, 
                    Mayavi (optional), ParaView (optional)

        Modules:    python3-aplpy, python3-scipy, python3-numpy, python3-h5py
                    python3-matplotlib, python3-astropy, python3-mayavi,
                    python3-radmc3dPy

"""

import os, sys
import argparse
import requests
import subprocess
import numpy as np
from pathlib import Path
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from glob import glob

import utils
import dustmixer


class Pipeline:
    
    def __init__(self, lam=1300, amax=10, nphot=1e5, nthreads=1, sootline=300,
            lmin=0.1, lmax=1e6, nlam=200, star=None, dgrowth=False, csubl=0, 
            material='sg', polarization=False, alignment=False, overwrite=False, verbose=True):
        self.steps = []
        self.lam = int(lam)
        self.lmin = lmin
        self.lmax = lmax
        self.nlam = nlam
        self.lgrid = np.logspace(np.log10(lmin), np.log10(lmax), nlam)
        self.amax = str(int(amax))
        self.material = material
        self.nphot = int(nphot)
        self.nthreads = int(nthreads)
        self.polarization = polarization
        self.alignment = alignment
        if polarization:
            self.scatmode = 5
            self.inputstyle = 10
        else:
            self.scatmode = 2
            self.inputstyle = 1

        if alignment:
            self.scatmode = 4
            self.inputstyle = 20
            self.polarization = True
    
        self.csubl = csubl
        self.nspec = 1 if self.csubl == 0 else 2
        self.dcomp = [material]*2 if self.csubl == 0 else [material, material+'o']
        self.sootline = sootline
        self.dgrowth = dgrowth
        if star is None:
            self.xstar = 0
            self.ystar = 0
            self.zstar = 0
            self.rstar = 2e11
            self.mstar = 3e22
            self.tstar = 4000        
        else:
            self.xstar = star[0]
            self.ystar = star[1]
            self.zstar = star[2]
            self.rstar = star[3]
            self.mstar = star[4]
            self.tstar = star[5]

        self.overwrite = overwrite
        self.verbose = verbose

    @utils.elapsed_time
    def create_grid(self, sphfile, source='sphng', ncells=None, bbox=None,  
            rout=None, vector_field=None, show_2d=False, show_3d=False, 
            vtk=False, render=False):
        """ Initial step in the pipeline: creates an input grid for RADMC3D """

        self.sphfile = sphfile
        self.ncells = ncells
        self.bbox = bbox
        self.rout = rout
        self.vector_field = vector_field

        # Register the pipeline step 
        self.steps.append('create_grid')

        # Create a grid instance
        print('')
        utils.print_('Creating model grid ...\n', bold=True)
        self.grid = CartesianGrid(
            ncells=self.ncells, 
            bbox=self.bbox, 
            rout=self.rout,
            csubl=self.csubl, 
            nspec=self.nspec, 
            sootline=self.sootline, 
        )

        # Read the SPH data
        self.grid.read_sph(self.sphfile, source=source.lower())

        # Set a bounding box to trim the new grid
        if self.bbox is not None:
            self.grid.trim_box(bbox=self.bbox * u.au.to(u.cm))

        # Set a radius at which to trim the new grid
        if self.rout is not None:
            self.grid.trim_box(rout=self.rout * u.au.to(u.cm))

        # Interpolate the SPH points onto a regular cartesian grid
        self.grid.interpolate_points(field='dens', show_2d=show_2d, show_3d=show_3d)
        self.grid.interpolate_points(field='temp', show_2d=show_2d, show_3d=show_3d)

        # Write the new cartesian grid to radmc3d file format
        self.grid.write_grid_file()

        # Write the dust density distribution to radmc3d file format
        self.grid.write_density_file()
        
        # Write the dust temperature distribution to radmc3d file format
        self.grid.write_temperature_file()

        if self.vector_field is not None:
            self.grid.write_vector_field(morphology=self.vector_field)
        
        # Call RADMC3D to read the grid file and generate a VTK representation
        if vtk:
            self.grid.create_vtk(dust_density=False, dust_temperature=True, rename=True)
        
        # Visualize the VTK grid file using ParaView
        if render:
            self.grid.render()

    @utils.elapsed_time
    def dust_opacity(self, amin, amax, na, q=-3.5, nang=3, material=None, 
            show_nk=False, show_opac=False, savefig=None):
        """
            Call dustmixer to generate dust opacity tables. 
            New dust materials can be manually defined here if desired.
        """

        print('')
        utils.print_("Calculating dust opacities ...\n", bold=True)

        if material is not None: self.material = material
        self.amin = amin
        self.amax = amax
        self.na = na
        self.q = q 
        self.a_dist = np.logspace(np.log10(amin), np.log10(amax), na)
        if self.polarization and nang < 181:
            self.nang = 181
        else:
            self.nang = nang
        #nth = self.nthreads
        # use 1 until the parallelization of polarization is properly implemented
        nth = 1
        
        if self.material == 's':
            mix = dustmixer.Dust(name='Silicate')
            mix.set_nk('astrosil-Draine2003.lnk', skip=1, get_density=True)
            mix.set_lgrid(self.lmin, self.lmax, self.nlam)
            mix.get_opacities(a=self.a_dist, nang=self.nang, nproc=nth)
        
        elif self.material == 'g':
            mix = dustmixer.Dust(name='Graphite')
            mix.set_nk('c-gra-Draine2003.lnk', skip=1, get_density=True)
            mix.set_lgrid(self.lmin, self.lmax, self.nlam)
            mix.get_opacities(a=self.a_dist, nang=self.nang, nproc=nth)

        elif self.material == 'p':
            mix = dustmixer.Dust(name='Pyroxene')
            mix.set_nk('pyrmg70.lnk', get_density=False)
            mix.set_density(3.01, cgs=True)
            mix.set_lgrid(self.lmin, self.lmax, self.nlam)
            mix.get_opacities(a=self.a_dist, nang=self.nang, nproc=nth)
        
        elif self.material == 'o':
            mix = dustmixer.Dust(name='Organics')
            mix.set_nk('organics.nk', get_density=False)
            mix.set_density(1.50, cgs=True)
            mix.set_lgrid(self.lmin, self.lmax, self.nlam)
            mix.get_opacities(a=self.a_dist, nang=self.nang, nproc=nth)
        
        elif self.material == 'sg':
            sil = dustmixer.Dust(name='Silicate')
            gra = dustmixer.Dust(name='Graphite')
            sil.set_nk('astrosil-Draine2003.lnk', skip=1, get_density=True)
            gra.set_nk('c-gra-Draine2003.lnk', skip=1, get_density=True)
            sil.set_lgrid(self.lmin, self.lmax, self.nlam)
            gra.set_lgrid(self.lmin, self.lmax, self.nlam)
            sil.get_opacities(a=self.a_dist, nang=self.nang, nproc=nth)
            gra.get_opacities(a=self.a_dist, nang=self.nang, nproc=nth)

            # Sum the opacities weighted by their mass fractions
            mix = sil * 0.625 + gra * 0.375

        else:
            try:
                mix = dustmixer.Dust(self.material.split('/')[-1].split('.')[0])
                mix.set_nk(path=self.material, skip=1, get_density=True)
                mix.set_lgrid(self.lmin, self.lmax, self.nlam)
                mix.get_opacities(a=self.a_dist, nang=self.nang, nproc=nth)
                self.material = mix.name

            except Exception as e:
                utils.print_(e, red=True)
                raise ValueError(f'Material = {material} not found.')

        if show_nk or savefig is not None:
            mix.plot_nk(show=show_nk, savefig=savefig)

        if show_opac or savefig is not None:
            mix.plot_opacities(show=show_opac, savefig=savefig)

        # Write the opacity table
        mix.write_opacity_file(scatmat=self.polarization, 
            name=f'{self.material}-a{int(self.amax)}um')

        # Write the alignment efficiencies
        if self.alignment:
            mix.write_align_factor(f'{self.material}-a{int(self.amax)}um')

        # Register the pipeline step 
        self.steps.append('dustmixer')
    
    def radmc3d_banner(self):
        utils.print_(f'{"="*21}  <RADMC3D>  {"="*21}', bold=True)

    def generate_input_files(self, inpfile=False, wavelength=False, stars=False, 
            dustopac=False, dustkappa=False, dustkapalignfact=False,
            grainalign=False):
        """ Generate the necessary input files for radmc3d """
        if inpfile:
            # Create a RADMC3D input file
            with open('radmc3d.inp', 'w+') as f:
                f.write(f'incl_dust = 1\n')
                f.write(f'istar_sphere = 0\n')
                f.write(f'modified_random_walk = 1\n')
                f.write(f'setthreads = {self.nthreads}\n')
                f.write(f'nphot_scat = {self.nphot}\n')
                f.write(f'scattering_mode = {self.scatmode}\n')
                if self.alignment: f.write(f'alignment_mode = 1\n')

        if wavelength: 
            # Create a wavelength grid in micron
            with open('wavelength_micron.inp', 'w+') as f:
                f.write(f'{self.lgrid.size}\n')
                for wav in self.lgrid:
                    f.write(f'{wav:13.6}\n')

        if stars:
            # Create a stellar spectrum file
            with open('stars.inp', 'w+') as f:
                f.write('2\n')
                f.write(f'1 {self.lgrid.size}\n')
                f.write(f'{self.rstar} {self.mstar} ')
                f.write(f'{self.xstar} {self.ystar} {self.zstar}\n')
                for wav in self.lgrid:
                    f.write(f'{wav:13.6}\n')
                f.write(f'{-self.tstar}\n')

        if dustopac:
            # Create a dust opacity file
            self.amax = int(self.amax)
            with open('dustopac.inp', 'w+') as f:
                f.write('2\n')
                f.write(f'{self.nspec}\n')
                f.write('---------\n')
                f.write(f'{self.inputstyle}\n')
                f.write('0\n')
                if self.csubl > 0:
                    f.write(f'{self.dcomp[1]}-a{self.amax}um-'+ \
                        '{int(self.csubl)}org\n')
                else:
                    f.write(f'{self.material}-a{self.amax}um\n')

                if self.nspec > 1:
                    # Define a second species 
                    f.write('---------\n')
                    f.write(f'{self.inputstyle}\n')
                    f.write('0\n')
                    if self.dgrowth:
                        f.write(f'{self.dcomp[0]}-a1000um\n')
                    else:
                        f.write(f'{self.dcomp[0]}-a{self.amax}um\n')
                f.write('---------\n')

        if dustkappa:

            # Fetch the corresponding opacity table from a public repo
            table = 'https://raw.githubusercontent.com/jzamponi/utils/main/' +\
                f'opacity_tables/dustkappa_{self.dcomp[0]}-a{self.amax}um.inp'

            utils.download_file(table)

            if self.csubl > 0:
                # Download also the table for a second dust composition
                table = table.replace(f'{self.dcomp[0]}', f'{self.dcomp[1]}')
                if 'sgo' in table:
                    table = table.replace('um.inp', f'um-{int(self.csubl)}org.inp')

                if self.dgrowth:
                    # Download also the table for grown dust
                    table = table.replace(f'{self.amax}', '1000') 

                utils.download_file(table)

        if dustkapalignfact:
            # To do: convert the graphite_oblate.dat and silicate_oblate.dat 
            # from the polaris repo, into a radmc3d format. Then upload the
            # radmc3d table to my github repo and download it from here
            
            raise ImportError(f'{utils.color.red}There is no ' +\
                f'dustkapalignfact_*.inp file. Run the pipeline again with ' +\
                f'the option --opacity --alignment.{utils.color.none}')

        if grainalign:
            raise ImportError(f'{utils.color.red}There is no ' +\
                f'grainalign_dir.inp file. Run the pipeline again adding ' +\
                f'--vector-field to --grid to create the alignment field ' +\
                f'from the input model.{utils.color.none}')

    def file_exists(self, filename):
        """ Raise an error if a file doesnt exist. Supports linux wildcards. """

        msg = f'{utils.color.red}{filename} not found.{utils.color.none}'
        if '*' in filename:
            if len(glob(filename)) == 0:
                raise FileNotFoundError(msg)
            
        elif not os.path.exists(filename): 
            raise FileNotFoundError(msg)

    @utils.elapsed_time
    def monte_carlo(self, nphot, radmc3d_cmds=''):
        """ 
            Call radmc3d to calculate the radiative temperature distribution 
        """

        print('')
        utils.print_("Running a thermal Monte Carlo ...", bold=True)
        self.nphot = nphot
        self.radmc3d_banner()
        subprocess.run(
            f'radmc3d mctherm nphot {self.nphot} {radmc3d_cmds}'.split())
        self.radmc3d_banner()

        # Register the pipeline step 
        self.steps.append('monte_carlo')

    @utils.elapsed_time
    def raytrace(self, incl, npix, sizeau, lam=None, show=True, noscat=False, 
            fitsfile='radmc3d_I.fits', radmc3d_cmds=''):
        """ 
            Call radmc3d to raytrace the newly created grid and plot an image 
        """

        print('')
        utils.print_("Ray-tracing the model density and temperature ...\n", 
            bold=True)

        if lam is not None:
            self.lam = lam

        # To do: What's the diff. between passing noscat and setting scatmode=0
        if noscat: self.scatmode = 0

        # Generate only the input files that are not available in the directory
        if not os.path.exists('radmc3d.inp') or self.overwrite:
            self.generate_input_files(inpfile=True)

        if not os.path.exists('wavelength_micron.inp') or self.overwrite:
            self.generate_input_files(wavelength=True)

        if not os.path.exists('stars.inp') or self.overwrite:
            self.generate_input_files(stars=True)

        # Write a new dustopac file only if dustmixer was used or if unexistent
        if not os.path.exists('dustopac.inp') or \
            'dustmixer' in self.steps or self.overwrite:
            self.generate_input_files(dustopac=True)

        # If opacites were calculated within the pipeline, don't overwrite them
        if 'dustmixer' not in self.steps:
            # If not manually provided, download it from the repo
            if not self.polarization:
                if len(glob('dustkappa*')) == 0 or self.overwrite:
                    self.generate_input_files(dustkappa=True)

        # If align factors were calculated within the pipeline, don't overwrite
        if self.alignment:
            if 'dustmixer' not in self.steps:
                # If not manually provided, download it from the repo
                if len(glob('dustkapalignfact*')) == 0 or self.overwrite:
                    self.generate_input_files(dustkapalignfact=True)

            if not os.path.exists('grainalign_dir.inp'):
                self.generate_input_files(grainalign=True)

        # Now double check that all necessary input files are available 
        self.file_exists('amr_grid.inp')
        self.file_exists('dust_density.inp')
        self.file_exists('dust_temperature.dat')
        self.file_exists('radmc3d.inp')
        self.file_exists('wavelength_micron.inp')
        self.file_exists('stars.inp')
        self.file_exists('dustopac.inp')
        self.file_exists('dustkapscat*' if self.polarization else 'dustkappa*')
        if self.alignment: 
            self.file_exists('dustkapalignfact*')
            self.file_exists('grainalign_dir.inp')
 
        # Explicitly the model rotate by 180.
        # Only for the current model. This line should be later removed.
        incl = 180 - int(incl)

        # Set the RADMC3D command by concatenating options
        cmd = f'radmc3d image '
        cmd += f'lambda {lam} ' if lam is not None else ' '
        cmd += f'incl {incl} ' if incl is not None else ' '
        cmd += f'npix {npix} ' if npix is not None else ' '
        cmd += f'sizeau {sizeau} ' if sizeau is not None else ' '
        cmd += f'stokes ' if self.polarization else ' '
        cmd += f'{" ".join(radmc3d_cmds)} '
        
        # Call RADMC3D and pipe the output also to radmc3d.out
        utils.print_(f'Executing command: {cmd}')
        self.radmc3d_banner()

        try:
            os.system(f'{cmd} 2>&1 | tee radmc3d.out')
        except KeyboardInterrupt:
            raise Exception('Received SIGKILL. Execution halted by user.')

        self.radmc3d_banner()
        
        # Read radmc3d.out and stop the pipeline if RADMC3D finished in error
        with open ('radmc3d.out', 'r') as out:
            for line in out.readlines():
                if 'error' in line.lower() or 'stop' in line.lower():
                    raise Exception(
                        f'{utils.color.red}[RADMC3D] {line}{utils.color.none}')
    
        # Generate a FITS file from the image.out
        if os.path.exists(fitsfile): os.remove(fitsfile)
        utils.radmc3d_casafits(fitsfile, stokes='I')

        # Clean extra keywords from the header to avoid APLPy axis errors
        utils.edit_header(fitsfile, 'CDELT3', 'del', False)
        utils.edit_header(fitsfile, 'CRVAL3', 'del', False)
        utils.edit_header(fitsfile, 'CUNIT3', 'del', False)
        utils.edit_header(fitsfile, 'CTYPE3', 'del', False)
        utils.edit_header(fitsfile, 'CRPIX3', 'del', False)

        # Also for Q and U Stokes components if considering polarization
        if self.polarization:
            for s in ['Q', 'U']:
                # Write FITS file for each component
                stokesfile = f'radmc3d_{s}.fits'
                if os.path.exists(stokesfile):
                        os.remove(stokesfile)
                utils.radmc3d_casafits(stokesfile, stokes=s)

                # Clean extra keywords from the header to avoid APLPy errors 
                utils.edit_header(stokesfile, 'CDELT3', 'del', False)
                utils.edit_header(stokesfile, 'CRVAL3', 'del', False)
                utils.edit_header(stokesfile, 'CUNIT3', 'del', False)
                utils.edit_header(stokesfile, 'CTYPE3', 'del', False)
                utils.edit_header(stokesfile, 'CRPIX3', 'del', False)

        # Plot the new image in Jy/pixel
        if show:
            utils.print_('Plotting image.out')

            if self.polarization:
                fig = utils.polarization_map(
                    source='radmc3d',
                    render='I', 
                    rotate=0, 
                    step=15, 
                    scale=10, 
                    min_pfrac=0, 
                    const_pfrac=True, 
                    vector_color='white',
                    vector_width=1, 
                    verbose=False,
                )
            else:
                fig = utils.plot_map(
                    filename='radmc3d_I.fits', 
                    bright_temp=False,
                    verbose=False,
                )

        # Register the pipeline step 
        self.steps.append('raytrace')

    @utils.elapsed_time
    def synthetic_observation(self, lam=None, show=False, cleanup=True, 
            graphic=True, verbose=False):
        """ 
            Prepare the input for the CASA simulator from the RADMC3D output,
            and call CASA to run a synthetic observation.
        """

        print('')
        utils.print_('Running synthetic observation ...\n', bold=True)

        # Read in the current pipeline wavelength
        lam = self.lam if lam is None else str(int(lam))

        # Dict containing only the wavelengths with a script available
        lam_mm = {
            '1300': '1.3mm',
            '3000': '3mm',
            '7000': '7mm',
            '18000': '18mm',
        }
        try:
            lam = lam_mm[lam]
        except KeyError as e:
            raise Exception('CASA scripts are currently only available at ' + \
            '1.3, 3, 7 & 18 mm.') from e
        
        # Make sure the CASA script is available in the current directory
        obs = 'alma' if lam in ['1.3mm', '3mm'] else 'vla'
        script = f'{obs}_simulation.py'  

        if not os.path.exists(script):
            # Download the CASA script
            url = 'https://raw.githubusercontent.com/jzamponi/utils/main/' + \
                f'synthetic_observations/{lam}/{script}'

            utils.download_file(
                url, f'No CASA script found. Downloading from: {url}')

            # Tailor the script
            subprocess.run(f"sed -i s/polaris/radmc3d/g {script}", shell=True) 
            if not graphic:
                subprocess.run(f"sed -i 's/both/file/g' {script}", shell=True)
            if not verbose:
                subprocess.run(
                f"sed -i 's/verbose = True/verbose = False/g' {script}",
                shell=True)

        # Delete any previous project to avoid the script clashing
        if len(glob('band*')) > 0:
            subprocess.run('rm -r band*', shell=True)

        # Run the ALMA/JVLA simulation script
        subprocess.run(f'casa -c {script} --nologger'.split())

        # Show the new synthetic image
        if show:
            utils.print_(f'Plotting the new synthetic image')

            if self.polarization:
                fig = utils.polarization_map(
                    source=obs,
                    render='I', 
                    rotate=0, 
                    step=15, 
                    scale=10, 
                    min_pfrac=0, 
                    const_pfrac=True, 
                    vector_color='white',
                    vector_width=1, 
                    verbose=False,
                )
            else:
                fig = utils.plot_map(
                    filename=f'{obs}_I.fits', 
                    bright_temp=False,
                    verbose=False,
                )

        # Clean-up and remove unnecessary files created by CASA
        if cleanup:
            subprocess.run('rm *.last casa-*.log', shell=True)

        # Register the pipeline step 
        self.steps.append('synobs')



class CartesianGrid(Pipeline):
    def __init__(self, ncells, bbox=None, rout=None, fill='min', csubl=0, 
            nspec=1, sootline=300):
        """ 
        Create a cartesian grid from a set of 3D points.

        The input SPH particle coordinates should be given as cartesian 
        coordinates in units of cm.

        The box can be trimmed to a given size calling self.trim_box. The 
        size can be given either as single half-length for a retangular box, 
        as cartesian vertices of a bounding box or as the radius of a sphere. 
        Any quantity should be given in units of cm.

        Examples: bbox = [[xmin, xmax], [ymin, ymax], [zmin, zmax]]  
                  bbox = 50 
                  rout = 50

        """
        self.ncells = ncells
        self.bbox = bbox
        self.rout = rout
        self.g2d = 100
        self.fill = fill
        self.csubl = csubl
        self.nspec = nspec
        self.sootline = sootline
        self.carbon = 0.375
        self.subl_mfrac = 1 - self.carbon * self.csubl/100

    def read_sph(self, filename, source='sphng'):
        """ Read SPH data """

        utils.print_(f'Reading data from: {filename} | Format: {source}')
        if not os.path.exists(filename): 
            raise FileNotFoundError('Input SPH file does not exist')

        if source == 'sphng':
            self.sph = utils.read_sph(filename, remove_sink=True, cgs=True)
            self.x = self.sph[:, 2]
            self.y = self.sph[:, 3]
            self.z = self.sph[:, 4]
            self.dens = self.sph[:, 10] / self.g2d
            self.temp = self.sph[:, 11]
            self.npoints = len(self.dens)

        elif source == 'gizmo':
            import h5py
            self.sph = h5py.File(filename, 'r')['PartType0']
            coords = np.array(self.sph['Coordinates']) * 1.43 * u.kpc.to(u.cm)
            self.x = coords[:, 0]
            self.y = coords[:, 1]
            self.z = coords[:, 2]
            self.dens = np.array(self.sph['Density']) * 1.382e-21 / self.g2d
            self.temp = np.array(self.sph['KromeTemperature'])
            self.npoints = len(self.dens)

            # Recenter the particles based on the center of mass
            self.x -= np.average(self.x, weights=self.dens)
            self.y -= np.average(self.y, weights=self.dens)
            self.z -= np.average(self.z, weights=self.dens)
            
        else:
            raise ValueError(f'Source = {source} is currently not supported')

    def trim_box(self, bbox=None, rout=None):
        """ 
        Trim the original grid to a given size, can be rectangular or spherical. 
        """

        # Emtpy dynamic array to store the indices of the particles to remove
        to_remove = []

        if bbox is not None:

            self.bbox = bbox

            # Create a matrix of vertices if bbox is provided as a single number
            if np.isscalar(bbox):
                bbox = [[-bbox, bbox]] * 3
            
            # Store the vertices
            x1 = bbox[0][0]
            x2 = bbox[0][1]
            y1 = bbox[1][0]
            y2 = bbox[1][1]
            z1 = bbox[2][0]
            z2 = bbox[2][1]
            utils.print_(f'Deleting particles outside a box ' +
                f'half-length of {self.bbox * u.cm.to(u.au)} au')

            # Iterate over particles and delete upon reject
            for i in range(self.x.size):
                if self.x[i] < x1 or self.x[i] > x2:
                    to_remove.append(i)

            for j in range(self.y.size):
                if self.y[j] < y1 or self.y[j] > y2:
                    to_remove.append(j)

            for k in range(self.z.size):
                if self.z[k] < z1 or self.z[k] > z2:
                    to_remove.append(k)

        
        if rout is not None and bbox is None:
            # Override any previous value of rout
            self.rout = rout
            utils.print_('Deleting particles outside a radius of ' +
                f'{self.rout * u.cm.to(u.au)} au ...')

            # Convert cartesian to polar coordinates to define a radial trim
            r = np.sqrt(self.x**2 + self.y**2 + self.z**2)

            for i in range(self.x.size):
                if r[i] > self.rout:
                    to_remove.append(i)

        # Remove the particles from each quantity
        self.x = np.delete(self.x, to_remove)
        self.y = np.delete(self.y, to_remove)
        self.z = np.delete(self.z, to_remove)
        self.dens = np.delete(self.dens, to_remove)
        self.temp = np.delete(self.temp, to_remove)
        utils.print_(f'{self.npoints - self.x.size} ' +
            'particles were not included in the grid')

    def interpolate_points(self, field='temp', show_2d=False, show_3d=False):
        """
            Interpolate a set of points in cartesian coordinates along with their
            values into a rectangular grid.
        """

        # Construct the rectangular grid
        utils.print_('Creating a box of ' +
            f'{self.ncells} x {self.ncells} x {self.ncells} cells')
        rmin = np.min([self.x.min(), self.y.min(), self.z.min()])
        rmax = np.max([self.x.max(), self.y.max(), self.z.max()])
        self.xc = np.linspace(rmin, rmax, self.ncells)
        self.yc = np.linspace(rmin, rmax, self.ncells)
        self.zc = np.linspace(rmin, rmax, self.ncells)
        self.X, self.Y, self.Z = np.meshgrid(self.xc, self.yc, self.zc)

        # Determine which quantity is to be interpolated
        if field == 'dens':
            utils.print_(f'Interpolating density values onto the grid')
            values = self.dens
        elif field == 'temp':
            utils.print_(f'Interpolating temperature values onto the grid')
            values = self.temp

        # Interpolate the point values at the grid points
        fill = np.min(values) if self.fill == 'min' else self.fill
        xyz = np.vstack([self.x, self.y, self.z]).T
        interp = griddata(xyz, values, (self.X, self.Y, self.Z), 
            method='linear', fill_value=fill)

        # Plot the midplane at z=0 using Matplotlib
        if show_2d:
            try:
                from matplotlib.colors import LogNorm
                utils.print_(f'Plotting the {field} grid midplane at z = 0')
                plt.rcParams['text.usetex'] = True
                plt.rcParams['font.family'] = 'Times New Roman'

                if self.bbox is None:
                    extent = self.bbox
                else:
                    extent = [-self.bbox*u.cm.to(u.au), 
                            self.bbox*u.cm.to(u.au)] * 2

                if field == 'dens':
                    plt.imshow(interp[:,:,self.ncells//2-1].T * 100, 
                        norm=LogNorm(vmin=None, vmax=None),
                        cmap='BuPu',
                        extent=extent,
                    )
                else:
                    plt.imshow(interp[:,:,self.ncells//2-1].T, 
                        norm=LogNorm(vmin=None, vmax=None),
                        cmap='inferno',
                        extent=extent,
                    )
                plt.colorbar()
                plt.xlabel('AU')
                plt.ylabel('AU')
                plt.title({
                    'dens': r'Density Midplane at $z=0$ (g cm$^-3$)', 
                    'temp': r'Temperature Midplane at $z=0$ (K)'
                }[field])
                plt.show()

            except Exception as e:
                utils.print_('Unable to show the 2D grid slices.',  bold=True)
                utils.print_(e, bold=True)
        
        if show_3d:
            # Render the interpolated 3D field using Mayavi
            try:
                utils.print_('Visualizing the interpolated field ...')
                from mayavi import mlab
                mlab.contour3d(interp, contours=20, opacity=0.2)
                mlab.show()

            except Exception as e:
                utils.print_('Unable to show the 3D grid.',  bold=True)
                utils.print_(e, bold=True)

        # Store the interpolated field
        if field == 'dens':
            self.interp_dens = interp
        elif field == 'temp':
            self.interp_temp = interp


    def write_grid_file(self):
        """ Write the regular cartesian grid file """
        with open('amr_grid.inp','w+') as f:
            # iformat
            f.write('1\n')                     
            # Regular grid
            f.write('0\n')                      
            # Coordinate system: cartesian
            f.write('1\n')                     
            # Gridinfo
            f.write('0\n')                       
            # Number of cells
            f.write('1 1 1\n')                   
            # Size of the grid
            f.write(f'{self.ncells:d} {self.ncells:d} {self.ncells:d}\n')

            # Shift the cell centers right to set the cell walls
            dx = np.diff(self.xc)[0]
            dy = np.diff(self.yc)[0]
            dz = np.diff(self.zc)[0]
            xw = self.xc + dx
            yw = self.yc + dy
            zw = self.zc + dz

            # Add the missing wall at the beginning of each axis
            xw = np.insert(xw, 0, self.xc[0])
            yw = np.insert(yw, 0, self.yc[0])
            zw = np.insert(zw, 0, self.zc[0])

            # Write the cell walls
            for i in xw:
                f.write(f'{i:13.6e}\n')      
            for j in yw:
                f.write(f'{j:13.6e}\n')      
            for k in zw:
                f.write(f'{k:13.6e}\n')      


    def write_density_file(self):
        """ Write the density file """
        utils.print_('Writing dust density file')

        # Flatten the array into a 1D fortran-style indexing
        density = self.interp_dens.ravel(order='F')
        with open('dust_density.inp','w+') as f:
            f.write('1\n')                      
            f.write(f'{density.size:d}\n')
            f.write(f'{self.nspec}\n')                

            if self.nspec == 1:
                # Write a single dust species
                for d in density:
                    f.write(f'{d:13.6e}\n')
            else:
                utils.print_(f'Writing two density species ...')
                # Write two densities: 
                # one with original value outside the sootline and zero within 
                temp1d = self.interp_temp.ravel(order='F')
                for i, d in enumerate(density):
                    if temp1d[i] < self.sootline:
                        f.write(f'{d:13.6e}\n')
                    else:
                        f.write('0\n')

                # one with zero outside the sootline and reduced density within
                for i, d in enumerate(density):
                    if temp1d[i] < self.sootline:
                        f.write('0\n')
                    else:
                        f.write(f'{(d * self.subl_mfrac):13.6e}\n')


    def write_temperature_file(self):
        """ Write the temperature file """
        utils.print_('Writing dust temperature file')
        
        temperature = self.interp_temp.ravel(order='F')
        with open('dust_temperature.dat','w+') as f:
            f.write('1\n')                      
            f.write(f'{temperature.size:d}\n')
            f.write(f'{self.nspec}\n')                

            # Write the temperature Nspec times for Nspec dust species
            for i in range(self.nspec):
                for t in temperature:
                    f.write(f'{t:13.6e}\n')


    def create_vtk(self, dust_density=False, dust_temperature=True, rename=False):
        """ Call radmc3d to create a VTK file of the grid """
        self.radmc3d_banner()

        if dust_density:
            subprocess.run('radmc3d vtk_dust_density 1'.split())
            if rename:
                subprocess.run('mv model.vtk model_dust_density.vtk'.split())

        if dust_temperature:
            subprocess.run('radmc3d vtk_dust_temperature 1'.split())
            if rename:
                subprocess.run('mv model.vtk model_dust_temperature.vtk'.split())

        if not dust_density and not dust_temperature:
            subprocess.run('radmc3d vtk_grid'.split())

        self.radmc3d_banner()

    def render(self, state=None, dust_density=False, dust_temperature=True):
        """ Render the new grid in 3D using ParaView """
        if isinstance(state, str):
            subprocess.run(f'paraview --state {state} 2>/dev/null'.split())
        else:
            try:
                if dust_density:
                    subprocess.run(
                    f'paraview model_dust_density.vtk 2>/dev/null'.split())
                elif dust_temperature:
                    subprocess.run(
                    f'paraview model_dust_temperature.vtk 2>/dev/null'.split())
            except Exception as e:
                utils.print_('Unable to render using ParaView.',  bold=True)
                utils.print_(e, bold=True)

    def write_vector_field(self, morphology):
        """ Create a vector field for dust alignment """

        utils.print_('Writing grain alignment direction file')

        self.vector_field = morphology
        field = np.zeros((self.ncells, self.ncells, self.ncells, 3))
        xx = self.X
        yy = self.Y
        zz = self.Z

        if morphology in ['t', 'toroidal']:
            rrc = np.sqrt(xx**2 + yy**2)
            field[..., 0] = yy / rrc
            field[..., 1] = -xx / rrc

        elif morphology in ['r', 'radial']:
            field[..., 0] = yy
            field[..., 1] = xx

        elif morphology in ['h', 'hourglass']:
            pass
        
        elif morphology == 'x':
            field[..., 0] = 1.0

        elif morphology == 'y':
            field[..., 1] = 1.0

        elif morphology == 'z':
            field[..., 2] = 1.0

        # Normalize the field
        l = np.sqrt(field[..., 0]**2 + field[..., 1]**2 + field[..., 2]**2)
        field[..., 0] = np.squeeze(field[..., 0]) / l
        field[..., 1] = np.squeeze(field[..., 1]) / l
        field[..., 2] = np.squeeze(field[..., 2]) / l

        # Assume perfect alignment (a_eff = 1). We can change it in the future
        a_eff = 1
        field[..., 0] *= a_eff
        field[..., 1] *= a_eff
        field[..., 2] *= a_eff
        
        # Write the vector field 
        with open('grainalign_dir.inp','w+') as f:
            f.write('1\n')
            f.write(f'{int(self.ncells**3)}\n')
            for iz in range(self.ncells):
                for iy in range(self.ncells):
                    for ix in range(self.ncells):
                        f.write(f'{field[ix, iy, iz, 0]:13.6e} ' +\
                                f'{field[ix, iy, iz, 1]:13.6e} ' +\
                                f'{field[ix, iy, iz, 2]:13.6e}\n')


if __name__ == "__main__":

    # Initialize the argument parser
    parser = argparse.ArgumentParser(prog='Pipeline',
        description='Pipeline for synthetic observations. It can convert ' + \
        'from sphNG binary files to a RADMC3D regular cartesian grid and ' + \
        'run a raytracing and synthetic ALMA/JVLA observation.')

    # Define command line options
    parser.add_argument('-g', '--grid', action='store_true', default=False, 
        help='Create an input grid for the radiative transfer')

    parser.add_argument('--sphfile', action='store', default='', 
        help='Name of the input SPH file')

    parser.add_argument('--source', action='store', default='sphng', 
        help='Name of the code used to generate the sphfile.')

    parser.add_argument('--ncells', action='store', type=int, default=100,
        help='Number of cells in every direction')

    exclusive = parser.add_mutually_exclusive_group() 
    exclusive.add_argument('--bbox', action='store', type=float, default=None, 
        help='Size of the half-lenght of a bounding box in au (i.e., zoom in)')

    exclusive.add_argument('--rout', action='store', type=float, default=None, 
        help='Size of the outer radial boundary in au (i.e., zoom in)')

    parser.add_argument('--vector-field', action='store', type=str, default=None, 
        choices=['t', 'toroidal', 'r', 'radial', 'h', 'hourglass', 'x', 'y', 'z'], 
        help='Create a vectory field for alignment of elongated grains.')

    parser.add_argument('--show-grid-2d', action='store_true', default=False,
        help='Plot the midplane of the newly created grid')

    parser.add_argument('--show-grid-3d', action='store_true', default=False,
        help='Render the new cartesian grid in 3D')

    parser.add_argument('--vtk', action='store_true', default=False,
        help='Call RADCM3D to create a VTK file of the newly created grid')

    parser.add_argument('--render', action='store_true', default=False,
        help='Visualize the VTK file using ParaView')

    parser.add_argument('-op', '--opacity', action='store_true', default=False,
        help='Call dustmixer to generate a dust opacity table')

    parser.add_argument('--material', action='store', default='sg',
        help='Dust optical constants. Can be a predefined key, a path or a url')

    parser.add_argument('--amin', action='store', type=float, default=0.1,
        help='Minimum value for the grain size distribution')

    parser.add_argument('--amax', action='store', type=float, default=10,
        help='Maximum value for the grain size distribution')

    parser.add_argument('--na', action='store', type=int, default=100,
        help='Number of size bins for the logarithmic grain size distribution')

    parser.add_argument('--q', action='store', type=float, default=-3.5,
        help='Slope of the grain size distribution in logspace')

    parser.add_argument('--nang', action='store', type=int, default=3,
        help='Number of scattering angles used to sample the dust efficiencies')

    parser.add_argument('--show-opacity', action='store_true', default=False,
        help='Plot the resulting dust opacities.')

    parser.add_argument('--show-nk', action='store_true', default=False,
        help='Plot the input dust optical constants.')

    parser.add_argument('-mc', '--monte-carlo', action='store_true', default=False,
        help='Call RADMC3D to raytrace the new grid and plot an image')

    parser.add_argument('--nphot', action='store', type=float, default=1e5,
        help='Set the number of photons for scattering and thermal Monte Carlo')

    parser.add_argument('--nthreads', action='store', default=1, 
        help='Number of threads used for the Monte-Carlo runs')

    parser.add_argument('-rt', '--raytrace', action='store_true', default=False,
        help='Call RADMC3D to raytrace the new grid and plot an image')

    parser.add_argument('--lam', action='store', type=float, default=1300,
        help='Wavelength used to generate an image in units of micron')

    parser.add_argument('--lmin', action='store', type=float, default=0.1,
        help='Lower end of the wavelength grid.')

    parser.add_argument('--lmax', action='store', type=float, default=1e6,
        help='Upper end of the wavelength grid.')

    parser.add_argument('--nlam', action='store', type=float, default=200,
        help='Number of wavelengths to build a logarithmically spaced grid.')

    parser.add_argument('--npix', action='store', type=int, default=300,
        help='Number of pixels per side of new image')

    parser.add_argument('--incl', action='store', type=float, default=0,
        help='Inclination angle of the grid in degrees')

    parser.add_argument('--sizeau', action='store', type=int, default=100,
        help='Physical size of the image in AU')

    parser.add_argument('--star', action='store', default=None, nargs=6, 
        help='6 params to define a radiating star (cgs): x y z Rstar Mstar Teff')

    parser.add_argument('--polarization', action='store_true', default=False,
        help='Enable polarized RT and full scattering matrix opacity tables')

    parser.add_argument('--alignment', action='store_true', default=False,
        help='Enable polarized RT of thermal emission from aligned grains.')

    parser.add_argument('--noscat', action='store_true', default=False,
        help='Turn off the addition of scattered flux to the thermal flux')

    parser.add_argument('--sublimation', action='store', type=float, default=0,
        help='Percentage of refractory carbon that evaporates from the dust')

    parser.add_argument('--soot-line', action='store', type=float, default=300,
        help='Temperature at which carbon is supposed to sublimate')

    parser.add_argument('--dust-growth', action='store_true', default=False,  
        help='Enable dust growth within the soot-line')

    parser.add_argument('--show-rt', action='store_true', default=False,
        help='Plot the intensity map generated by ray-tracing')

    parser.add_argument('--radmc3d', action='store', type=str, default='',
        help='Additional commands to be passed to RADMC3D.', nargs="*")

    parser.add_argument('--synobs', action='store_true', default=False,
        help='Call CASA to run a synthetic observation from the new image')

    parser.add_argument('--show-synobs', action='store_true', default=False,
        help='Plot the ALMA/JVLA synthetic image generated by CASA')

    parser.add_argument('--overwrite', action='store_true', default=False,
        help='Overwrite radmc3d input files.')

    parser.add_argument('--quiet', action='store_true', default=False,
        help='Disable verbosity.')


    # Store the command-line given arguments
    cli = parser.parse_args()

    @utils.elapsed_time
    def pipeline():
        # Initialize the pipeline
        pipeline = Pipeline(lam=cli.lam, amax=cli.amax, nphot=cli.nphot, 
            lmin=cli.lmin, lmax=cli.lmax, nlam=cli.nlam, nthreads=cli.nthreads, 
            csubl=cli.sublimation, sootline=cli.soot_line, dgrowth=cli.dust_growth,
            polarization=cli.polarization, alignment=cli.alignment, star=cli.star, 
            material=cli.material, overwrite=cli.overwrite, verbose=not cli.quiet)

        # Generate the input grid for RADMC3D
        if cli.grid:
            pipeline.create_grid(sphfile=cli.sphfile, source=cli.source,  
                ncells=cli.ncells, bbox=cli.bbox, rout=cli.rout, 
                render=cli.render, vtk=cli.vtk, show_2d=cli.show_grid_2d, 
                show_3d=cli.show_grid_3d, vector_field=cli.vector_field)

        # Generate the dust opacity tables
        if cli.opacity:
            pipeline.dust_opacity(cli.amin, cli.amax, cli.na, cli.q, cli.nang,
                show_nk=cli.show_nk, show_opac=cli.show_opacity)

        # Run a thermal Monte-Carlo
        if cli.monte_carlo:
            pipeline.monte_carlo(nphot=cli.nphot, radmc3d_cmds=cli.radmc3d)

        # Run a ray-tracing on the new grid and generate an image
        if cli.raytrace:
            pipeline.raytrace(lam=cli.lam, incl=cli.incl, npix=cli.npix, 
                sizeau=cli.sizeau, show=cli.show_rt, noscat=cli.noscat,
                radmc3d_cmds=cli.radmc3d)

        # Run a synthetic observation of the new image by calling CASA
        if cli.synobs:
            pipeline.synthetic_observation(show=cli.show_synobs, lam=cli.lam, 
                graphic=False, verbose=False)

    pipeline()

