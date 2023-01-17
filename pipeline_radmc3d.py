#!/usr/bin/env python3
"""
    Pipeline script to calculate synthetic ALMA images from an SPH model,
    using RADMC3D for the raytracing and CASA for the ALMA simulation. 

    Example:

    $ pipeline_radmc3d.py --sphfile snap_001.dat --ncells 100 --bbox 50
        --show-grid-2d --show-grid-3d --raytrace --lam 3000 --amax 10 
        --polarization --show-rt --synobs --show-synobs

    For details, run:
    $ pipeline_radmc3d.py --help

"""

import os
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
from dustmixer import Dust


class Pipeline:
    
    def __init__(self, lam=1300, amax=10, nphot=1e5, nthreads=4, csubl=0, 
            dgrowth=False, sootline=300, polarization=False):
        self.steps = []
        self.lam = int(lam)
        self.amax = str(int(amax))
        self.nphot = int(nphot)
        self.nthreads = int(nthreads)
        self.polarization = polarization
        self.scatmode = 5 if polarization else 2
        self.inputstyle = 10 if polarization else 1
        self.wavelengths = np.logspace(-2, 8, 200)
        self.csubl = csubl
        self.nspec = 1 if self.csubl == 0 else 2
        self.dcomp = ['sg', 'sg'] if self.csubl == 0 else ['sg','sgo']
        self.sootline = sootline
        self.dgrowth = dgrowth
        self.rstar = 2e11
        self.mstar = 3e22
        self.tstar = 4000        
        self.xstar = 0
        self.ystar = 0
        self.zstar = 0

    @utils.elapsed_time
    def create_grid(self, sphfile, ncells=None, bbox=None, rout=None, 
            show_2d=False, show_3d=False, vtk=False, render=False):
        """ Initial step in the pipeline: creates an input grid for RADMC3D """

        self.sphfile = sphfile
        self.ncells = ncells
        self.bbox = bbox
        self.rout = rout

        # Register the pipeline step 
        self.steps.append('create_grid')

        # Create a grid instance
        print('')
        utils.print_('Creating the input grid ...\n', bold=True)
        grid = CartesianGrid(
            ncells=self.ncells, 
            bbox=self.bbox, 
            rout=self.rout,
            csubl=self.csubl, 
            nspec=self.nspec, 
            sootline=self.sootline, 
        )

        # Read the SPH data
        grid.read_sph(self.sphfile)

        # Set a bounding box to trim the new grid
        if self.bbox is not None:
            grid.trim_box(bbox=self.bbox * u.au.to(u.cm))

        # Set a radius at which to trim the new grid
        if self.rout is not None:
            grid.trim_box(rout=self.rout * u.au.to(u.cm))

        # Interpolate the SPH points onto a regular cartesian grid
        grid.interpolate_points(field='dens', show_2d=show_2d, show_3d=show_3d)
        grid.interpolate_points(field='temp', show_2d=show_2d, show_3d=show_3d)

        # Write the new cartesian grid to radmc3d file format
        grid.write_grid_file()

        # Write the dust density distribution to radmc3d file format
        grid.write_density_file()
        
        # Write the dust temperature distribution to radmc3d file format
        grid.write_temperature_file()
        
        # Call RADMC3D to read the grid file and generate a VTK representation
        if vtk:
            grid.create_vtk(dust_density=False, dust_temperature=True, rename=True)
        
        # Visualize the VTK grid file using ParaView
        if render:
            grid.render()

    def dust_opacity(self, amin, amax, na, q=-3.5, nang=3, 
            show=False, savefig=None):
        """
            Call dustmixer to generate dust opacity tables
        """
        self.amin = amin
        self.amax = amax
        self.na = na
        self.q = q 
        self.nang = nang
        self.a_dist = np.logspace(np.log10(amin), np.log10(amax), na)
        
        # Create Dust materials
        silicate = Dust(name='Silicate')
        grap_per = Dust(name='Graphite Perpendicular')
        grap_par = Dust(name='Graphite Parallel')

        # Load refractive indices n and k from file. Filenames can also be a url
        silicate.set_nk(path='silicate.nk', meters=True, data_start=2)
        grap_per.set_nk(path='graphite_perpend.nk', meters=True, data_start=2)
        grap_par.set_nk(path='graphite_parallel.nk', meters=True, data_start=2)

        # Set the mass fraction and bulk density of each component
        silicate.set_density(3.50, cgs=True)
        grap_per.set_density(2.25, cgs=True)
        grap_par.set_density(2.25, cgs=True)

        # Convert the refractive indices into dust opacities
        silicate.get_opacities(a=self.a_dist, nang=self.nang)
        grap_per.get_opacities(a=self.a_dist, nang=self.nang)
        grap_par.get_opacities(a=self.a_dist, nang=self.nang)

        # Sum the opacities weighted by their mass fractions
        mixture = (silicate * 0.625) + (grap_per * 0.250) + (grap_par * 0.125)

        # Save the mixture opacity to file
        mixture.plot_opacities(show=show, savefig=savefig)

        # Write the opacity table of the mixture including scattering matrix
        mixture.write_opacity_file(
            scatmat=self.polarization, name=f'sg_a{int(self.amax)}um')
    
    def radmc3d_banner(self):
        utils.print_(f'{"="*21}  <RADMC3D>  {"="*21}', bold=True)

    def generate_input_files(self, inpfile=False, wavelength=False, stars=False, 
            dustopac=False, dustkappa=False):
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

        if wavelength: 
            # Create a wavelength grid in micron
            with open('wavelength_micron.inp', 'w+') as f:
                f.write(f'{self.wavelengths.size}\n')
                for wav in self.wavelengths:
                    f.write(f'{wav:13.6}\n')

        if stars:
            # Create a stellar spectrum file
            with open('stars.inp', 'w+') as f:
                f.write('2\n')
                f.write(f'1 {self.wavelengths.size}\n')
                f.write(f'{self.rstar} {self.mstar} ')
                f.write(f'{self.xstar} {self.ystar} {self.zstar}\n')
                for wav in self.wavelengths:
                    f.write(f'{wav:13.6}\n')
                f.write(f'{-self.tstar}\n')

        if dustopac:
            # Create a dust opacity file
            with open('dustopac.inp', 'w+') as f:
                f.write('2\n')
                f.write(f'{self.nspec}\n')
                f.write('---------\n')
                f.write(f'{self.inputstyle}\n')
                f.write('0\n')
                if self.csubl > 0:
                    f.write(f'{self.dcomp[1]}-a{self.amax}um-{int(self.csubl)}org\n')
                else:
                    f.write(f'{self.dcomp[0]}-a{self.amax}um\n')

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
            
    @utils.elapsed_time
    def monte_carlo(self, nphot):
        """ 
            Call radmc3d to calculate the radiative temperature distribution 
        """

        print('')
        utils.print_("Running a thermal Monte Carlo ...", bold=True)
        self.nphot = nphot
        self.radmc3d_banner()
        subprocess.run(f'radmc3d mctherm nphot {self.nphot}'.split())
        self.radmc3d_banner()

        # Register the pipeline step 
        self.steps.append('monte_carlo')

    @utils.elapsed_time
    def raytrace(self, incl, npix, sizeau, lam=None, show=True, noscat=False, 
            fitsfile='radmc3d_I.fits'):

        """ 
            Call radmc3d to raytrace the newly created grid and plot an image 
        """

        print('')
        utils.print_("Ray-tracing the grid's density and temperature ...\n", 
            bold=True)

        if lam is not None:
            self.lam = lam

        self.scatmode = 0 if noscat else 2

        # Generate only the input files that are not available in the directory
        if not os.path.exists('radmc3d.inp'):
            self.generate_input_files(inpfile=True)

        if not os.path.exists('wavelength_micron.inp'):
            self.generate_input_files(wavelength=True)

        if not os.path.exists('stars.inp'):
            self.generate_input_files(stars=True)

        if not os.path.exists('dustopac.inp'):
            self.generate_input_files(dustopac=True)

        if not os.path.exists(glob('dustkappa*')[0]):
            self.generate_input_files(dustkappa=True)

        # Now double-check all necessary radmc3d input files are available 
        assert os.path.exists('amr_grid.inp')
        assert os.path.exists('dust_density.inp')
        assert os.path.exists('dust_temperature.dat')
        assert os.path.exists('wavelength_micron.inp')
        assert os.path.exists('stars.inp')
        assert os.path.exists('radmc3d.inp')
        assert os.path.exists('dustopac.inp')
        assert os.path.exists(glob('dustkappa*')[0])
         
        # Rotate explicitly by 180
        incl = 180 - int(incl)

        # Set the RADMC3D command
        #cmd = f'radmc3d-notherm image debug_set_thermemistot_to_zero '
        cmd = f'radmc3d image '
        cmd += f'lambda {lam} ' if lam is not None else ' '
        cmd += f'incl {incl} ' if incl is not None else ' '
        cmd += f'npix {npix} ' if npix is not None else ' '
        cmd += f'sizeau {sizeau}' if sizeau is not None else ' '

        # Call RADMC3D
        utils.print_(f'Executing command: {cmd}')
        self.radmc3d_banner()
        subprocess.run(cmd.split())
        self.radmc3d_banner()

        # Remove file if already existent
        if os.path.exists(fitsfile):
            os.remove(fitsfile)

        # Generate a CASA compatible FITS file from the image.out
        utils.radmc3d_casafits(fitsfile)

        # Plot the new image in Jy/pixel
        if show:
            utils.print_('Plotting image.out')
            img = utils.radmc3d_data('image.out', npix=npix, sizeau=sizeau)
            plt.imshow(img, origin='lower', cmap='magma', 
                extent=[-sizeau/2, sizeau/2, -sizeau/2, sizeau/2])
            plt.xlabel('AU')
            plt.ylabel('AU')
            plt.colorbar().set_label('Jy/pixel')
            plt.show()

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
            subprocess.run(f'ds9 radmc3d_I.fits {obs}_I.fits'.split())

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

    def read_sph(self, filename):
        """ Read SPH data """

        utils.print_(f'Reading point coordinates and values from: {filename}')
        if not os.path.exists(filename): 
            raise FileNotFoundError('Input SPH file does not exist')

        self.sph = utils.read_sph(filename, remove_sink=True, cgs=True)
        self.x = self.sph[:, 2]
        self.y = self.sph[:, 3]
        self.z = self.sph[:, 4]
        self.dens = self.sph[:, 10] / self.g2d
        self.temp = self.sph[:, 11]
        self.npoints = len(self.dens)

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
            utils.print_(f'Deleting particles per axis outside a box ' +
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
        self.x_cents = np.linspace(rmin, rmax, self.ncells)
        self.y_cents = np.linspace(rmin, rmax, self.ncells)
        self.z_cents = np.linspace(rmin, rmax, self.ncells)
        X, Y, Z = np.meshgrid(self.x_cents, self.y_cents, self.z_cents)

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
        interp = griddata(xyz, values, (X,Y,Z), 'linear', fill_value=fill)

        # Plot the midplane at z=0 using Matplotlib
        if show_2d:
            from matplotlib.colors import LogNorm
            utils.print_('Plotting the grid midplane at z = 0')
            plt.rcParams['text.usetex'] = True
            plt.rcParams['font.family'] = 'Times New Roman'
            extent = [-self.bbox*u.cm.to(u.au), self.bbox*u.cm.to(u.au)] * 2
            if field == 'dens':
                plt.imshow(interp[:,:,self.ncells//2-1].T * 100, 
                    norm=LogNorm(vmin=1e-15, vmax=2e-10),
                    cmap='BuPu',
                    extent=extent,
                )
            else:
                plt.imshow(interp[:,:,self.ncells//2-1].T, 
                    norm=LogNorm(vmin=80, vmax=None),
                    cmap='inferno',
                    extent=extent,
                )
            plt.colorbar()
            plt.xlabel('AU')
            plt.ylabel('AU')
            plt.title({
                'dens':r'Density Midplane at $z=0$ (g cm$^-3$)', 
                'temp': 'Temperature Midplane at $z=0$ (K)'
            }[field])
            plt.show()
        
        # Render the interpolated 3D field using Mayavi
        if show_3d:
            utils.print_('Visualizing the interpolated field ...')
            from mayavi import mlab
            mlab.contour3d(interp, contours=20, opacity=0.2)
            mlab.show()

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
            dx = np.diff(self.x_cents)[0]
            dy = np.diff(self.y_cents)[0]
            dz = np.diff(self.z_cents)[0]
            x_walls = self.x_cents + dx
            y_walls = self.y_cents + dy
            z_walls = self.z_cents + dz

            # Add the missing wall at the beginning of each axis
            x_walls = np.insert(x_walls, 0, self.x_cents[0])
            y_walls = np.insert(y_walls, 0, self.y_cents[0])
            z_walls = np.insert(z_walls, 0, self.z_cents[0])

            # Write the cell walls
            for i in x_walls:
                f.write(f'{i:13.6e}\n')      
            for j in y_walls:
                f.write(f'{j:13.6e}\n')      
            for k in z_walls:
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
            if dust_density:
                subprocess.run(
                    f'paraview model_dust_density.vtk 2>/dev/null'.split())
            elif dust_temperature:
                subprocess.run(
                    f'paraview model_dust_temperature.vtk 2>/dev/null'.split())



if __name__ == "__main__":

    # Initialize the argument parser
    parser = argparse.ArgumentParser(prog='SphericalGrid',
        description='Pipeline for synthetic observations. It can convert ' + \
        'from sphNG binary files to a RADMC3D regular cartesian grid and ' + \
        'run a raytracing and synthetic ALMA/JVLA observation.')

    # Define command line options
    parser.add_argument('-g', '--grid', action='store_true', default=False, 
        help='Create an input grid for the radiative transfer')

    parser.add_argument('--sphfile', action='store', default='', 
        help='Name of the input SPH file')

    parser.add_argument('--ncells', action='store', type=int, default=100,
        help='Number of cells in every direction')

    exclusive = parser.add_mutually_exclusive_group() 
    exclusive.add_argument('--bbox', action='store', type=float, default=None, 
        help='Size of the side lenght of a bounding box in au (i.e., zoom in)')

    exclusive.add_argument('--rout', action='store', type=float, default=None, 
        help='Size of the outer radial boundary in au (i.e., zoom in)')

    parser.add_argument('--show-grid-2d', action='store_true', default=False,
        help='Plot the midplane of the newly created grid')

    parser.add_argument('--show-grid-3d', action='store_true', default=False,
        help='Render the new cartesian grid in 3D')

    parser.add_argument('--vtk', action='store_true', default=False,
        help='Call RADCM3D to create a VTK file of the newly created grid')

    parser.add_argument('--render', action='store_true', default=False,
        help='Visualize the VTK file using ParaView')

    parser.add_argument('--opacity', action='store_true', default=False,
        help='Call dustmixer to generate a dust opacity table')

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

    parser.add_argument('-mc', '--monte-carlo', action='store_true', default=False,
        help='Call RADMC3D to raytrace the new grid and plot an image')

    parser.add_argument('--nphot', action='store', type=float, default=1e7,
        help='Set the number of photons for the Monte Carlo temperature calculation')

    parser.add_argument('--nthreads', action='store', default=4, 
        help='Number of threads used for the Monte-Carlo runs')

    parser.add_argument('-rt', '--raytrace', action='store_true', default=False,
        help='Call RADMC3D to raytrace the new grid and plot an image')

    parser.add_argument('--lam', action='store', type=float, default=3000,
        help='Wavelength used to generate an image in units of micron')

    parser.add_argument('--npix', action='store', type=int, default=300,
        help='Number of pixels per side of new image')

    parser.add_argument('--incl', action='store', type=float, default=0,
        help='Inclination angle of the grid in degrees')

    parser.add_argument('--sizeau', action='store', type=int, default=100,
        help='Physical size of the image in AU')

    parser.add_argument('--polarization', action='store_true', default=False,
        help='Enable polarized RT and full scattering matrix opacity tables')

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

    parser.add_argument('--synobs', action='store_true', default=False,
        help='Call CASA to run a synthetic observation from the new image')

    parser.add_argument('--show-synobs', action='store_true', default=False,
        help='Plot the ALMA/JVLA synthetic image generated by CASA')


    # Store the command-line given arguments
    cli = parser.parse_args()

    # Initialize the pipeline
    pipeline = Pipeline(lam=cli.lam, amax=cli.amax, nphot=cli.nphot, 
        nthreads=cli.nthreads, csubl=cli.sublimation, sootline=cli.soot_line, 
        dgrowth=cli.dust_growth, polarization=cli.polarization)

    # Generate the input grid for RADMC3D
    if cli.grid:
        pipeline.create_grid(sphfile=cli.sphfile, ncells=cli.ncells, 
            bbox=cli.bbox, rout=cli.rout, show_2d=cli.show_grid_2d, 
            show_3d=cli.show_grid_3d, vtk=cli.vtk, render=cli.render)

    # Generate the dust opacity tables
    if cli.opacity:
        pipeline.dust_opacity(cli.amin, cli.amax, cli.na, cli.q, cli.ang)

    # Run a thermal Monte-Carlo
    if cli.monte_carlo:
        pipeline.monte_carlo(nphot=cli.nphot)

    # Run a ray-tracing on the new grid and generate an image
    if cli.raytrace:
        pipeline.raytrace(lam=cli.lam, incl=cli.incl, npix=cli.npix, 
            sizeau=cli.sizeau, show=cli.show_rt, noscat=cli.noscat)

    # Run a synthetic observation of the new image by calling the CASA simulator
    if cli.synobs:
        pipeline.synthetic_observation(show=cli.show_synobs, lam=cli.lam, 
            graphic=False, verbose=False)


