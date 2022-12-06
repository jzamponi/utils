#!/usr/bin/env python3
"""
    Pipeline script to calculate synthetic ALMA images from an SPH model,
    using RADMC3D for the raytracing and CASA for the ALMA simulation. 

    Example:

    $ python3 sph2cartesian-radmc3d.py --sphfile snap_001.dat --ncells 100 
        --bbox 50 --show --vtk --render --raytrace --synobs

    For details, run:
    $ python3 sph2cartesian-radmc3d.py --help

"""

import os
import utils
import argparse
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from tqdm import tqdm


class Pipeline:
    def __init__(self, lam=1300):
        self.lam = lam
        self.steps = []

    def create_grid(self, ncells=None, bbox=None, rout=None):
        """ Initial step in the pipeline: creates an input grid for RADMC3D """

        # Register the pipeline step 
        self.steps.append('create_grid')

        # Create and return a grid instance
        self.grid = CartesianGrid(ncells=ncells, bbox=bbox, rout=rout)
        return self.grid
    
    def radmc3d_banner(self):
        utils.print_(f'{"="*21}  <RADMC3D>  {"="*21}', bold=True)

    def monte_carlo(self, nphot):
        """ 
            Call radmc3d to calculate the radiative temperature distribution 
        """

        self.radmc3d_banner()
        os.system(f'radmc3d mctherm nphot {nphot}')
        self.radmc3d_banner()

        # Register the pipeline step 
        self.steps.append('monte_carlo')


    @utils.elapsed_time
    def raytrace(self, incl, npix, sizeau, lam=None, show=True, fitsfile='radmc3d_I.fits'):

        """ 
            Call radmc3d to raytrace the newly created grid and plot an image 
        """
        if lam is not None:
            self.lam = lam

        # Make sure all necessary radmc3d input files are available 
        # in the current directory
        assert os.path.exists('amr_grid.inp')
        assert os.path.exists('dust_density.inp')
        assert os.path.exists('dust_temperature.dat')
        assert os.path.exists('dustopac.inp')
        assert os.path.exists('wavelength_micron.inp')
        assert os.path.exists('stars.inp')
        assert os.path.exists('radmc3d.inp')
         
        self.radmc3d_banner()

        # Rotate implicitly by 180
        incl = 180 - int(incl)

        # Stardard RADMC3D command
        cmd = f'radmc3d image '
        cmd += f'lambda {lam} ' if lam is not None else ' '
        cmd += f'incl {incl} ' if incl is not None else ' '
        cmd += f'npix {npix} ' if npix is not None else ' '
        cmd += f'sizeau {sizeau}' if sizeau is not None else ' '

        utils.print_(f'Executing command: {cmd}')
        os.system(cmd)

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


    def synthetic_observation(self, lam=None, show=False):
        """ 
            Prepare the input for the CASA simulator from the RADMC3D output,
            and call CASA to run a synthetic observation.
        """
        import requests
        from pathlib import Path

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
        if lam in ['1.3mm', '3mm']:
            script = 'alma_simulation.py'  
        else:
            script = 'vla_simulation.py'  

        if not os.path.exists(script):
            # Download the CASA script
            url = 'https://raw.githubusercontent.com/jzamponi/utils/main/' + \
                f'synthetic_observations/{lam}/{script}'

            utils.print_(f'No CASA script found. Downloading from: ', bold=True)
            utils.print_(f'{url}')

            download = Path(script).write_bytes(requests.get(url).content)

            # Tailor the script
            os.system(f"sed --in-place s/polaris/radmc3d/g {script}") 
            if lam == '3mm':
                os.system(f"sed --in-place '55,61d' {script}")

        # Run the ALMA/JVLA simulation script
        os.system(f'casa -c {script}')

        # Register the pipeline step 
        self.steps.append('synobs')



class CartesianGrid(Pipeline):
    def __init__(self, ncells, bbox=None, rout=None):
        """ 
        Create a cartesian grid from a set of 3D point.

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
        self.fill = 0

    def read_sph(self, filename):
        """ Read SPH data """
        utils.print_(f'Reading point coordinates and values from: {filename}')
        self.sph = utils.read_sph(filename, remove_sink=True, cgs=True)
        self.x = self.sph[:, 2]
        self.y = self.sph[:, 3]
        self.z = self.sph[:, 4]
        self.mass = self.sph[:, 8] / self.g2d
        self.h = self.sph[:, 9][::-1]
        self.dens = self.sph[:, 10] / self.g2d
        self.temp = self.sph[:, 11]
        self.npoints = len(self.dens)

    @utils.elapsed_time
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
            for i in tqdm(range(self.x.size)):
                if self.x[i] < x1 or self.x[i] > x2:
                    to_remove.append(i)

            for j in tqdm(range(self.y.size)):
                if self.y[j] < y1 or self.y[j] > y2:
                    to_remove.append(j)

            for k in tqdm(range(self.z.size)):
                if self.z[k] < z1 or self.z[k] > z2:
                    to_remove.append(k)

        
        if rout is not None and bbox is None:
            # Override any previous value of rout
            self.rout = rout
            utils.print_('Deleting particles outside a radius of ' +
                f'{self.rout * u.cm.to(u.au)} au ...')

            # Convert cartesian to polar coordinates to define a radial trim
            r = np.sqrt(self.x**2 + self.y**2 + self.z**2)

            for i in tqdm(range(self.x.size)):
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

    @utils.elapsed_time
    def interpolate_points(self, field='temp', show=False):
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
        xyz = np.vstack([self.x, self.y, self.z]).T
        interp = griddata(xyz, values, (X,Y,Z), 'linear', fill_value=self.fill)

        # Render the interpolated 3D field using Mayavi
        if show:
            from mayavi import mlab
            utils.print_('Visualizing the interpolated field ...')
            mlab.contour3d(interp, contours=20, opacity=0.2)
            plt.imshow(interp[:,:,self.ncells//2-1])
            plt.colorbar()
            plt.title({
                'dens':r'Density Midplane at $z=0$ (g cm$^-3$)', 
                'temp': 'Temperature Midplane at $z=0$ (K)'
            }[field])
            plt.show()
        
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
            # Format number
            f.write('1\n')                      
            # Number of cells
            f.write(f'{density.size:d}\n')
            # Number of dust species
            f.write('1\n')                
            # Write the cell values
            for d in density:
                f.write(f'{d:13.6e}\n')

    def write_temperature_file(self):
        """ Write the temperature file """
        utils.print_('Writing dust temperature file')
        
        # Flatten the array into a 1D fortran-style indexing
        temperature = self.interp_temp.ravel(order='F')
        with open('dust_temperature.dat','w+') as f:
            # Format number
            f.write('1\n')                      
            # Number of cells
            f.write(f'{temperature.size:d}\n')
            # Number of dust species
            f.write('1\n')                
            # Write the cell values
            for t in temperature:
                f.write(f'{t:13.6e}\n')

    def create_vtk(self, dust_density=False, dust_temperature=True, rename=False):
        """ Call radmc3d to create a VTK file of the grid """
        self.radmc3d_banner()

        if dust_density:
            os.system('radmc3d vtk_dust_density 1')
            if rename:
                os.system('mv model.vtk model_dust_density.vtk')

        if dust_temperature:
            os.system('radmc3d vtk_dust_temperature 1')
            if rename:
                os.system('mv model.vtk model_dust_temperature.vtk')

        if not dust_density and not dust_temperature:
            os.system('radmc3d vtk_grid')

        self.radmc3d_banner()

    def render(self, state=None, dust_density=False, dust_temperature=True):
        """ Render the new grid in 3D using ParaView """
        if isinstance(state, str):
            os.system(f'paraview --state {state} 2>/dev/null')
        else:
            if dust_density:
                os.system(f'paraview model_dust_density.vtk 2>/dev/null')
            elif dust_temperature:
                os.system(f'paraview model_dust_temperature.vtk 2>/dev/null')




if __name__ == "__main__":

    # Initialize the argument parser
    parser = argparse.ArgumentParser(prog='SphericalGrid',
        description='Convert from sphNG files into a RADMC3D regular cartesian grid')

    # Define command line options
    parser.add_argument('--sphfile', action='store', default=None, 
        help='Name of the input SPH file.')

    parser.add_argument('-n', '--ncells', action='store', type=int, default=100,
        help='Number of cells in every direction.')

    exclusive = parser.add_mutually_exclusive_group() 
    exclusive.add_argument('--bbox', action='store', type=float, default=None, 
        help='Size of the side lenght of a bounding box in au (i.e., zoom in)')

    exclusive.add_argument('--rout', action='store', type=float, default=None, 
        help='Size of the outer radial boundary in au (i.e., zoom in).')

    parser.add_argument('--show_grid', action='store_true', default=False,
        help='Plot the midplane of the newly created grid.')

    parser.add_argument('--vtk', action='store_true', default=False,
        help='Call RADCM3D to create a VTK file of the newly created grid.')

    parser.add_argument('--render', action='store_true', default=False,
        help='Visualize the VTK file using ParaView.')

    parser.add_argument('-mc', '--monte-carlo', action='store_true', default=False,
        help='Call RADMC3D to raytrace the new grid and plot an image')

    parser.add_argument('--nphot', action='store', type=float, default=1e7,
        help='Set the number of photons for the Monte Carlo temperature calculation.')

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

    parser.add_argument('--show_rt', action='store_true', default=False,
        help='Plot the intensity map generated by ray-tracing.')

    parser.add_argument('--synobs', action='store_true', default=False,
        help='Call CASA to run a synthetic observation from the new image')

    parser.add_argument('--show_synobs', action='store_true', default=False,
        help='Plot the ALMA/JVLA synthetic image generated by CASA')


    # Store the command-line given arguments
    cli = parser.parse_args()


    # Initialize the pipeline
    pipeline = Pipeline()
    
    # Generate the input grid for RADMC3D
    if cli.sphfile is not None:
        
        # Create a grid instance
        grid = pipeline.create_grid(ncells=cli.ncells)

        # Read the SPH data
        grid.read_sph(cli.sphfile)

        # Set a bounding box to trim the new grid
        if cli.bbox is not None:
            grid.trim_box(bbox=cli.bbox * u.au.to(u.cm))

        # Set a radius at which to trim the new grid
        if cli.rout is not None:
            grid.trim_box(rout=cli.rout * u.au.to(u.cm))

        # Interpolate the SPH points onto a regular cartesian grid
        grid.interpolate_points(field='dens', show=cli.show_grid)
        grid.interpolate_points(field='temp', show=cli.show_grid)

        # Write the new cartesian grid to radmc3d file format
        grid.write_grid_file()

        # Write the dust density distribution to radmc3d file format
        grid.write_density_file()
        
        # Write the dust temperature distribution to radmc3d file format
        grid.write_temperature_file()
        
        # Call RADMC3D to read the grid file and generate a VTK representation
        if cli.vtk:
            grid.create_vtk(dust_density=False, dust_temperature=True, rename=True)
        
        # Visualize the VTK grid file using ParaView
        if cli.render:
            grid.render()

    # Run a thermal Monte-Carlo
    if cli.monte_carlo:
        pipeline.monte_carlo(nphot=cli.nphot)

    # Run a ray-tracing on the new grid and generate an image
    if cli.raytrace:
        pipeline.raytrace(lam=cli.lam, incl=cli.incl, npix=cli.npix, 
            sizeau=cli.sizeau, show=cli.show_rt)

    # Run a synthetic observation of the new image by calling the CASA simulator
    if cli.synobs:
        pipeline.synthetic_observation(show=cli.show_synobs, lam=cli.lam)


