import os
import utils
import argparse
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from sph_to_sphergrid import sph_to_sphergrid


class SphericalGrid:
    def __init__(self, nr, nth, nph, theta=np.pi/2, rin=1.5e13, rout=1.5e15):
        """ 
        Create a spherical grid from a set of 3D point.

        The input SPH particle coordinates should be given in 
        cartesian coordinates.
        """
        self.rin = rin
        self.rout = rout
        self.nr = nr
        self.nth = nth
        self.nph = nph
        self.theta = theta

    def read_sph(self, filename='snap_541.dat'):
        """ Read SPH data """
        self.sph = utils.read_sph(filename, remove_sink=True, cgs=True)
        self.xyz = self.sph[:, 2:5]
        self.mass = self.sph[:, 8] / 100
        self.h = self.sph[:, 9][::-1]
        self.rho = self.sph[:, 10] / 100
        self.temp = self.sph[:, 11]
        self.npoints = len(self.rho)

    def find_rout(self):
        """ Set r_out as the position of the farthest particle from the center """
        sx = utils.stats(self.xyz[:, 0], verbose=False)
        sy = utils.stats(self.xyz[:, 1], verbose=False)
        outterx = np.array([np.abs(sx['min']), np.abs(sx['max'])]).max()
        outtery = np.array([np.abs(sy['min']), np.abs(sy['max'])]).max()
        self.rout = np.max([outterx, outtery])
        # Add a small extension to cover a few missing particles
        self.rout += 230*u.au.to(u.cm)

    @utils.elapsed_time
    def map_to_spherical(self):
        """ Conver the SPH data points into a spherical grid """
        self.grid = sph_to_sphergrid(
            xyz = self.xyz,
            masses = self.mass, 
            hrkernel = self.h, 
            rin=self.rin, 
            rout=self.rout, 
            nr=self.nr, 
            ntheta=self.nth, 
            nphi=self.nph,
            hrup=self.theta,
            temp=self.temp, 
        )
        # Store the grid parameters
        self.r_i = self.grid.grid_ri
        self.th_i = self.grid.grid_thetai
        self.ph_i = self.grid.grid_phii
        self.nr = self.grid.grid_nr
        self.nth = self.grid.grid_ntheta
        self.nph = self.grid.grid_nphi
        self.sph_excl = self.grid.sph_excluded

        # Warn if certain particles were not mapped into the grid
        if len(self.sph_excl) > 0:
            utils.print_(f'{len(self.sph_excl)} particles lie outside the grid',
                True, fail=True)
            with open('particles_non_converted.txt', 'w+') as f:
                print(f'{[i for i in self.sph_excl]}', file=f)

    def write_grid_file(self):
        """ Write the spherical grid file """
        with open('amr_grid.inp','w+') as f:
            # iformat
            f.write('1\n')                     
            # Regular grid
            f.write('0\n')                      
            # Coordinate system: spherical
            f.write('100\n')                     
            # Gridinfo
            f.write('0\n')                       
            # Include r, theta, phi coordinates
            f.write('1 1 1\n')                   
            # Size of the grid
            f.write('%d %d %d\n'%(self.nr, self.nth, self.nph))  
            # Cell walls
            for value in self.r_i:
                f.write('%13.6e\n'%(value))      
            for value in self.th_i:
                f.write('%17.10e\n'%(value))     
            for value in self.ph_i:
                f.write('%13.6e\n'%(value))     

    def write_density_file(self):
        """ Write the density file """
        with open('dust_density.inp','w+') as f:
            # Format number
            f.write('1\n')                      
            # Number of cells
            f.write('%d\n'%(self.nr * self.nth * self.nph))     
            # Number of dust species
            f.write('1\n')                
            # Flatten the array into a 1D fortran-style indexing
            data = self.grid.rho.ravel(order='F')   
            data.tofile(f, sep='\n', format="%13.6e")
            f.write('\n')

    def write_temperature_file(self):
        """ Write the temperature file """
        with open('dust_temperature.dat','w+') as f:
            # Format number
            f.write('1\n')                      
            # Number of cells
            f.write('%d\n'%(self.nr * self.nth * self.nph))     
            # Number of dust species
            f.write('1\n')                
            # Flatten the array into a 1D fortran-style indexing
            data = self.grid.celltemp.ravel(order='F')   
            data.tofile(f, sep='\n', format="%13.6e")
            f.write('\n')

    def create_vtk(self, dust_density=False, dust_temperature=True):
        """ Call radmc3d to create a VTK file of the grid """
        self.radmc3d_banner()
        if dust_density:
            os.system('radmc3d vtk_dust_density 1')
        elif dust_temperature:
            os.system('radmc3d vtk_dust_temperature 1')
        else:
            os.system('radmc3d vtk_grid')
        self.radmc3d_banner()

    def render(self, state=None):
        """ Render the new grid in 3D using ParaView """
        if isinstance(state, str):
            os.system(f'paraview --state {state}')
        else:
            os.system(f'paraview model.vtk')

    def radmc3d_banner(self):
        utils.print_(f'{"="*21}  <RADMC3D>  {"="*21}', verbose=True, bold=True)


class Pipeline:
    def __init__(self, lam):
        self.lam = lam
    
    @utils.elapsed_time
    def raytrace(self, incl, npix, sizeau, show=True):
        """ 
            Call radmc3d to raytrace the newly created grid and plot an image 
        """
        self.radmc3d_banner()
        os.system(
        f'radmc3d image lambda {self.lam} incl {incl} npix {npix} sizeau {sizeau}')
        self.radmc3d_banner()

        if show:
            img = utils.radmc3d_data('image.out', npix=npix, sizeau=sizeau)
            plt.imshow(img, extent=[-sizeau/2, sizeau/2, -sizeau/2, sizeau/2])
            plt.xlabel('AU')
            plt.ylabel('AU')
            plt.colorbar().set_label('Jy/pixel')
            plt.show()

    def synthetic_observation(self, show=False):
        """ 
            Prepare the input for the CASA simulator from the RADMC3D output,
            and call CASA to run a synthetic observation.
        """
        
        utils.radmc3d_casafits(fitsfile='radmc3d_I.fits')
        os.system('casa -c alma_simulation.py')

        if show:
            os.system('casaviewer alma_I.fits')



if __name__ == "__main__":

    # Initialize the argument parser
    parser = argparse.ArgumentParser(prog='SphericalGrid',
        description='Convert from sphNG files into a RADMC3D spherical grid')

    # Define command line options
    parser.add_argument('--sphfile', action='store', default='snap_541.dat', 
        help='Name of the input SPH file.')
    parser.add_argument('-rin', action='store', type=float, default=1, 
        help='Size of the inner radial boundary in au.')
    parser.add_argument('-rout', action='store', type=float, default=100, 
        help='Size of the outer radial boundary in au.')
    parser.add_argument('-nr', action='store', type=int, default=100,
        help='Number of cells in the radial direction.')
    parser.add_argument('-nph', action='store', type=int, default=50,
        help='Number of cells in the azimuthal direction.')
    parser.add_argument('-nth', action='store', type=int, default=50,
        help='Number of cells in elevation.')
    parser.add_argument('--find-rout', action='store_true', default=False,
        help='Set the outter radius as the position of the farthest particle.')
    parser.add_argument('--vtk', action='store_true', default=False,
        help='Call RADCM3D to create a VTK file of the newly created grid.')
    parser.add_argument('--render', action='store_true', default=False,
        help='Visualize the VTK file using ParaView.')
    parser.add_argument('--raytrace', action='store_true', default=False,
        help='Call RADMC3D to raytrace the new grid and plot an image')
    parser.add_argument('--synobs', action='store_true', default=False,
        help='Call CASA to run a synthetic observation from the new image')

    # Store the command-line given arguments
    cli = parser.parse_args()


    # Initialize the pipeline
    pipeline = Pipeline(lam=3000)
    
    # Create a grid instance
    grid = SphericalGrid(nr=cli.nr, nph=cli.nph, nth=cli.nth, theta=np.pi/2)

    # Read the SPH data
    grid.read_sph(cli.sphfile)

    # Set the grid inner boundary
    grid.rin = cli.rin * u.au.to(u.cm)

    # Set the grid outter boundary
    grid.rout = cli.rout * u.au.to(u.cm)

    # Reset the grid outter radius based on the cartesian particle positions
    if cli.find_rout:
        grid.find_rout()

    # Map the SPH points into the spherical grid
    grid.map_to_spherical()

    # Write the new spherical grid to radmc3d file format
    grid.write_grid_file()

    # Write the dust density distribution to radmc3d file format
    grid.write_density_file()
    
    # Write the dust temperature distribution to radmc3d file format
    grid.write_temperature_file()
    
    # Call RADMC3D to read the grid file and generate a VTK representation
    if cli.vtk:
        grid.create_vtk()
    
    # Visualize the VTK grid file using ParaView
    if cli.render:
        grid.render()

    # Call RADMC3D to raytrace the newly created grid and plot an image
    if cli.raytrace:
        pipeline.raytrace(incl=0, npix=100, sizeau=2*grid.rout*u.cm.to(u.au))

    # Call CASA to run a synthetic observation from the new image
    if cli.synobs:
        pipeline.synthetic_observation(show=True)

