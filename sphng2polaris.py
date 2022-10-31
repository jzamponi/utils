import sys
import time
import math
import struct
import bisect
import random
import argparse
import numpy as np
from tqdm import tqdm
import astropy.units as u
import astropy.constants as c
from scipy.spatial import Delaunay
from collections import defaultdict
from functools import wraps


class Converter():
    def __init__(self, input_file, output_file, verbose=False):
        self.sphfile = input_file
        self.output = output_file
        self.format = None
        self.verbose = verbose
        self.data = {}

    def print_(self, string):
        """Customized print function to work only when verbose is enabled."""
        if self.verbose:
            print(f'[{self.__class__.__name__}] {string}')

    def add_temp(self, add):
        self.add_temp = add

    def add_bfield(self, add):
        self.add_bfield = add

    def add_dust_pop(self, add):
        self.add_dust_pop = add

    def create_grid(self, format):

        # Measure time before it runs
        start = time.time()

        # Start creating the grid
        self.print_(f'Creating grid: {self.output}')

        # Polaris grid ID (50 = Voronoi)
        grid_id = 50

        # Polaris ID for every physical quantity
        data_ids = [28, 29, 2, 3, 4, 5, 6, 7, 8, 9, 21] 

        # Remove gas and dust temperature from the ID list if not required
        if not self.add_temp:
            data_ids.remove(2)
            data_ids.remove(3)

        # Remove B field components from the ID list if not required
        if not self.add_bfield:
            data_ids.remove(4)
            data_ids.remove(5)
            data_ids.remove(6)
        
        # Remove dust population indices from the ID list if not required
        if not self.add_dust_pop:
            data_ids.remove(21)

        # Read the data from snapshot
        self.print_(f"Loading SPH file {self.sphfile}")
        d = self.read_sph(self.sphfile)
        n_gas = d[:,0].size

        self.print_(f'Gas particles: {n_gas}')
        self.print_(f'Time = {d[0,1]} years')

        self.print_('Reading IDs')
        self.data['id'] = d[:,0]

        self.print_('Reading positions')
        self.data['pos'] = d[:,2:5]

        self.print_('Reading velocities')
        self.data['vel'] = d[:,5:8]

        self.print_('Reading masses')
        self.data['mass'] = d[:,8]

        self.print_('Reading densities')
        self.data['rho'] = d[:,10]

        self.print_('Reading gas temperatures')
        self.data['tgas'] = d[:,11]

        points = []

        # Reduce the fraction number to reduce the total number of cells,
        # note that the cell volume may have an error in this case
        fraction = 1.0 
        rel_volume = 1.0/fraction

        sample_set = random.sample(range(n_gas), int(fraction*n_gas))

        points = self.data['pos'][sample_set]

        nr_of_points = len(points)

        self.print_(f'Number of points: {nr_of_points}')

        self.print_("creating delaunay")

        tri = Delaunay(points)

        self.print_("finding all neigbors")

        neighbors = self.find_neighbors(tri)

        self.print_("creating convex hull")
        convex_hull = self.create_convex_hull_list(tri.convex_hull)
        tri.close()

        self.print_(f'length of convex hull: {len(convex_hull)}')
        self.print_('Delaunay: done')

        # Number of quantities to write
        data_len = len(data_ids)

        # Define the length of the box
        l_max =  4602.56*u.au.to(u.cm) * 2

        self.print_("Writing data to the grid ...")

        # Define output format
        self.format = format
        if self.format == 'binary':
            self.write_mode = "wb"
            self.ext = ".dat"

        elif self.format == 'ascii':
            self.write_mode = "w"
            self.ext = ".txt"

        # Add extension to filename if missing
        if not self.output.endswith(self.ext):
            self.output += self.ext

        # Write the header
        file = open(self.output, self.write_mode)

        self.write_data(file, "H", grid_id, endl=True)
        self.write_data(file, "H", data_len, endl=True)

        [self.write_data(file, "H", i, endl=True) for i in data_ids]

        self.write_data(file, "d", nr_of_points, endl=True)
        self.write_data(file, "d", l_max, endl=True)

        pos_counter = 0

        # Suppress the values of the outlier cell. It is probably a sink particle.
        #i_outlier = 31330
        #d[i_outlier,5] = 0
        #d[i_outlier,6] = 0
        #d[i_outlier,7] = 0
        #d[i_outlier,8] = d[:,8].min()
        #d[i_outlier,9] = d[:,9].min()
        #d[i_outlier,10] = 3e-11 
        #d[i_outlier,11] = 900

        # WRITE DATA 
        for i in tqdm(sample_set):
            # Spatial coordinates
            x = d[i,2]*u.au.to(u.cm)
            y = d[i,3]*u.au.to(u.cm)
            z = d[i,4]*u.au.to(u.cm)

            # Mass of every particle
            mass = d[i,8]*u.M_sun.to(u.g)

            # Gas density of every particle
            rho_g = d[i,10]

            # Dust density of every particle
            rho_d = 0.01 * rho_g

            if i == len(sample_set)-1:
                x_i = d[i-1,2]*u.au.to(u.cm)
                y_i = d[i-1,3]*u.au.to(u.cm)
                z_i = d[i-1,4]*u.au.to(u.cm)
            else:
                x_i = d[i+1,2]*u.au.to(u.cm)
                y_i = d[i+1,3]*u.au.to(u.cm)
                z_i = d[i+1,4]*u.au.to(u.cm)
            
            # Radius of a sphere
            r = np.sqrt((x-x_i)**2 + (y-y_i)**2 + (z-z_i)**2)

            # Volume of a sphere of radius equal to the smoothing length
            r = d[i,9]*u.au.to(u.cm)

            # Volume of sphere of radius equal to the distance to the neighbour
            vol = (4/3) * np.pi * r**3

            # Override the volume as simply the mass over density of the cell
            vol = (mass / rho_g) / fraction
            
            # Assign the physical quantities
            Tg = d[i,11]
            Td = Tg
            vx = d[i,5]
            vy = d[i,6]
            vz = d[i,7]
            
            # Write the coordinates
            self.write_data(file, "f", x)
            self.write_data(file, "f", y)
            self.write_data(file, "f", z)
            
            # Write the volume 
            self.write_data(file, "d", vol)
            
            # Write the dust densities
            # Rescale the dust-to-gas ratio when two dust populations are defined
            if self.add_dust_pop:
                # Criterion used to define the two populations
                criterion = Td > 300

                # Factor to downscale the dust mass within the sublimation zone
                scale_factor = 0.8125 # mix5
                scale_factor = 0.7755 # mix6
                scale_factor = 100
                rho_g = scale_factor * rho_g if criterion else rho_g
                rho_d = scale_factor * rho_d if criterion else rho_d

            # Write the gas and dust densities
            self.write_data(file, "f", rho_g)
            self.write_data(file, "f", rho_d)

            # Write the gas and dust temperatures
            if self.add_temp:
                self.write_data(file, "f", Td)
                self.write_data(file, "f", Tg)
            
            # Write the components of the magnetic field
            if self.add_bfield:
                # Create a simple fully poloidal field of 42.5uG
                b_intensity = 42.5e-6
                bx = np.zeros(vx.shape)
                by = np.zeros(vy.shape)
                bz = np.full(vz.shape, b_intensity)
                self.write_data(file, "f", vx*1e-4)
                self.write_data(file, "f", vy*1e-4)
                self.write_data(file, "f", vz*1e-4)

            # Write the components of the velocity field
            self.write_data(file, "f", vx)
            self.write_data(file, "f", vy)
            self.write_data(file, "f", vz)

            # Write 2 different dust populations
            # Based on a temperature threshold (i.e., dust sublimation)
            if self.add_dust_pop:
                pop = np.ones(Td.shape)
                pop[criterion] = 2
                self.write_data(file, "f", pop)

            p_list = list(neighbors[pos_counter])
            sign = int(self.is_in_hull(convex_hull, pos_counter))
            neigbors = int(len(p_list))
            nr_of_neigbors = sign * neigbors

            self.write_data(file, "i", int(nr_of_neigbors))

            for j in range(0, int(abs(nr_of_neigbors))):
                tmp_n=int(p_list[j])
                self.write_data(file, "i", tmp_n)

            # Add a line break after every cell
            if self.format == 'ascii': 
                self.write_data(file, 'H', "", endl=True)

            # Update counter of the loop 
            pos_counter+=1

        # Measure time difference after it finishes
        run_time = time.time() - start

        # Print the elapsed time nicely formatted
        self.print_(
            f'Elapsed time: {time.strftime("%H:%M:%S", time.gmtime(run_time))}'
        )

    def is_in_hull(self, a, x):
        i = bisect.bisect_left(a, x)
        if i != len(a) and a[i] == x:
            return -1

        return 1

    def find_neighbors(self, tri):
        neighbors = defaultdict(set)

        tri_len=len(tri.simplices)
        counter = 0

        for simplex in tri.simplices:
            if counter % 100000 == 0:
                sys.stdout.write('Determining neighbors: ' + str(100.0 * counter / tri_len) + '%    \r')
                sys.stdout.flush()

            counter+=1

            for idx in simplex:
                other = set(simplex)
                other.remove(idx)
                neighbors[idx] = neighbors[idx].union(other)

        return neighbors

    def create_convex_hull_list(self, hull):
        neighbors = defaultdict(set)

        for simplex in hull:
            for idx in simplex:
                other = set(simplex)
                other.remove(idx)
                neighbors[0] = neighbors[0].union(other)

        return sorted(list(neighbors[0]))

    def write_data(self, file_, dtype, var, endl=False):
        if self.format == 'binary':
            file_.write(struct.pack(dtype, var))

        elif self.format == 'ascii':
            if not endl:
                file_.write(str(var)+' ')
            else:
                file_.write(str(var)+'\n')

    def read_sph(self, snapshot="snap_541.dat", write_hdf5=False):
        """
        Assumes your binary file is formatted with a header stating the quantities,
        assumed to be f4 floats. May not be widely applicable.

        Header goes as: id t x y z vx vy vz mass hsml rho T u

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
        import h5py

        # Read file in binary format
        with open(snapshot, "rb") as f:
            names = f.readline()[1:].split()
            #data = np.fromstring(f.read()).reshape(-1, len(names))
            data = np.frombuffer(f.read()).reshape(-1, len(names))
            data = data.astype("f4")


        if write_hdf5:
            # Follow the GADGET/GIZMO convention
            print_(f'Creating HDF5 file: {snapshot.replace("dat","h5")}')
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



if __name__ == "__main__":

    # Initialize the argument parser
    parser = argparse.ArgumentParser(prog='sphng2polaris',
            description='Convert from SPHng dumpfiles into a POLARIS Voronoi grid')

    # Define command line options
    parser.add_argument('input', action='store', \
            help='Name of the output grid. It will suffixed by the frame number')

    parser.add_argument('output', action='store', \
            help='Name of the output grid. It will suffixed by the frame number')

    parser.add_argument('-f', '--format', action='store', choices=['binary','ascii'], default='binary', \
            help='Format of the output grid. POLARIS uses binary but ascii is useful to check the grid.')

    parser.add_argument('-t', '--add-temp', action='store_true', \
            help='Boolean switch to decide whether to add temperature in the polaris grid or not')

    parser.add_argument('-b', '--add-bfield', action='store_true', \
            help='Boolean switch to decide whether to add a B field in the polaris grid or not')

    parser.add_argument('-d', '--dust_pop', action='store_true', \
            help='Boolean switch to decide whether to define 2 dust populations in the polaris grid or not')

    parser.add_argument('-v', '--verbose', action='store_true')

    cli = parser.parse_args()

    # Create a converter instance
    grid = Converter(cli.input, cli.output, cli.verbose)

    # Set the boolean switches
    grid.add_temp(cli.add_temp)
    grid.add_bfield(cli.add_bfield)
    grid.add_dust_pop(cli.dust_pop)

    # Create and populate the grid
    grid.create_grid(cli.format)
