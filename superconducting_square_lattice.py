import os
#os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.99'
#os.environ["OPENBLAS_NUM_THREADS"] = "32"
#os.environ["MKL_NUM_THREADS"] = "32"
#os.environ["BLIS_NUM_THREADS"] = "32"
import jax
jax.config.update("jax_enable_x64", True)
from jax import jit
import numpy as np
import jax.numpy as jnp
from jax.lib import xla_bridge
from functools import partial
import logging
from jaxlib.xla_extension import XlaRuntimeError
import time
import gc
from numpy import linalg as LA
#from jax.profiler import start_trace, stop_trace

from jax.lax import dynamic_slice
from jax.sharding import Mesh, PositionalSharding, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map

#jax.config.update('jax_platform_name', 'cpu')
print(xla_bridge.get_backend().platform, jax.devices(),jax.devices('cpu'))
#first_gpu = jax.devices('gpu')[0]
#second_gpu = jax.devices('gpu')[1]
mesh=Mesh(np.array(jax.devices()),('devices',))
spec = P('devices')
sharding = NamedSharding(mesh,spec)
logging.basicConfig(filename='/home/susva433/Workstation_Code/superconducting_square_lattice_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info('----------------------------------------------------------------------------------------')

# Writing the Hamiltonian matrix as per lattice structure and index will be useful to see the horizontal and vertical interaction placements 
#e.g. [11] lattice for N_diag = 4, N = 6 where N_repeat = N - N_diag = 2, represents the number of repeating/extra horizontal and vertical chains besides the isosceles right triangle. 
# 1 -- 7  -- 13
# |    |     |
# 2 -- 8  -- 14 -- 19 
# |    |     |
# 3 -- 9  -- 15 -- 20 -- 24                         ^ y [01]
# |    |     |     |     |                          |
# 4 -- 10 -- 16 -- 21 -- 25 -- 28                    --> x [10]
# |    |     |     |     |     |
# 5 -- 11 -- 17 -- 22 -- 26 -- 29                   13-16-28-13 is the isosceles right triangle with N_diag = 4
# |    |     |     |     |     |                    13-19-24-28 is the [11] edge with 4 sites (in the diagonal)
# 6 -- 12 -- 18 -- 23 -- 27 -- 30                   The code returns a normal square lattice with N_diag = 0 or 1

#Jitted functions --> To be used only on inputs that do not vary in size
kron_jit = jit(jnp.kron)
conjugate_jit = jit(jnp.conjugate)
tanh_jit=jax.jit(jnp.tanh)
sum_jit = jit(jnp.sum)
subtract_jit = jit(jnp.subtract)
max_jit = jit(jnp.max)
argmax_jit = jit(jnp.argmax)
average_jit = jit(jnp.mean)

@partial(jit, in_shardings=None, out_shardings=None)
def herm_conj(H):
    return jnp.conjugate(jnp.transpose(H))

@partial(jit, in_shardings=None, out_shardings=None) #Input and output sharding makes no difference for eighenvalue computation (global function)
def eigh_jit(H):
    return jnp.linalg.eigh(H)

@partial(jit, in_shardings=None, out_shardings=None)
def s_wave_selfconsistent_condition(ui_up, ui_down, vi_up, vi_down, eigen_index):
    return (ui_up * conjugate_jit(vi_down))  #Zero-temperature formula (Only use single spin contribution formula for the below reason)
    #As jax is auto-block diagonalizing and adding a -1 phase to the eigenvectors of the other TRS pair/spin block of Hamiltonian --> Using the composite formula with two spin contributions results in zero
    #Using the other spin contibution instead ui_down * conjugate_jit(vi_up) results in a negative sign delta_s (initial calculation magnitude matches) which latter causes convergence issues

def hermitian_conjugate(H): #non-jitted version
    return jnp.conjugate(jnp.transpose(H))

#General functions
def debug_func():
    print('Total number of sites (n) is:',n,'for N_diag =',N_diag,', N =',N, ', and N_repeat =',N_repeat)
    #print(np.real(vertical_01_N_int_vec()))
    #print(np.real(np.diag(vertical_01_N_int_vec(),k=1)),np.diag(vertical_01_N_int_vec(),k=1).shape)
    #print(np.real(vertical_01_NN_int_vec()))
    #print(np.real(np.diag(vertical_01_NN_int_vec(),k=2)),np.diag(vertical_01_NN_int_vec(),k=2).shape)
    #print(np.real(horizontal_10_N_interactions()),horizontal_10_N_interactions().shape,np.sum(np.real(horizontal_10_N_interactions())))
    #print(np.real(horizontal_10_NN_interactions()),horizontal_10_NN_interactions().shape,np.sum(np.real(horizontal_10_NN_interactions())))
    #print(np.real(vertical_01_N_pbc_vec()))
    #print(np.diag(np.real(vertical_01_N_pbc_vec()),k=N-1), np.diag(vertical_01_N_pbc_vec(),k=N-1).shape)
    #print(np.real(vertical_01_NN_pbc_vec()))
    #print(np.diag(np.real(vertical_01_NN_pbc_vec()),k=N-2), np.diag(vertical_01_NN_pbc_vec(),k=N-2).shape)
    #print(np.diag(np.real(horizontal_10_N_pbc_vec()),k=N*(N-1)), np.diag(horizontal_10_N_pbc_vec(),k=N*(N-1)).shape)
    #print(np.real(horizontal_10_NN_pbc_vec()))
    #print(np.diag(np.real(horizontal_10_NN_pbc_vec()),k=N*(N-2)), np.diag(horizontal_10_NN_pbc_vec(),k=N*(N-2)).shape)
def gpu_memory_stats(): #Memory stats for GPU 0
    ms = jax.devices()[0].memory_stats()
    bytes_limit = ms["bytes_limit"] / 1024**3
    bytes_peak = ms["peak_bytes_in_use"] / 1024**3
    bytes_usage = ms["bytes_in_use"] / 1024**3  
    print(f'Memory usage of gpu 0: {bytes_usage:.2f}/{bytes_limit:.2f} GB, Peak memory usage: {bytes_peak:.2f}/{bytes_limit:.2f} GB')
def gpu_memory_stats_1(): #Memory stats for GPU 1
    ms = jax.devices()[1].memory_stats()
    bytes_limit = ms["bytes_limit"] / 1024**3
    bytes_peak = ms["peak_bytes_in_use"] / 1024**3
    bytes_usage = ms["bytes_in_use"] / 1024**3  
    print(f'Memory usage of gpu 1: {bytes_usage:.2f}/{bytes_limit:.2f} GB, Peak memory usage: {bytes_peak:.2f}/{bytes_limit:.2f} GB')

def vertical_01_N_int_vec(): #Creates a vector for vertical interactions between pairwise sites (N-interactions) in adjacent rows --> create diag matrix with this vector with k=+1 (+ h.c for lower triangular matrix)
    vertical_01 = []
    for i in np.arange(N_repeat): #Place 1's at adjacent interaction sites and 0's at adjacent non-interacting sites of the repeating chains
        vertical_01.extend([1+0j]*(N-1))
        vertical_01.append(0+0j) 
    for i in np.arange(N-1,N_repeat-1,-1): #Place 1's at adjacent interaction sites and 0's at adjacent non-interacting sites of the isosceles right triangle
        vertical_01.extend([1+0j]*i) 
        vertical_01.append(0+0j)
    for i in range(np.abs(len(vertical_01)-(n-1))): #As vector is placed at k=1 in diag matrix, pop the elements more than n-k = n-1 that are not part of the lattice
        vertical_01.pop()
    return np.array(vertical_01,dtype=np.complex128)
def vertical_01_NN_int_vec(): #Create a vector for vertical interactions between alternating sites (NN-interactions) --> create diag matrix with this vector with k=+2 (+ h.c for lower triangular matrix
    vertical_01 = []
    for i in np.arange(N_repeat): #Place 1's at NN interaction sites and 0's at NN non-interacting sites of the repeating chains
        vertical_01.extend([1+0j]*(N-2))
        vertical_01.extend([0+0j]*2) #pad two zeros because we are looking at NN interactions which are alternating between sites
    for i in np.arange(N-2,N_repeat-2,-1): #Place 1's at NN interaction sites and 0's at NN non-interacting sites of the isosceles right triangle
        vertical_01.extend([1+0j]*i) 
        vertical_01.extend([0+0j]*2)
    for i in range(np.abs(len(vertical_01)-(n-2))): #As vector is placed at k=2 in diag matrix, pop the elements more than n-k = n-2 that are not part of the lattice
        vertical_01.pop()
    return np.array(vertical_01,dtype=np.complex128)
def horizontal_10_N_interactions(): #Horizontal interactions between pairwise sites in adjacent columns 
    if N_diag == 0 or N_diag == 1: #Return pattern for normal square lattice
        return np.eye(n,n,k=N)
    horizontal_10 = [] #First construct the repeating square/rectangle lattice
    horizontal_10.extend([1]*(N*N_repeat)) #Place 1's at the interaction sites of repeating chains
    horizontal_10.extend([0]*(np.abs(len(horizontal_10)-(n-N)))) #As vector will be placed at k=N pad zeroes for non-interacting sites accordingly to the right
    horizontal_10_matrix = np.diag(jnp.array(horizontal_10,dtype=np.complex128),k=N) #k=N because horizntal bonds are between (i)th and (i+N)th sites in a rectangle/square lattice (Repeating lattice is a square/rectangle)
    bonds_in_triangle = np.arange(N-1,N_repeat,-1) #Number of horizontal bonds/sites in the isosceles right triangle
    for i in range(len(bonds_in_triangle)): #Code to construct interaction patterns for the isosceles right triangle
        horizontal_10 = [] #Cannot vmap for loop becuase we need/access the value "i" in the loop
        horizontal_10.extend([1]*bonds_in_triangle[i]) #Place 1's in at the interaction sites
        horizontal_10.extend([0]*(np.sum(bonds_in_triangle[i+1:]))) #As we proceed column-wise (each column has one lesser site), we pad zeroes (= number of bonds in the leftover/remaining columns of the triangle) at the right (non-interacting for this column)  
        left_padding_length = np.abs(len(horizontal_10) - (n-bonds_in_triangle[i])) #As vector will be placed at k=bonds_in_triangle[i] pad zeroes (= len(vec) - (n-k)) for non-interacting sites accordingly to the left
        left_padding = [0]*left_padding_length
        horizontal_10 = left_padding + horizontal_10
        horizontal_10_matrix = np.diag(np.array(horizontal_10,dtype=np.complex128),k=bonds_in_triangle[i]) + horizontal_10_matrix
    return horizontal_10_matrix
def horizontal_10_NN_interactions(): #Horizontal interactions between sites in alternating columns
    if N_diag == 0 or N_diag == 1:
        return np.eye(n,n,k=2*N) #Return pattern for normal square lattice --> k=2N because we consider alternating columns (NN interactions)
    horizontal_10 = []
    if N_repeat!=0: #For sharp triangles, the repeating column is not present, so avoid creation of incorrect dimension zero matrix
        horizontal_10.extend([1]*(N*(N_repeat-1))) #N_repeat-1 because we consider alternating columns. The repeating column (with full N sites) adjacent to the isosceles triangle is not considered as it already sees the effect of decreasing lattice sites in the triangle 
        horizontal_10.extend([0]*(np.abs(len(horizontal_10)-(n-2*N)))) #As vector will be placed at k=2N pad zeroes for non-interacting sites accordingly to the right
        horizontal_10_matrix = np.diag(np.array(horizontal_10,dtype=np.complex128),k=2*(N)) #k = 2*(N) because horizntal bonds are between (i)th and (i+2*N)th sites in a rectangle/square lattice (Repeating lattice is a square/rectangle) 
    bonds_in_triangle = np.arange(N-1,N_repeat,-1) #Number of horizontal NN bonds in the isosceles right triangle and its adjacent full N-sites column
    if N_repeat==0: #For sharp triangles, the repeating column is not present, so we start with N-2 bonds within the triangle
        bonds_in_triangle = np.arange(N-2,0,-1)
        horizontal_10_matrix = np.zeros((n,n),dtype=np.complex128)
    for i in range(len(bonds_in_triangle)): #Code to construct interaction patterns for the isosceles right triangle
        horizontal_10 = []
        horizontal_10.extend([1]*(bonds_in_triangle[i])) #Place 1's in at the interaction sites
        horizontal_10.extend([0]*(np.sum(bonds_in_triangle[i+1:]))) #As we proceed column-wise (each column has one lesser site), we pad zeroes (= number of bonds in the leftover/remaining columns of the triangle) at the right (non-interacting for this column)
        left_padding_length = np.abs(len(horizontal_10) - (n-(2*bonds_in_triangle[i]+1))) #As vector will be placed at k=2*bonds_in_triangle[i]+1 pad zeroes (= len(vec) - (n-k)) for non-interacting sites accordingly to the left
        left_padding = [0]*left_padding_length
        horizontal_10 = left_padding + horizontal_10
        horizontal_10_matrix = np.diag(np.array(horizontal_10),k=2*bonds_in_triangle[i]+1) + horizontal_10_matrix #k=2*(bonds_in_triangle[i])+1 the bonds within the triangle (starting from the adjacent full N-site column) are in between (i)th and (i+(2*N)+1)th because each column in the triangle has one lesser site 
    return horizontal_10_matrix
def vertical_01_N_pbc_vec(): #Only applies to a square lattice
    vertical_01_pbc = [1+0j]
    vertical_01_pbc.extend([0+0j]*(N-1)) #create base vector for vertical interactions with periodic boundary conditions
    vertical_01_pbc.extend(vertical_01_pbc*N) #repeat the base vector N times for N columns
    for i in range(np.abs(len(vertical_01_pbc)-(n-(N-1)))): #As vector is placed at k = N-1 in diag matrix, pop the elements more than n-k = n-(N-1) that are not part of the lattice
        vertical_01_pbc.pop() #resultant matrix using this vector must be kroned with hermitian conjugate of the vertical 01 interaction matrix as the direction of interaction is opposite in the periodic boundary (down to up lattice unlike the usual up to down lattice)
    return np.array(vertical_01_pbc,dtype=np.complex128) #vector to be placed at k=N-1 in diag matrix because bond is between (i)th and (i+N-1)th sites in a square lattice
def vertical_01_NN_pbc_vec(): #Only applies to a square lattice
    return np.pad(vertical_01_N_pbc_vec(),(0,1),'constant') #pad one zero to the right for NN interactions and place the vector at k=N-2 in diag matrix because bond is between (i)th and (i+N-2)th sites in a square lattice
def horizontal_10_N_pbc_vec(): #Only applies to a square lattice
    return np.ones(N) #vector to be placed at k=N*(N-1) in diag matrix because bond is between (i)th and (i+N*(N-1))th sites in a square lattice
def horizontal_10_NN_pbc_vec(): #Only applies to a square lattice
    return np.pad(np.ones(N),(0,N),'constant') #vector to be placed at k=N*(N-2) in diag matrix because bond is between (i)th and (i+N*(N-2))th sites in a square lattice --> np.abs(N - (n-N*(N-2))) = N

def lorentzian(eeta, x, x0):
    return (1/np.pi)*(eeta/((x-x0)**2 + eeta**2))
def lorentzian_weighing(eeta, omega, energy):
    return lorentzian(eeta, omega, energy)+lorentzian(eeta, omega, energy*(-1))
lorentzian_weighing_v = jax.vmap(lorentzian_weighing, in_axes=(None, None, 0))
def lorentzian_DOS(eeta, omega_axes, energy):
    return jnp.sum(lorentzian_weighing_v(eeta, omega_axes, energy)) #Add contribution from each eigenenergy for one omega value
lorentzian_DOS_v = jax.vmap(lorentzian_DOS, in_axes=(None, 0, None))
def rho(omega, state_val, eeta): #state_val corresponds to one energy in [omega-"energy"] --> rho encapsulates the spatial information/lattice sites
        PA = np.array(eigenstates[:,int(state_val)]).reshape(n,2,2) #This gives [[u_up, u_down], [v_up, v_down]] for each site for the selected energy 
        #print(PA.shape)
        PA2 = LA.norm(PA, axis=2)**2 #This gives (|u_i_up|^2 + |u_i_down|^2) --> index 0, and (|v_i_up|^2 + |v_i_down|^2 --> index 1, for each site
        #print(PA2.shape)
        PA3 = np.zeros(n)
        for i in range(n): #i is the site index accessing probability amplitudes/weights at each site (|u_i_up|^2 + |u_i_down|^2), (|v_i_up|^2 + |v_i_down|^2)
            PA3[i] = PA2[i][0]*lorentzian(eeta,omega,eigenenergies[state_val])+PA2[i][1]*lorentzian(eeta,omega,eigenenergies[state_val]*(-1))
        #print(PA3.shape)
        return PA3
def lorentzian_LDOS(omega, eeta):
    LDOS_for_each_energy = [rho(omega, state_val, eeta) for state_val in range(len(eigenenergies))] #This gives the LDOS at each site "i" for each [omega-"energy"] value
    LDOS = np.sum(np.array(LDOS_for_each_energy), axis=0) #Sum over all energies to get the total LDOS at each site "i"
    LDOS = LDOS/np.max(LDOS) #Normalize the LDOS to 1 with the maximum value
    #print(LDOS.shape) #LDOS is a 1D array with "n" elements corresponding to total sites "n"
    return LDOS #--> Pass this 1D LDOS array to a plotter function to plot the LDOS at each site "i" for a given [omega-"energy"] value

def s_wave_data_collect_and_compute(site_index, eigen_index): #Collects the relevant u,v amplitudes for s-wave self-consistent condition
    ui_up = dynamic_slice(eigenstates, (df*(site_index),eigen_index), (1,1))[0,0] #Picks an eigenvector as eigenstate[eigen_index] and extracts the u_i_up amplitude for a specific site "i"
    ui_down = dynamic_slice(eigenstates, (df*(site_index)+1,eigen_index), (1,1))[0,0] #4*(site_index)+"something" as we have 4 degrees of freedom at each site
    vi_up = dynamic_slice(eigenstates, (df*(site_index)+2,eigen_index), (1,1))[0,0] #Order of u,v amplitudes depend on the basis fermionic vector [c_up_i, c_down_i, c_up_dagger_i, c_down_dagger_i]
    vi_down = dynamic_slice(eigenstates, (df*(site_index)+3,eigen_index), (1,1))[0,0]
    delta_ii_for_one_eigenvector = s_wave_selfconsistent_condition(ui_up, ui_down, vi_up, vi_down, eigen_index) #Compute the s-wave self-consistent condition for a specific site and eigenvector 
    return delta_ii_for_one_eigenvector
s_wave_data_collect_and_compute_v = jax.vmap(s_wave_data_collect_and_compute, in_axes=(None, 0)) #batching for all eigenvectors

def self_consistent_s_wave(site_index, eigen_index_vec):
    delta_s_new = sum_jit(s_wave_data_collect_and_compute_v(site_index,eigen_index_vec))*(V_sc) #Sum s-wave self-consistent condition over relevant eigenvectors for a specific site
    return delta_s_new #Returns a single value for a single site after implementing the s-wave self-consistent condition summed over relevant eigenvectors
self_consistent_s_wave_v = jax.vmap(self_consistent_s_wave, in_axes=(0,None)) #batching for all sites

@partial(jit, in_shardings=None, out_shardings=None)
def H_comp_s(delta_s_vec): 
    H_on_s_SC = jnp.zeros((df*n,df*n),dtype=jnp.complex128)
    for i in range(n):
        s_sc_mat = SC_mat*delta_s_vec[i] + hermitian_conjugate(SC_mat*delta_s_vec[i]) #Construct 4x4 site specific s-wave SC Hamiltonian
        H_on_s_SC = H_on_s_SC.at[df*i:df*(i+1),df*(i):df*(i+1)].set(s_sc_mat) #Place the site specific s-wave SC Hamilton on the diagonal on-site part of main Hamiltonian
    #print('Memory before moving H_on_s_SC to CPU:')
    #gpu_memory_stats()
    H_on_s_SC = jax.device_put(H_on_s_SC, jax.devices('cpu')[0]) #Move H_on_s_SC to CPU
    #print('Memory after moving H_on_s_SC to CPU:')
    #gpu_memory_stats()
    H_comp = H_comp_static + H_on_s_SC #Construct composite Hamiltonian on CPU
    del s_sc_mat #Delete intermediate matrices to free up memory
    gc.collect()
    #print('Memory after constructing H_comp with s-wave SC (moved to CPU) and deleting intermediate matrices on GPU:')
    #gpu_memory_stats()
    return H_comp

def delta_s_selfconsistency(delta_s_vec): 
    global eigenenergies, eigenstates, run_count
    H = H_comp_s(delta_s_vec)
    H = jax.device_put(H, jax.devices('gpu')[0]) #Move H_comp to GPU 0 for diagonalizing
    print('Memory after after moving H_comp to GPU for diagonalizing:')
    gpu_memory_stats()
    t0 = time.time()
    eigenenergies, eigenstates = eigh_jit(H)
    t1 = time.time()
    #print(f'Time taken to diagonalize Composite Hamiltonian for run: {run_count} with {n} sites is {t1 - t0:.4f} s')
    #print(eigenenergies.shape,eigenstates.shape)
    logging.info('########################')
    logging.info('Eigenenergy and eigenvector computation for Run: %s took %s seconds',run_count,t1 - t0)
    delta_s_new_vec = self_consistent_s_wave_v(site_index_vec,eigen_index_vec) #Compute the new s-wave SC order parameter vector
    average_delta_s = average_jit(delta_s_new_vec)
    logging.info('New (unmixed) s-wave SC order parameter vector computed for run: %s has the average value: %s',run_count, average_delta_s)
    error_vec = jnp.abs(subtract_jit(delta_s_vec,delta_s_new_vec))
    max_error = max_jit(error_vec)
    
    if max_error > convergence_limit:
        run_count += 1
        logging.info('Maximum error for run: %s is %s with index %s',run_count,max_error, argmax_jit(error_vec))
        if run_count>max_runs:
            print('Run Count Exceeded %f - Self-Consistency not achieved for set convergence limit: %f'%(max_runs,convergence_limit))
            logging.info('Run Count Exceeded %s - Self-Consistency not achieved for set convergence limit: %s',max_runs,convergence_limit)
            return delta_s_new_vec
        else:    
            delta_s_new_vec = delta_s_vec * (1-alpha) + delta_s_new_vec * alpha
            del max_error, error_vec, average_delta_s, delta_s_vec, H, eigenenergies, eigenstates
            gc.collect()
            return delta_s_selfconsistency(delta_s_new_vec)
    else:
        print('Self-Consistency achieved for set convergence limit:',convergence_limit)
        logging.info('Self-Consistency achieved for set convergence limit: %s',convergence_limit)
        return delta_s_new_vec

#(GPU diagonalization possible for N = 73, N_diag = 63, n = 3376 with N_repeat = 10 --> Peak memory usage: 16.36/18.18 GB, Time taken for first run with jit to diagonalize Composite Hamiltonian with 3376 sites is 100.6677 s)
#Can vary N and N_repeat accordingly to GPU diagonalize just a slightly larger lattice 

#Model Parameters
N = 30 #number of lattice sites in x and y direction
N_diag = 0 #number of lattice sites in the isosceles right triangle hypotenuse edge
N_repeat = N - N_diag #number of lattice sites in the repeating/extra horizontal and vertical chains besides the isosceles right triangle
n = int((N_diag*(N_diag+1))/2) + N_repeat*N_diag + (N_diag+N_repeat)*N_repeat #Total number of lattice sites
print('Total number of sites (n) is:',n,'for N_diag =',N_diag,', N =',N, ', and N_repeat =',N_repeat)
df = 4 #degrees of freedom at each site
t_N = 1+0j #neighbour (N) hopping
t_NN = 0 #-0.25+0j #next-neighbour (NN) hopping
mu = 0 #Chemical Potential
V_sc = 2*t_N #s-wave SC coupling strength
delta_s_init = 0.3+0j #s-wave superconducting order parameter
delta_s_vec = np.ones(n,dtype=np.complex128)*delta_s_init #s-wave superconducting order parameter vector for each site in square lattice
PBC = True #Periodic Boundary Conditions - True for PBC and False for OBC

#Code Parameters
eeta = 0.04 #Lorentzian broadening factor for DOS and LDOS calculations
max_runs = 50
convergence_limit = 1e-4
alpha = 0.4 #Mixing parameter for the new and old s-wave SC order parameter vectors
site_index_vec = jnp.arange(n)
eigen_index_vec = jnp.arange(int(n*df/2),int(n*df)) 

#Pauli Matrices and other base matrices
X=np.array([[0,1],[1,0]])
Y=np.array([[0,0-1j],[0+1j,0]])
Z=np.array([[1,0],[0,-1]])
W=np.array([[0,0+1j],[0,0]])
SC_mat=np.kron(W,Y) #This pattern matrix needs to be added with its herm_conj to get the SC Hamiltonian = SC_mat*delta + herm_conj(SC_mat*delta)

#Define "static" site-specific Hamiltonian of dimension (df,df) --> basis vector is [c_up_i, c_down_i, c_up_dagger_i, c_down_dagger_i], i is the site index
H_on = np.kron(Z, np.eye(2, 2, k=0)) * (-1*mu)
H_off_N_int = np.kron(Z, np.eye(2, 2, k=0)) * (1*t_N)
H_off_NN_int = np.kron(Z, np.eye(2, 2, k=0)) * (1*t_NN)

#region Construct the "static" part of the Composite Hamiltonian and place it on CPU (+timing this one-time operation)
#print('Memory before H_comp_upper_off construction:')
#gpu_memory_stats()
t0 = time.time()

if PBC==True: #PBC Hamiltonian
    H_comp_upper_off = kron_jit(jnp.diag(vertical_01_N_int_vec(),k=1), H_off_N_int) + kron_jit(jnp.diag(vertical_01_N_pbc_vec(),k=N-1), hermitian_conjugate(H_off_N_int)) + kron_jit(jnp.diag(vertical_01_NN_int_vec(),k=2), H_off_NN_int) + kron_jit(jnp.diag(vertical_01_NN_pbc_vec(),k=N-2), hermitian_conjugate(H_off_NN_int))+ kron_jit(horizontal_10_N_interactions(), H_off_N_int) + kron_jit(jnp.diag(horizontal_10_N_pbc_vec(),k=N*(N-1)), hermitian_conjugate(H_off_N_int)) + kron_jit(horizontal_10_NN_interactions(), H_off_NN_int) + kron_jit(jnp.diag(horizontal_10_NN_pbc_vec(),k=N*(N-2)), hermitian_conjugate(H_off_NN_int))
else: #OBC Hamiltonian
    H_comp_upper_off = kron_jit(jnp.diag(vertical_01_N_int_vec(),k=1), H_off_N_int) + kron_jit(jnp.diag(vertical_01_NN_int_vec(),k=2), H_off_NN_int) + kron_jit(horizontal_10_N_interactions(), H_off_N_int) + kron_jit(horizontal_10_NN_interactions(), H_off_NN_int)
#print('Memory after H_comp_upper_off construction:')
#gpu_memory_stats()

H_comp_off = H_comp_upper_off + herm_conj(H_comp_upper_off)
del H_comp_upper_off
#print('Memory after H_comp_off construction and deletion of intermediate arrays:')
#gpu_memory_stats()

H_comp_off = jax.device_put(H_comp_off, jax.devices('cpu')[0])
#print('Memory after moving H_comp_upper to CPU:')
#gpu_memory_stats()

H_comp_on = kron_jit(jnp.eye(n, n, k=0), H_on)
#print('Memory after H_comp_on construction:')
#gpu_memory_stats()
H_comp_on = jax.device_put(H_comp_on, jax.devices('cpu')[0])
#print('Memory after moving H_comp_on to CPU:')
#gpu_memory_stats()

H_comp_static = H_comp_on + H_comp_off #H_comp_static is on the CPU
t1 = time.time()
print('Memory after H_comp_static construction and moving it to CPU:')
gpu_memory_stats()
print(f'Time taken to construct Composite Hamiltonian with {n} sites is {t1 - t0:.4f} s')
#endregion

#Testing the limits of GPU Diagonalization with Composite Hamiltonian
if False:
    t0 = time.time()
    H_comp_static = jax.device_put(H_comp_static, jax.devices('gpu')[0])
    print('Memory after moving H_comp_static to GPU:')
    gpu_memory_stats()
    eigenvalues, eigenvectors = jnp.linalg.eigh(H_comp_static)
    print('Memory after diagonalization:')
    gpu_memory_stats()
    t1 = time.time()
    print(f'Time taken to diagonalize Composite Hamiltonian with {n} sites is {t1 - t0:.4f} s')

eigenenergies, eigenstates, run_count = None, None, 0
if delta_s_init == 0+0j:
    print('Performing normal state calculations...')
    H = jax.device_put(H_comp_static, jax.devices('gpu')[0]) #Move H_comp to GPU for diagonalizing
    eigenenergies, eigenstates = eigh_jit(H)
    delta_s_new_vec = delta_s_vec
else:
    print('Entering self-consistent calculations...')
    delta_s_new_vec = delta_s_selfconsistency(delta_s_vec)
    print(delta_s_new_vec)

eigenenergies_window = eigenenergies[int((n*df)/2)-int((n*df)/4):int((n*df)/2)+int((n*df)/4)] #Window half the spectrum around zero energy
#eigenstates_window = eigenstates[:,int((n*df)/2)-200:int((n*df/2))+200]
print('Saving DOS and half the spectrum of eigenvalues around zero energy:',eigenenergies_window.shape)
omega_axes = jnp.arange(-1,1+0.02,0.02) #Energy range, points, and spacing for DOS calculation
DOS = lorentzian_DOS_v(eeta, omega_axes, eigenenergies) #--> Pass normalized 1D DOS array to a plotter function to plot the DOS for a given energy range
DOS = DOS/jnp.max(DOS) #Normalize the DOS to 1 with the maximum value
np.savez('/home/susva433/Test_Data/SC_s_square_lattice_N=30_PBC_delta_s_init=0.4_eeta=0.04_V=2.npz',eigenenergies=eigenenergies_window, omega_axes=omega_axes, DOS=DOS, delta_s=delta_s_new_vec)