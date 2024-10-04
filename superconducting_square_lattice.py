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
#logging.basicConfig(filename='/home/susva433/Data/d-wave_phase_crystal_self-consistency_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

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
@partial(jit, in_shardings=None, out_shardings=None)
def herm_conj(H):
    return jnp.conjugate(jnp.transpose(H))

@partial(jit, in_shardings=None, out_shardings=None) #Input and output sharding makes no difference for eighenvalue computation (global function)
def eigh_jit(H):
    return jnp.linalg.eigh(H)

def hermitian_conjugate(H):
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
def gpu_memory_stats():
    ms = jax.devices()[0].memory_stats()
    bytes_limit = ms["bytes_limit"] / 1024**3
    bytes_peak = ms["peak_bytes_in_use"] / 1024**3
    bytes_usage = ms["bytes_in_use"] / 1024**3  
    print(f'Memory usage: {bytes_usage:.2f}/{bytes_limit:.2f} GB, Peak memory usage: {bytes_peak:.2f}/{bytes_limit:.2f} GB')

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

def lorentzian(x, x0, eeta):
    return (1/np.pi)*(eeta/((x-x0)**2 + eeta**2))
def rho(omega, state_val, eeta): #state_val corresponds to one energy in [omega-"energy"] --> rho encapsulates the spatial information/lattice sites
        PA = np.array(eigenstates[:,int(state_val)]).reshape(n,2,2) #This gives [[u_up, u_down], [v_up, v_down]] for each site for the selected energy 
        #print(PA.shape)
        PA2 = LA.norm(PA, axis=2)**2 #This gives (|u_i_up|^2 + |u_i_down|^2) --> index 0, and (|v_i_up|^2 + |v_i_down|^2 --> index 1, for each site
        #print(PA2.shape)
        PA3 = np.zeros(n)
        for i in range(n): #i is the site index accessing probability amplitudes/weights at each site (|u_i_up|^2 + |u_i_down|^2), (|v_i_up|^2 + |v_i_down|^2)
            PA3[i] = PA2[i][0]*lorentzian(omega,eigenenergies[state_val],eeta)+PA2[i][1]*lorentzian(omega,eigenenergies[state_val]*(-1),eeta)
        #print(PA3.shape)
        return PA3
def lorentzian_LDOS(omega, eeta):
    LDOS_for_each_energy = [rho(omega, state_val, eeta) for state_val in range(len(eigenenergies))] #This gives the LDOS at each site "i" for each [omega-"energy"] value
    LDOS = np.sum(np.array(LDOS_for_each_energy), axis=0) #Sum over all energies to get the total LDOS at each site "i"
    LDOS = LDOS/np.max(LDOS) #Normalize the LDOS to 1 with the maximum value
    #print(LDOS.shape) #LDOS is a 1D array with "n" elements corresponding to total sites "n"
    return LDOS #--> Pass this 1D LDOS array to a plotter function to plot the LDOS at each site "i" for a given [omega-"energy"] value
def lorentzian_DOS(eeta):
    max_energy, min_energy = jnp.max(eigenenergies), jnp.min(eigenenergies)
    print(max_energy, min_energy)
    omega_axes = jnp.arange(min_energy,max_energy+0.01,0.01) #Energy range, points, and spacing for DOS calculation
    DOS = jnp.zeros(len(omega_axes))
    print('Entering DOS calculation...')
    for i in range(len(omega_axes)):
        for state_val in range(len(eigenenergies)):
            #DOS[i]+=lorentzian(omega_axes[i],eigenenergies[state_val],eeta)+lorentzian(omega_axes[i],eigenenergies[state_val]*(-1),eeta)
            DOS = DOS.at[i].add(lorentzian(omega_axes[i],eigenenergies[state_val],eeta)+lorentzian(omega_axes[i],eigenenergies[state_val]*(-1),eeta))
    DOS = DOS/jnp.max(DOS) #Normalize the DOS to 1 with the maximum value
    print(DOS.shape) #DOS is a 1D array with "len(omega)" elements corresponding to the energy range
    return omega_axes, DOS #--> Pass this 1D DOS array to a plotter function to plot the DOS for a given energy range

def H_comp_s(delta_s_vec): 
    H_on_s_SC = jnp.zeros((df*n,df*n),dtype=jnp.complex128)
    for i in range(n):
        s_sc_mat = SC_mat*delta_s_vec[i] + hermitian_conjugate(SC_mat*delta_s_vec[i]) #Construct 4x4 site specific s-wave SC Hamiltonian
        H_on_s_SC = H_on_s_SC.at[df*i:df*(i+1),df*(i):df*(i+1)].set(s_sc_mat) #Place the site specific s-wave SC Hamilton on the diagonal on-site part of main Hamiltonian
    print('Memory before moving H_comp_static to GPU:')
    gpu_memory_stats()
    H_comp_static_gpu = jax.device_put(H_comp_static, jax.devices('gpu')[0])
    H_comp = H_comp_static_gpu + H_on_s_SC #Add the s-wave SC Hamiltonian to the static part of the Composite Hamiltonian
    print('Memory after constructing H_comp:')
    gpu_memory_stats()
    del H_comp_static_gpu, H_on_s_SC #Delete the intermediate Hamiltonians
    print('Memory after deleting intermediate Hamiltonians:')
    gpu_memory_stats()
    return H_comp

def delta_s_selfconsistency(delta_s_vec): 
    global eigenenergies, eigenstates
    eigenenergies, eigenstates = jnp.linalg.eigh(H_comp_s(delta_s_vec))
    print('Memory after diagonalizing Composite Hamiltonian:')
    gpu_memory_stats()
    print(eigenenergies.shape,eigenstates.shape)
    eigenenergies_window = eigenenergies[int((n*df)/2)-200:int((n*df)/2)+200]
    #eigenstates_window = eigenstates[:,int((n*df)/2)-200:int((n*df/2))+200]
    print(eigenenergies_window.shape)
    omega_axes, DOS = lorentzian_DOS(eeta)
    np.savez('/home/susva433/Test_Data/SC_s_square_lattice_N=60_delta_s=0.4_eeta=0.04.npz',eigenenergies=eigenenergies_window, omega_axes=omega_axes, DOS=DOS)
    return delta_s_vec

#(GPU diagonalization possible for N = 73, N_diag = 63, n = 3376 with N_repeat = 10 --> Peak memory usage: 16.36/18.18 GB, Time taken for first run with jit to diagonalize Composite Hamiltonian with 3376 sites is 100.6677 s)
#Can vary N and N_repeat accordingly to GPU diagonalize just a slightly larger lattice 
#Main code with parameters 
N = 55 #number of lattice sites in x and y direction
N_diag = 0 #number of lattice sites in the isosceles right triangle hypotenuse edge
N_repeat = N - N_diag #number of lattice sites in the repeating/extra horizontal and vertical chains besides the isosceles right triangle
n = int((N_diag*(N_diag+1))/2) + N_repeat*N_diag + (N_diag+N_repeat)*N_repeat #Total number of lattice sites
print('Total number of sites (n) is:',n,'for N_diag =',N_diag,', N =',N, ', and N_repeat =',N_repeat)
df = 4 #degrees of freedom at each site
t_N = 1+0j #neighbour (N) hopping
t_NN = 0 #-0.25+0j #next-neighbour (NN) hopping
mu = 0 #Chemical Potential
eeta = 0.04 #Lorentzian broadening factor for DOS and LDOS calculations
delta_s_init = 0.4+0j #s-wave superconducting order parameter
delta_s_vec = np.ones(n,dtype=np.complex128)*delta_s_init #s-wave superconducting order parameter vector for each site in square lattice

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
print('Memory before H_comp_upper_off construction:')
gpu_memory_stats()
t0 = time.time()

H_comp_upper_off = jnp.kron(jnp.diag(vertical_01_N_int_vec(),k=1), H_off_N_int) + jnp.kron(jnp.diag(vertical_01_NN_int_vec(),k=2), H_off_NN_int) + jnp.kron(horizontal_10_N_interactions(), H_off_N_int) + jnp.kron(horizontal_10_NN_interactions(), H_off_NN_int)
print('Memory after H_comp_upper_off construction:')
gpu_memory_stats()

H_comp_off = H_comp_upper_off + herm_conj(H_comp_upper_off)
del H_comp_upper_off
print('Memory after H_comp_off construction and deletion of intermediate arrays:')
gpu_memory_stats()

H_comp_off = jax.device_put(H_comp_off, jax.devices('cpu')[0])
print('Memory after moving H_comp_upper to CPU:')
gpu_memory_stats()

H_comp_on = jnp.kron(jnp.eye(n, n, k=0), H_on)
print('Memory after H_comp_on construction:')
gpu_memory_stats()
H_comp_on = jax.device_put(H_comp_on, jax.devices('cpu')[0])
print('Memory after moving H_comp_on to CPU:')
gpu_memory_stats()

H_comp_static = H_comp_on+H_comp_off #H_comp_static is on the CPU
t1 = time.time()
print('Memory after H_comp_static construction:')
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

eigenenergies, eigenstates = None, None
print('Entering self-consistent calculations...')
delta_s_new_vec = delta_s_selfconsistency(delta_s_vec)