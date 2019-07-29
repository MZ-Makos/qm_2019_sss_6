import numpy as np
import fock_fast as ff

class scf():
    def __init__(self,hamiltonian_matrix,interaction_matrix,density_matrix,chi_tensor,energy_ion,ionic_charge, orbitals_per_atom):
        self.hamiltonian_matrix = hamiltonian_matrix
        self.interaction_matrix = interaction_matrix
        self.density_matrix = density_matrix
        self.chi_tensor = chi_tensor
        self.energy_ion = energy_ion
        self.ionic_charge= ionic_charge
        self.converged = False
        self.orbitals_per_atom = orbitals_per_atom

    '''
    @property
    def fock_matrix(self):
        return self.fock_matrix

    @_fock_matrix.setter
    def _fock_matrix(self, new_fock_matrix):
        self.fock_matrix = new_fock_matrix

    @property
    def density_matrix(self):
        return self.density_matrix

    @_density_matrix.setter
    def _density_matrix(self, new_density_matrix):
        self.density_matrix = new_density_matrix
    '''

    def scf_cycle(self, max_scf_iterations = 100,
                mixing_fraction = 0.25, convergence_tolerance = 1e-4):
        """Returns converged density & Fock matrices defined by the input Hamiltonian, interaction, & density matrices.

        Parameters
        ----------
        Initial Hamiltonian : np.array
            Defines the initial orbital energy and phase space.
        Interaction Matrix: ndarray
            Defines the initial interaction between different atoms.
        Density Matrix: ndarray
            Defines the electron density on atoms.

        Returns
        -------
        If SCF converges, then returns modified density matrix and modified fock matrix.
    """
        old_density_matrix = self.density_matrix.copy()
        for iteration in range(max_scf_iterations):
            fock_matrix = self.fast_fock_matrix(self.hamiltonian_matrix, self.interaction_matrix,
        self.density_matrix)
            density_matrix = self.calculate_density_matrix(fock_matrix)
            error_norm = np.linalg.norm( old_density_matrix - density_matrix)
            if error_norm < convergence_tolerance:
                self.converged=True
                return density_matrix, fock_matrix

            old_density_matrix = (mixing_fraction * density_matrix
                                + (1.0 - mixing_fraction) * old_density_matrix)
        print("WARNING: SCF cycle didn't converge")
        return density_matrix, fock_matrix

    def calculate_energy_scf(self,hamiltonian_matrix,fock_matrix,density_matrix):
        '''Returns the Hartree-Fock total energy defined by the input Hamiltonian, Fock, & density matrices.

        Inputs
        ------
        hamiltonian_matrix : np.array

        fock_matrix : np.array

        density_matrix : np.array


        Output
        ------
        energy_scf : float
            Hartree-Fock total energy

        '''
        energy_scf = np.einsum('pq,pq', hamiltonian_matrix + fock_matrix,
                            density_matrix)
        return energy_scf

    def calculate_density_matrix(self,fock_matrix):
        '''Returns the 1-electron density matrix defined by the input Fock matrix.

            Parameters
        ------------
        fock_matrix : np.array

        Return
        ------------
        density_matrix : np.array


        '''
        num_occ = (self.ionic_charge // 2) * np.size(fock_matrix,
                                                0) // self.orbitals_per_atom
        orbital_energy, orbital_matrix = np.linalg.eigh(fock_matrix)
        occupied_matrix = orbital_matrix[:, :num_occ]
        density_matrix = occupied_matrix @ occupied_matrix.T
        return density_matrix
    
    def fast_fock_matrix(self,hamiltonian_matrix,interaction_matrix,density_matrix):
        dipole = 2.781629275106456
        return ff.fast_fock_matrix(hamiltonian_matrix,interaction_matrix,density_matrix,dipole)


    def calculate_fock_matrix(self,hamiltonian_matrix,interaction_matrix,density_matrix, chi_tensor):
        '''Returns the Fock matrix defined by the input Hamiltonian, interaction, & density matrices.

        Parameters
        ------------
        hamiltonian_matrix : np.array
        interaction_matrix : np.array
        density_matrix : np.array
        chi_tensor : np.array

        Return
        ------------
        fock_matrix : np.array

        '''
        fock_matrix = self.hamiltonian_matrix.copy()
        fock_matrix += 2.0 * np.einsum('pqt,rsu,tu,rs',
                                    self.chi_tensor,
                                    self.chi_tensor,
                                    self.interaction_matrix,
                                    self.density_matrix,
                                    optimize=True)
        fock_matrix -= np.einsum('rqt,psu,tu,rs',
                                self.chi_tensor,
                                self.chi_tensor,
                                self.interaction_matrix,
                                self.density_matrix,
                                optimize=True)
        return fock_matrix

    def initialize(self):
        #self.fock_matrix = self.calculate_fock_matrix(self.hamiltonian_matrix, self.interaction_matrix,
        #self.density_matrix, self.chi_tensor)
        self.fock_matrix = self.fast_fock_matrix(self.hamiltonian_matrix, self.interaction_matrix,
        self.density_matrix)
        self.density_matrix = self.calculate_density_matrix(self.fock_matrix)

    def kernel(self):
        self.initialize()
        self.density_matrix, self.fock_matrix = self.scf_cycle()
        self.energy_scf = self.calculate_energy_scf(self.hamiltonian_matrix,self.fock_matrix,self.density_matrix)
        self.total_energy = self.energy_ion + self.energy_scf
        return self.total_energy
