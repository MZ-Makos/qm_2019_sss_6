class scf():
    def __init__(self,hamiltonian_matrix,interaction_matrix,density_matrix,chi_tensor, energy_ion, converged=False):
        self.hamiltonian_matrix = hamiltonian_matrix
        self.interaction_matrix = interaction_matrix
        self.density_matrix = density_matrix
        self.chi_tensor = chi_tensor
        self.energy_ion = energy_ion
        self.converged = converged

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
            self.fock_matrix = calculate_fock_matrix(self.hamiltonian_matrix, self.interaction_matrix, old_density_matrix, self.chi_tensor)
            self.density_matrix = calculate_density_matrix(self.fock_matrix)

            error_norm = np.linalg.norm( old_density_matrix - self.density_matrix)
            if error_norm < convergence_tolerance:
                return self.density_matrix, self.fock_matrix

            old_density_matrix = (mixing_fraction * self.density_matrix
                                + (1.0 - mixing_fraction) * old_density_matrix)
        print("WARNING: SCF cycle didn't converge")
        return self.density_matrix, self.fock_matrix

    def calculate_energy_scf(self):
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
        self.energy_scf = np.einsum('pq,pq', self.hamiltonian_matrix + self.fock_matrix,
                            self.density_matrix)
        return self.energy_scf

    def calculate_density_matrix(self):
        '''Returns the 1-electron density matrix defined by the input Fock matrix.

            Parameters
        ------------
        fock_matrix : np.array

        Return
        ------------
        density_matrix : np.array


        '''
        num_occ = (ionic_charge // 2) * np.size(self.fock_matrix,
                                                0) // orbitals_per_atom
        orbital_energy, orbital_matrix = np.linalg.eigh(self.fock_matrix)
        occupied_matrix = orbital_matrix[:, :num_occ]
        self.density_matrix = occupied_matrix @ occupied_matrix.T
        return self.density_matrix

    def calculate_fock_matrix(self):
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
        self.fock_matrix = self.hamiltonian_matrix.copy()
        sefl.fock_matrix += 2.0 * np.einsum('pqt,rsu,tu,rs',
                                    self.chi_tensor,
                                    self.chi_tensor,
                                    self.interaction_matrix,
                                    self.density_matrix,
                                    optimize=True)
        self.fock_matrix -= np.einsum('rqt,psu,tu,rs',
                                self.chi_tensor,
                                self.chi_tensor,
                                self.interaction_matrix,
                                self.density_matrix,
                                optimize=True)
        return self.fock_matrix

    def initialize(self):
        self.fock_matrix = self.calculate_fock_matrix(self.hamiltonian_matrix, self.interaction_matrix,
        self.density_matrix, self.chi_tensor)
        self.density_matrix = self.calculate_density_matrix(self.fock_matrix)

    def kernel(self):
        self.initialize()
        self.density_matrix, self.fock_matrix = self.scf_cycle()
        self.energy_ion = self.calculate_energy_ion(self.atomic_coordinates)
        self.total_energy = energy_ion + energy_scf
        return self.total_energy
