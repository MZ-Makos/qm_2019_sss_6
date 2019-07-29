#include <Eigen/Dense>
#include <map>
#include <algorithm>
#include <vector>

int atom(int ao_index)
{
    int orbitals_per_atom=4;
    return ao_index / orbitals_per_atom;
}

int indexOf(std::vector<std::string> my_list, std::string element)
{
    for(size_t i=0;i<sizeof(my_list);i++)
    {
        if(element==my_list[i])
        {
            return i;
        }
    }
    return -1;
}

bool list_contains(std::vector<std::string>  my_list, std::string element)
{
    bool found = 0;
    for(size_t i=0;i<sizeof(my_list);i++)
    {
        if(element==my_list[i])
        {
            found = 1;
        }
    }
    return found;
}

std::string orb(int ao_index, std::vector<std::string> orbital_types)
{
    int orbitals_per_atom=4;
    int orb_index = ao_index % orbitals_per_atom;
    return orbital_types[orb_index];
}

int ao_index(int atom_p, std::string orb_p, std::vector<std::string>  orbital_types)
{
        int orbitals_per_atom=4;
        int p = atom_p * orbitals_per_atom;
        p += indexOf(orbital_types,orb_p);
        return p;
}

float chi_on_atom(std::string o1, std::string o2, std::string o3, std::vector<std::string> p_orbitals,double dipole)
{
    if(o1 == o2 && o3 == "s")
        return 1.0;
    bool o3_in_p_orbitals = list_contains(p_orbitals,o3);
    if((o1 == o3 && o3_in_p_orbitals) && (o2 == "s"))
        return dipole;
    if ((o2 == o3 && o3_in_p_orbitals) && (o1 == "s"))
        return dipole;
    return 0.0;
}

Eigen::MatrixXd fast_fock_matrix(Eigen::MatrixXd hamiltonian, Eigen::MatrixXd interaction, Eigen::MatrixXd rho,double dipole)
{
    std::vector<std::string> orbital_types ={"s","px", "py", "pz"};
    std::vector<std::string>  p_orbitals = {"px", "py", "pz"};
    size_t ndof = hamiltonian.size();
    Eigen::MatrixXd fock_mat = hamiltonian;
    //Hartree potential term
    for(int p=0;p<ndof;p++)
    {
        for(auto orb_q: orbital_types)
        {
            int q = ao_index(atom(p), orb_q,orbital_types); // p & q on same atom
            for(auto orb_t : orbital_types)
            {
                int t = ao_index(atom(p), orb_t,orbital_types); // p & t on same atom
                float chi_pqt = chi_on_atom(orb(p,orbital_types), orb_q, orb_t,p_orbitals,dipole);
                for(int r=0;r<ndof;r++)
                {
                    for(auto orb_s : orbital_types)
                    {
                        int s = ao_index(atom(r), orb_s,orbital_types); // r & s on same atom
                        for(auto orb_u : orbital_types)
                        {
                            int u = ao_index(atom(r), orb_u,orbital_types); // r & u on same atom
                            float chi_rsu = chi_on_atom(orb(r,orbital_types), orb_s, orb_u, p_orbitals,dipole);
                            fock_mat(p,q) += 2.0 * chi_pqt * chi_rsu * interaction(t,u) * rho(r,s);
                        }
                    }
                }
            }
        }   
    }
    //Fock exchange term
    for(int p=0;p<ndof;p++)
    {
        for(auto orb_s: orbital_types)
        {
            int s = ao_index(atom(p), orb_s,orbital_types); // p & s on same atom
            for(auto orb_u : orbital_types)
            {
                int u = ao_index(atom(p), orb_u,orbital_types); //p & u on same atom
                float chi_psu = chi_on_atom(orb(p,orbital_types), orb_s, orb_u,p_orbitals,dipole);
                for(int q=0; q<ndof; q++)
                {
                    for(auto orb_r : orbital_types)
                    {
                        int r = ao_index(atom(q), orb_r,orbital_types); // q & r on same atom
                        for(auto orb_t : orbital_types)
                        {
                            int t = ao_index(atom(q), orb_t,orbital_types); //q & t on same atom
                            float chi_rqt = chi_on_atom(orb_r, orb(q,orbital_types), orb_t, p_orbitals,dipole);
                            fock_mat(p,q) -= chi_rqt * chi_psu * interaction(t,u) * rho(r,s);
                        }
                    }
                }
            }
        }
    }
    return fock_mat;

}

int main(void)
{
    return 0;
}