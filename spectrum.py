def get_spectrum(solver):
    energies = list()
    for pair in solver.eigensystems:
        for energy in pair[:][0]:
            energies.append(energy)
    return sorted(energies)
