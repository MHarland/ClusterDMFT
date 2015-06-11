from numpy import array, polyval, polyfit

def quasiparticle_residue(sigma, n_freq, spin = 'up', orb = (0, 0)):
    mesh = array([w.imag for w in sigma.mesh])[0: n_freq]
    sigma_values = sigma[spin].data[0: n_freq, orb[0], orb[1]].real
    fit_coefficients = polyfit(mesh, sigma_values, len(mesh) - 1)
    derivative_coeff = __derivative(fit_coefficients)
    z = 1 / (1 - derivative_coeff[-1])
    return z
    
def __derivative(polynomial_coefficients):
    deg = len(polynomial_coefficients)
    der = array([(deg - n - 1) * polynomial_coefficients[n] for n in range(deg - 1)])
    return der
