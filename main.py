import numpy as np
from scipy.integrate import quad
import sympy as sp

x = sp.symbols('x')

func_str = input("Enter the function h(x): ")
func = sp.sympify(func_str, evaluate=False)

func_prime = sp.diff(func, x)

func_numeric = sp.lambdify(x, func, 'numpy')
func_prime_numeric = sp.lambdify(x, func_prime, 'numpy')

def integrand(x):
    return 2 * np.pi * func_numeric(x) * np.sqrt(1 + func_prime_numeric(x)**2)

c = float(input("Enter the lower bound c: "))
d = float(input("Enter the upper bound d: "))

S, _ = quad(integrand, c, d)

print(f"The surface area S is: {S}")
