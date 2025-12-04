# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 09:23:54 2025

@author: Sergio Mendiola Arraez
"""
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Definición de funciones
# ==========================

def g1(x1, x2):
    return 2*x1**2 - 2*x2 - 1

def g2(x1, x2):
    return -x1 + 2*x2**2 - 1

def f(x1, x2):
    return 2*x1**3 - 2*x2**2 + x2

# Gradientes
def grad_f(x1, x2):
    return np.array([6*x1**2, 1 - 4*x2])

def grad_g1(x1, x2):
    return np.array([4*x1, -2])

def grad_g2(x1, x2):
    return np.array([-1, 4*x2])

# ==========================
# Función para imprimir
# ==========================

def print_candidato(tipo, x1, x2):
    factible = (g1(x1,x2) <= 1e-6 and g2(x1,x2) <= 1e-6)
    print(f"{tipo:15s}  x = ({round(x1,3)}, {round(x2,3)})   Factible = {factible}")

# ==========================
# Lista de candidatos válidos
# ==========================

candidatos = []

# ============================================================
# 1) Caso sin restricciones activas (λ1=0, λ2=0)
# grad f = 0  →  x1=0,  x2=1/4
# ============================================================

x1 = 0.0
x2 = 0.25
candidatos.append(("Ninguna activa", x1, x2))

# ============================================================
# 2) Caso g1 activa (λ1>0, λ2=0)
# 4x1² - 3x1 - 3 = 0
# ============================================================

roots1 = np.roots([4, -3, -3])

for r in roots1:
    if abs(r.imag) < 1e-10:
        x1 = r.real
        x2 = (1 + 3*x1) / 4   # de g1=0

        A = grad_g1(x1,x2).reshape(2,1)
        b = -grad_f(x1,x2)
        lam1 = np.linalg.lstsq(A, b, rcond=None)[0][0]

        if lam1 >= -1e-6:
            candidatos.append(("g1 activa", x1, x2))

# Solución especial x1=0 → x2=-1/2
x1 = 0.0
x2 = -0.5
lam1 = (1 - 4*x2)/2
if lam1 >= -1e-6:
    candidatos.append(("g1 activa", x1, x2))

# ============================================================
# 3) Caso g2 activa (λ2>0, λ1=0)
# P(x1) = (x1+1)*(4 - 24x1²)² - 2 = 0
# ============================================================

P = np.poly1d([1,1]) * np.poly1d([576,0,-192,0,16]) - 2
roots2 = np.roots(P)

for r in roots2:
    if abs(r.imag) < 1e-6:
        x1 = r.real
        denom = 4 - 24*x1**2
        if abs(denom) < 1e-12:
            continue
        
        x2 = 1/denom

        A = grad_g2(x1,x2).reshape(2,1)
        b = -grad_f(x1,x2)
        lam2 = np.linalg.lstsq(A, b, rcond=None)[0][0]

        if lam2 >= -1e-6:
            candidatos.append(("g2 activa", x1, x2))

# ============================================================
# 4) Caso ambas activas (λ1>0, λ2>0)
# Q(x1) = 2 x1^4 - 2 x1^2 - x1 - 1/2 = 0
# ============================================================

roots12 = np.roots([2, 0, -2, -1, -0.5])

for r in roots12:
    if abs(r.imag) < 1e-6:
        x1 = r.real
        x2 = x1**2 - 0.5

        A = np.column_stack((grad_g1(x1,x2), grad_g2(x1,x2)))
        b = -grad_f(x1,x2)

        try:
            lam1, lam2 = np.linalg.solve(A, b)

            if lam1 >= -1e-6 and lam2 >= -1e-6:
                candidatos.append(("Ambas activas", x1, x2))

        except np.linalg.LinAlgError:
            pass

# ==========================
# Impresión final
# ==========================

print("\nCandidatos KKT (3 decimales) y factibilidad:\n")
for tipo, x1, x2 in candidatos:
    print_candidato(tipo, x1, x2)

# ==========================
# Evaluar candidatos y óptimo
# ==========================

vals_f = np.array([f(x1, x2) for _, x1, x2 in candidatos])

factibles_idx = [i for i, (_, x1, x2) in enumerate(candidatos)
                 if g1(x1, x2) <= 1e-6 and g2(x1, x2) <= 1e-6]

if len(factibles_idx) > 0:
    vals_f_fact = vals_f[factibles_idx]
    idx_min_local = factibles_idx[np.argmin(vals_f_fact)]
    x1_opt, x2_opt = candidatos[idx_min_local][1], candidatos[idx_min_local][2]
else:
    x1_opt = x2_opt = None

# ==========================
# Malla para el dibujo
# ==========================

x1_min, x1_max = -2, 2
x2_min, x2_max = -2, 2

X1, X2 = np.meshgrid(
    np.linspace(x1_min, x1_max, 400),
    np.linspace(x2_min, x2_max, 400)
)

G1 = g1(X1, X2)
G2 = g2(X1, X2)
F  = f(X1, X2)

factible_mask = (G1 <= 0) & (G2 <= 0)

# ==========================
# Figura
# ==========================

plt.figure(figsize=(8,6))

plt.contourf(X1, X2, factible_mask, levels=[-0.5, 0.5, 1.5],
             colors=['white', 'lightblue'], alpha=0.5)

# OJO: paso las etiquetas directamente aquí
plt.contour(X1, X2, G1, levels=[0], colors='red', linewidths=2, label='g1(x)=0')
plt.contour(X1, X2, G2, levels=[0], colors='green', linewidths=2, label='g2(x)=0')

niveles = np.linspace(min(vals_f)-1, max(vals_f)+1, 10)
plt.contour(X1, X2, F, levels=niveles, colors='gray', linestyles='dotted', alpha=0.8)

for tipo, x1, x2 in candidatos:
    plt.plot(x1, x2, 'ko', markersize=6)
    plt.text(x1+0.03, x2+0.03, tipo, fontsize=8)

if x1_opt is not None:
    plt.plot(x1_opt, x2_opt, 'rx', markersize=10, mew=2, label='Óptimo global')

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title('Conjunto factible, curvas de nivel de f y puntos KKT')
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()
