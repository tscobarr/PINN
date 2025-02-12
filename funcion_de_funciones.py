import numpy as np
import sympy as sp
import scipy.special as spc
import math

# 🔒 Diccionario seguro con todas las funciones avanzadas permitidas
SAFE_MATH_FUNCTIONS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "cot": sp.cot, "sec": sp.sec, "csc": sp.csc,
    "exp": sp.exp, "log": sp.ln, "log10": sp.log, "sqrt": sp.sqrt,
    "abs": sp.Abs, "pi": sp.pi, "e": sp.E,

    # 🔢 Funciones especiales
    "bessel": spc.jv, "bessel_y": spc.yn,  # Bessel de 1er y 2do tipo
    "laguerre": spc.eval_laguerre, "hermite": spc.eval_hermite, "legendre": spc.eval_legendre,
    "gamma": sp.gamma, "factorial": sp.factorial,

    # 📉 Operaciones avanzadas
    "diff": sp.diff, "integrate": sp.integrate, "laplacian": lambda f, x: sp.diff(f, x, 2)
}

def generate_function(option="sin"):
    """
    Genera una función matemática avanzada con parámetros personalizables.

    Args:
        option (str): Nombre de la función deseada.

    Returns:
        function: Una función lista para ser usada en una red neuronal.
    """
    x = sp.symbols('x')  # Variable simbólica global
    
    if option == "sin":
        return lambda x: np.sin(x)
    
    elif option == "bessel":
        n = int(input("Introduce el orden `n` de la función de Bessel: "))
        return lambda x: spc.jv(n, x)

    elif option == "laguerre":
        n = int(input("Introduce el grado `n` del polinomio de Laguerre: "))
        return lambda x: spc.eval_laguerre(n, x)

    elif option == "hermite":
        n = int(input("Introduce el grado `n` del polinomio de Hermite: "))
        return lambda x: spc.eval_hermite(n, x)

    elif option == "legendre":
        n = int(input("Introduce el grado `n` del polinomio de Legendre: "))
        return lambda x: spc.eval_legendre(n, x)

    elif option == "derivative":
        expr_str = input("Introduce una función para derivar en términos de `x`: ")
        try:
            expr = sp.sympify(expr_str, locals=SAFE_MATH_FUNCTIONS)
            derivative = sp.diff(expr, x)
            return sp.lambdify(x, derivative, 'numpy')  # Convertimos la derivada a función NumPy
        except Exception as e:
            print("Error al derivar la función:", e)
            return None

    elif option == "integral":
        expr_str = input("Introduce una función para integrar en términos de `x`: ")
        try:
            a = float(input("Introduce el límite inferior de integración `a`: "))
            expr = sp.sympify(expr_str, locals=SAFE_MATH_FUNCTIONS)
            integral = sp.integrate(expr, (x, a, x))  # Integral dependiente de `x`
            return sp.lambdify(x, integral, 'numpy')  # Devuelve una función en `x`
        except Exception as e:
            print("Error al integrar la función:", e)
            return None

    elif option == "laplacian":
        expr_str = input("Introduce una función en términos de `x` para calcular su Laplaciano: ")
        try:
            expr = sp.sympify(expr_str, locals=SAFE_MATH_FUNCTIONS)
            laplacian = sp.diff(expr, x, 2)
            return sp.lambdify(x, laplacian, 'numpy')  # Convertimos el Laplaciano a función NumPy
        except Exception as e:
            print("Error al calcular el Laplaciano:", e)
            return None

    elif option == "custom":
        expr_str = input("Introduce una función matemática en términos de `x` (puedes usar `integrate(f, a, x)`) : ")
        try:
            expr = sp.sympify(expr_str, locals=SAFE_MATH_FUNCTIONS)
            
            # Buscar integrales en la función ingresada
            integrals = expr.atoms(sp.Integral)
            for integral in integrals:
                if len(integral.limits) == 1:  # Asegurar que la integral es definida correctamente
                    var, a, b_limit = integral.limits[0]

                    # Si la variable de integración es `x`, la reemplazamos por `t`
                    if var == x:
                        var = t

                    new_expr = sp.integrate(integral.function.subs(x, t), (var, a, b_limit))
                    expr = expr.subs(integral, new_expr.subs(t, x))  # Volvemos a poner `x` en la solución

            return sp.lambdify(x, expr, 'numpy')  # Convertimos la función simbólica a NumPy
        except Exception as e:
            print("Error en la función ingresada:", e)
            return None

    else:
        raise ValueError("Función no reconocida. Usa 'sin', 'bessel', 'custom'.")


#testeo
f = generate_function("laplacian")  # El usuario puede ingresar su propia función

print (f(2))
if f:
    x_vals = np.linspace(-5, 5, 100)
    y_vals = f(x_vals)

    import matplotlib.pyplot as plt
    plt.plot(x_vals, y_vals, label="Función personalizada")
    plt.legend()
    plt.grid()
    plt.show()
