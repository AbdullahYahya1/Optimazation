import tkinter as tk
from tkinter import ttk, messagebox
import sympy as sp
import numpy as np

# Redefined math functions
def gradient_descent_step(xn, grad, learning_rate=0.01):
    return xn - learning_rate * grad

def is_positive_definite(matrix):
    num_matrix = np.array(matrix).astype(np.float64)
    eigenvalues = np.linalg.eigvals(num_matrix)
    return np.all(eigenvalues > 0)

def hybrid_optimization(f_expr, initial_guesses, epsilon, max_iter, divergence_threshold=-1e6):
    num_vars = len(initial_guesses)
    vars = sp.symbols(' '.join([f'x{i+1}' for i in range(num_vars)]))
    f = sp.sympify(f_expr)
    xn = np.array(initial_guesses, dtype=float)

    for n in range(max_iter):
        grad = np.array([float(f.diff(v).subs(dict(zip(vars, xn)))) for v in vars])
        hessian = sp.Matrix([[f.diff(i).diff(j) for j in vars] for i in vars]).subs(dict(zip(vars, xn)))
        function_value = float(f.subs(dict(zip(vars, xn))))

        if function_value < divergence_threshold:
            messagebox.showwarning("Warning", "Optimization may be diverging to negative infinity.")
            return xn, function_value, f"Divergence detected at iteration {n}"

        if np.linalg.norm(grad) < epsilon:
            return xn, function_value, f"Convergence reached at iteration {n}"

        if is_positive_definite(hessian):
            try:
                hessian_inv = np.array(hessian.inv()).astype(np.float64)
                xn_new = xn - hessian_inv.dot(grad)
            except (np.linalg.LinAlgError, ValueError):
                xn_new = gradient_descent_step(xn, grad)
        else:
            xn_new = gradient_descent_step(xn, grad)

        xn = xn_new

    return xn, f.subs(dict(zip(vars, xn))).evalf(), "Maximum iterations reached without convergence."

# GUI components
def run_optimization():
    f_expr = entry_function.get().replace('^', '**')
    initial_guess_str = entry_initial_guesses.get()
    initial_guesses = list(map(float, initial_guess_str.split(','))) if initial_guess_str else []
    epsilon = float(entry_tolerance.get())
    max_iter = int(entry_max_iterations.get())
    divergence_threshold =-10000

    num_vars = len(sp.sympify(f_expr).free_symbols)

    if len(initial_guesses) != num_vars:
        messagebox.showerror("Error", "The number of initial guesses does not match the number of variables in the function.")
        return

    result, function_value, message = hybrid_optimization(f_expr, initial_guesses, epsilon, max_iter, divergence_threshold)
    result_display.config(state=tk.NORMAL)
    result_display.delete(1.0, tk.END)
    result_display.insert(tk.END, f"Result: {result}\nFunction value at minimum: {function_value}\n{message}")
    result_display.config(state=tk.DISABLED)

# Main window
root = tk.Tk()
root.title("Hybrid Optimization GUI")
root.configure(bg="#f0f0f0")

# Styling
style = ttk.Style(root)
style.theme_use("clam")
style.configure("TButton", font=('Helvetica', 10), padding=6)
style.configure("TEntry", font=('Helvetica', 10))
style.configure("TLabel", font=('Helvetica', 10), background="#f0f0f0")

# Layout adjustments
pad_options = {'padx': 5, 'pady': 5}

# Labels and entries
ttk.Label(root, text="Function Expression (in terms of x1, x2, etc.):").grid(row=0, column=0, sticky="w", **pad_options)
entry_function = ttk.Entry(root, width=50)
entry_function.grid(row=0, column=1, **pad_options)

ttk.Label(root, text="Initial Guesses (comma-separated):").grid(row=1, column=0, sticky="w", **pad_options)
entry_initial_guesses = ttk.Entry(root, width=50)
entry_initial_guesses.grid(row=1, column=1, **pad_options)

ttk.Label(root, text="Tolerance (epsilon):").grid(row=2, column=0, sticky="w", **pad_options)
entry_tolerance = ttk.Entry(root, width=50)
entry_tolerance.grid(row=2, column=1, **pad_options)

ttk.Label(root, text="Maximum Iterations:").grid(row=3, column=0, sticky="w", **pad_options)
entry_max_iterations = ttk.Entry(root, width=50)
entry_max_iterations.grid(row=3, column=1, **pad_options)

run_button = ttk.Button(root, text="Run Optimization", command=run_optimization)
run_button.grid(row=5, column=0, columnspan=2, **pad_options)

# Result display
result_display = tk.Text(root, height=10, width=60, state=tk.DISABLED, font=('Helvetica', 10))
result_display.grid(row=6, column=0, columnspan=2, pady=10)

# Run the application
root.mainloop()
