
# Hybrid Optimization GUI Application

This Python application utilizes the `tkinter` library to provide a graphical user interface for performing hybrid optimization on mathematical functions. It uses gradient descent and the Newton-Raphson method depending on the condition of the Hessian matrix.

## Features

- Define any mathematical function with multiple variables.
- Specify initial guesses, tolerance, and maximum iterations for the optimization process.
- Automatic handling of divergences and convergence notification.

## Dependencies

- Python 3.x
- `numpy`
- `sympy`
- `tkinter`

## Setup and Run

1. **Ensure Python 3.x is installed on your system.**
2. **Install required packages if not already installed:**
   ```bash
   pip install numpy sympy
   ```
3. **Save the script in a `.py` file.**
4. **Run the script:**
   ```bash
   python cGUI.py
   ```
5. **Enter the function expression, initial guesses, tolerance, and maximum iterations in the GUI.**

## GUI Components

- **Function Expression:** Enter the function with variables as x1, x2, etc.
- **Initial Guesses:** Provide initial values separated by commas.
- **Tolerance (epsilon):** Set the convergence criterion.
- **Maximum Iterations:** Set the limit to the number of iterations.

## Usage

Upon running the optimization, the results including the optimal values, function value at minimum, and a status message will be displayed in the GUI.

