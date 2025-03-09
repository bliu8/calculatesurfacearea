from flask import Flask, render_template, request
import numpy as np
from scipy.integrate import quad
import sympy as sp
import plotly.graph_objs as go
import plotly.io as pio
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    func_str = request.form['function']
    a = float(request.form['lower_bound'])
    b = float(request.form['upper_bound'])
    axis = request.form['axis']

    x = sp.symbols('x')
    func = sp.sympify(func_str, evaluate=False)
    func_prime = sp.diff(func, x)

    func_numeric = sp.lambdify(x, func, 'numpy')
    func_prime_numeric = sp.lambdify(x, func_prime, 'numpy')

    if axis == 'x':
        def integrand(x):
            return 2 * np.pi * np.abs(func_numeric(x)) * np.sqrt(1 + func_prime_numeric(x)**2)
        S, _ = quad(integrand, a, b)
    elif axis == 'y':
        def integrand(x):
            return 2 * np.pi * np.abs(x) * np.sqrt(1 + func_prime_numeric(x)**2)
        S, _ = quad(integrand, a, b)

    x_vals = np.linspace(a, b, 400)
    y_vals = func_numeric(x_vals)

    X, Theta = np.meshgrid(x_vals, np.linspace(0, 2 * np.pi, 100))
    Y = func_numeric(X)

    fig = go.Figure()

    if axis == 'x':
        Z = Y * np.cos(Theta)
        Y_rot = Y * np.sin(Theta)
        fig.add_trace(go.Surface(z=Z, y=Y_rot, x=X))
    elif axis == 'y':
        Z = X * np.cos(Theta)
        X_rot = X * np.sin(Theta)
        fig.add_trace(go.Surface(x=X_rot, y=Y, z=Z))

    max_val = max(np.max(x_vals), np.max(y_vals))
    fig.add_trace(go.Scatter3d(x=[0, max_val], y=[0, 0], z=[0, 0], mode='lines', name='x-axis', line=dict(color='red', width=5)))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, max_val], z=[0, 0], mode='lines', name='y-axis', line=dict(color='green', width=5)))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, max_val], mode='lines', name='z-axis', line=dict(color='blue', width=5)))

    fig.update_layout(
        title=f'Solid of Revolution around {axis}-axis',
        autosize=True,
        margin=dict(l=65, r=50, b=65, t=90),
        scene=dict(aspectmode='data')
    )

    plot_path = 'static/solid_of_revolution.html'
    pio.write_html(fig, file=plot_path, auto_open=False)

    return render_template('result.html', surface_area=S, plot_path=plot_path)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
