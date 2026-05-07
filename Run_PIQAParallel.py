import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # ensures 3D plotting is available
from nqueens import RunPIQA
import matplotlib
import time
from itertools import product
import matplotlib.cm as cm
from matplotlib.colors import Normalize
matplotlib.use('TkAgg')

start = time.time()

# Test values for gamma and Trotter numbers.
# gamma_values = np.linspace(0.1, 1.0, 5)
# P_values = np.array([1, 2, 3, 4, 5])

# Define ranges for initial Gamma and Trotter number P.
# gamma_values = np.linspace(10, 80, 10)  # 10 values from 10 to 200.
gamma_values = np.array([120, 110, 100, 90, 80, 70, 60, 50, 40, 30, 20, 10])
P_values = np.array([24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2])       # different Trotter numbers.

r = 40               # number of repeats for averaging.

tasks = [(x,y) for x, y in product(gamma_values, P_values) for _ in range(r)]

# Container to store the average final energy for each (initial Gamma, P) pair.
final_energy_matrix = np.zeros((len(P_values), len(gamma_values)))

def Annealing_Thing(task):
    G, P = task
    # Fixed parameters.
    N = 8
    T_fixed = 0.2  # classical temperature used in acceptance.
    monte_carlo_steps = 8  # one sweep (8 moves) per annealing step.
    annealing_steps = 100  # number of annealing steps in each run.
    A, B, C = 2, 2, 2
    G_final = 0.001

    _, final_energy, _ = RunPIQA(N, P, 1.6/P, monte_carlo_steps, annealing_steps,
                                 G, G_final, A, B, C)
    print(".")
    print(multiprocessing.current_process().name)
    print("P:",P)
    print("G:", G)
    print("E:", final_energy)
    return (P, G, final_energy)

def main():
    with multiprocessing.Pool(processes=8) as pool:
        results = pool.map(Annealing_Thing, tasks)

    pool.close()
    pool.join()

    print("results")
    print(results)

    return results

if __name__ == "__main__":
    data = main()

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Convert to numpy array for convenience
    data = np.array(data)

    # Extract G, P, Final_Energy
    P = data[:, 0]
    G = data[:, 1]
    Final_Energy = data[:, 2]

    # Compute the average Final_Energy for each (G, P) pair
    unique_pairs = list(set(zip(P, G)))
    average_energy = {pair: np.mean(Final_Energy[(P == pair[0]) & (G == pair[1])]) for pair in unique_pairs}

    # Compute average final energy for each (P, G) pair
    for i, P_val in enumerate(P_values):
        for j, G_val in enumerate(gamma_values):
            mask = (P == P_val) & (G == G_val)
            if np.any(mask):
                final_energy_matrix[i, j] = np.mean(Final_Energy[mask])
            else:
                final_energy_matrix[i, j] = np.nan  # Handle missing data gracefully

    # Save the computed data.
    np.savez("3dbar_ptvaries_run6.npz", gamma_values=gamma_values, P_values=P_values, final_energy_matrix=final_energy_matrix)
    print("Data saved to 3dbar_ptvaries_run6.npz")

    # Extract values for plotting
    P_vals = np.array([pair[0] for pair in unique_pairs])
    G_vals = np.array([pair[1] for pair in unique_pairs])
    E_vals = np.array([average_energy[pair] for pair in unique_pairs])

    print("Time:", time.time()-start)

    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    norm = Normalize(vmin=min(E_vals), vmax=max(E_vals))
    colors = cm.jet(norm(E_vals))

    # Create the bars
    ax.bar3d(G_vals, P_vals, np.zeros_like(E_vals), 10, 2, E_vals, shade=True, color=colors)

    # Label axes
    ax.set_xlabel('Gamma Value G')
    ax.set_ylabel('Trotter Slice P')
    ax.set_zlabel('Average Final Energy')

    # Show the plot
    plt.title('3D Histogram of Average Final Energy')
    plt.show()
    # Optionally invert the y-axis if that fits your data visualization.
    #ax.invert_yaxis()

    plt.show()
