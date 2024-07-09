import os
import numpy as np
import matplotlib.pyplot as plt


def load_matrix(path):
    """Charge une matrice à partir d'un fichier texte."""
    return np.loadtxt(path, delimiter=" ")


def gram_schmidt(A):
    """Effectue la décomposition QR pour l'orthonormalisation de la matrice A."""
    Q, _ = np.linalg.qr(A)
    return Q


def build_bootstrap_sample(qd_decompositions, orthonormal_basis=True):
    """Génère des échantillons bootstrap à partir des décompositions propres données."""
    n = qd_decompositions[0][0].size
    n_bootstrap = 2**n
    samples = []
    for k in range(n_bootstrap):
        binary_str = bin(k)[2:].zfill(n)
        mask = np.array([int(bit) for bit in binary_str])
        eigenvecs = np.column_stack(
            [qd_decompositions[mask[i]][1][:, i] for i in range(len(mask))]
        )
        eigenvals = [qd_decompositions[mask[i]][0][i] for i in range(len(mask))]
        if orthonormal_basis:
            eigenvecs = gram_schmidt(eigenvecs)
        sample = np.dot(np.dot(eigenvecs, np.diag(eigenvals)), np.transpose(eigenvecs))
        samples.append(sample)
    return samples


def collect_statistics(samples):
    """Calcule des statistiques sur les échantillons, y compris les matrices de covariance."""
    covariance_matrices = [np.linalg.inv(sample) for sample in samples]
    FoM_stat = [
        np.sqrt(1.0 / (np.linalg.det(cov[2:4, 2:4]))) for cov in covariance_matrices
    ]
    average_matrix = np.average(samples, axis=0)
    np.savetxt("Fisher_average_FoM.txt", average_matrix)
    return average_matrix, covariance_matrices, FoM_stat


# def calculate_variances_of_variances(covariance_matrices):
#    """Calcule la variance des variances pour chaque paramètre."""
#    variances = [np.diag(cov) for cov in covariance_matrices]
#    variances_of_variances = np.var(variances, axis=0)
#    return variances_of_variances


def calculate_variances_of_variances(covariance_matrices):
    """Calcule la variance des variances pour chaque paramètre."""
    variances = [np.diag(cov) for cov in covariance_matrices]
    confidence_level = 0.68
    lower = np.quantile(variances, (1.0 - confidence_level) * 0.5, axis=0)
    middle = np.median(variances, axis=0)
    upper = np.quantile(variances, 1.0 - (1.0 - confidence_level) * 0.5, axis=0)
    return lower, middle, upper
 

def plot_FoM_histogram(FoM_stat):
    """Affiche un histogramme du Figure of Merit."""
    plt.hist(FoM_stat, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("FoM Histogram")
    plt.show()


def save_results(F_final):
    """Sauvegarde la matrice de Fisher finale et sa covariance."""
    F_final_inverse = np.linalg.inv(F_final)
    np.savetxt("F_final.txt", F_final)
    np.savetxt("F_final_inverse.txt", F_final_inverse)


# Main processing sequence
matrix_paths = ["./F1.txt", "./F2.txt"]
matrices = [load_matrix(path) for path in matrix_paths]
qd_decompositions = [np.linalg.eig(matrix) for matrix in matrices]
samples = build_bootstrap_sample(qd_decompositions)
average_matrix, covariance_matrices, FoM_stat = collect_statistics(samples)
covariance_average = np.linalg.inv(average_matrix)
variances_of_variances = calculate_variances_of_variances(covariance_matrices)

std = np.sqrt(variances_of_variances)

# Display results
print("Standard deviation for each parameter:")
print("parameter num, lower bound < estimated value < upper bound")
for i in range(0, 7):
    print(i, std[0][i], "<", std[1][i], "<", std[2][i])
plot_FoM_histogram(FoM_stat)
save_results(average_matrix)

# Custom command for environment
os.system("printFoM F_final.txt OPT GCph 0 0 F N")
