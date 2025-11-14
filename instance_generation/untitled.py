import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, norm


# Evaluate pdf on a grid
x = np.linspace(6, 20, 500)


def process_demand_distribution(x,
                                alphas = (-4, 0, 4),
                                locs = (8.5, 12.5, 17.5),
                                scales = (2, 1, 2),
                                weights = (0.45, 0.1, 0.45)):
    pdf_morning = skewnorm.pdf(x, a = alphas[0], loc = locs[0], scale = scales[0])
    pdf_noon = skewnorm.pdf(x, a = alphas[1], loc = locs[1], scale = scales[1])
    pdf_evening = skewnorm.pdf(x, a = alphas[2], loc = locs[2], scale = scales[2])
    pdf_demand = [weights[0] * val_morning + weights[1] * val_noon + weights[2] * val_evening 
                                    for val_morning, val_noon, val_evening in zip(pdf_morning, pdf_noon, pdf_evening)]
    return pdf_demand


pdf_demand = process_demand_distribution(x,
                                         alphas = (3, 0, 3),
                                         locs = (8, 12.5, 17),
                                         scales = (1, 1, 1),
                                         weights = (0.475, 0.05, 0.475))

# Plot
plt.figure(figsize=(8, 4))
plt.plot(x, pdf_demand)
plt.title("Skew Normal Distribution")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.show()