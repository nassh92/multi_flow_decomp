import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm, norm
import math


"""
Generate the peak demand in relation to the hour of the day
"""


def process_demand_weights_distribution(time_periods,
                                alphas = (3, 0, 3),
                                locs = (8, 12.5, 17),
                                scales = (1, 1, 1),
                                weights = (0.475, 0.05, 0.475)):
    pdf_morning = skewnorm.pdf(time_periods, a = alphas[0], loc = locs[0], scale = scales[0])
    pdf_noon = skewnorm.pdf(time_periods, a = alphas[1], loc = locs[1], scale = scales[1])
    pdf_evening = skewnorm.pdf(time_periods, a = alphas[2], loc = locs[2], scale = scales[2])
    pdf_demand = [weights[0] * val_morning + weights[1] * val_noon + weights[2] * val_evening 
                                    for val_morning, val_noon, val_evening in zip(pdf_morning, pdf_noon, pdf_evening)]
    max_demand = max(pdf_demand)
    weights_demand = [val/max_demand for val in pdf_demand]
    return weights_demand



def process_desired_flow_values(weight_demand,
                                maximal_congestion_rate,
                                maximal_flow_values):
    return [math.ceil(weight_demand * maximal_congestion_rate * flow_val) for flow_val in maximal_flow_values]


if __name__ == "__main__":
    # Evaluate pdf on a grid
    x = np.linspace(6, 20, (20 - 6) * 60)

    # 
    """pdf_demand = process_demand_weights_distribution(x,
                                            alphas = (3, 0, 3),
                                            locs = (8, 12.5, 17),
                                            scales = (1, 1, 1),
                                            weights = (0.475, 0.05, 0.475))"""
    pdf_demand = process_demand_weights_distribution(x)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(x, pdf_demand)
    plt.title("Skew Normal Distribution")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.show()