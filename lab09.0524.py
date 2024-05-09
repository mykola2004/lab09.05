import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def mean_in_confidence_interval(u, sigma, num_samples=10, confidence_level=0.95):
    random_numbers = np.random.normal(loc=u, scale=sigma, size=num_samples)
    
    sample_mean = np.mean(random_numbers)
    standard_error = stats.sem(random_numbers)
    
    confidence_interval = stats.t.interval(
        confidence_level, num_samples - 1, loc=sample_mean, scale=standard_error
    )
    
    if confidence_interval[0] <= u <= confidence_interval[1]:
        result = 1
    else:
        result = 0

    return result, confidence_interval, sample_mean

results_array = []

u = 0  
sigma = 1 
num_experiments = 10000
confidence_level = 0.9
num_samples = 10


for _ in range(num_experiments):
    result, conf_interval, sample_mean = mean_in_confidence_interval(u, sigma, num_samples, confidence_level)
    results_array.append(result)

probability = sum(results_array) / len(results_array)
print("Probability of mean being inside the inteval: ", probability)

labels = ['TAK', 'NIE']
counts = [results_array.count(1), results_array.count(0)]

plt.bar(labels, counts, color=['blue', 'red'])
plt.xlabel('Condition')
plt.ylabel('Count')
plt.title('Frequency of Mean within and outside Confidence Interval')
plt.show()

print(f"Probability of the mean being within the confidence interval: {probability}")

