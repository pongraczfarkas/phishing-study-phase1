import numpy as np
from scipy import stats

# These are the 5 'delta_f1' values from your Random Forest Seed 9 JSON
deltas = [-0.004327443202308046, 0.008766322710254792, 0.008403361344537785, 0.0, 0.0]

# 1. Calculate Mean Difference
mean_delta = np.mean(deltas)
# Result: -0.00928
print(mean_delta)

std_dev = np.std(deltas, ddof=1)
n = 5
se = std_dev / np.sqrt(n)
print(se)
# Result: 0.01105

# 3. Calculate t-statistic
t_stat = mean_delta / se
print(t_stat)
# Result: -0.84

# 4. Calculate p-value (Two-tailed)
df = n - 1  # Degrees of freedom (4)
p_val = stats.t.sf(np.abs(t_stat), df) * 2
print(p_val)
# Result: 0.448