import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

cm_data = np.array([[81, 19],
                       [4, 96]])

# Labels
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_counts = ["{0:0.0f}".format(value) for value in cm_data.flatten()]

row_sums = cm_data.sum(axis=1)
group_percentages = ["{0:.2%}".format(value) for value in (cm_data.T / row_sums).T.flatten()]

labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2,2)

# --- PLOTTING ---
plt.figure(figsize=(6, 5))
ax = sns.heatmap(cm_data, annot=labels, fmt='', cmap='Blues', cbar=False,
                 xticklabels=['Legitimate', 'Phishing'],
                 yticklabels=['Legitimate', 'Phishing'])

plt.title('Confusion Matrix: LLM Zero-Shot (Condition B)', fontsize=14)
plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.show()