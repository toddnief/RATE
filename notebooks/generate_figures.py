import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open('/net/projects/veitch/prompt_distributions/results/sft_experiment_results_gpt2_20240518_103825.json') as f:
    data = json.load(f)

test_statistic = data['test_statistic']

df = pd.DataFrame({'test_statistic': [test_statistic]})

sns.histplot(df['test_statistic'], bins=10, kde=True)

plt.title('Histogram of Test Statistic')
plt.xlabel('Test Statistic')
plt.ylabel('Frequency')

plt.savefig('test_statistic_histogram.png', dpi=300)

plt.show()
