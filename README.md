# boot_mlmem

This is a Python module for multilevel moderated mediation analysis with contrast testing and bootstrapping for statistical inference, based on the [SPSS MLmem macro](https://njrockwood.com/mlmed) by Nicholas Rockwood. Instead of using Monte Carlo, this implementation uses bootstrapping.
Note that this module is more limited in functionality and was developed for personal use for very specific requirements. Most users with access to SPSS should use the [MLmem macro](https://njrockwood.com/mlmed).

## Purpose

This module assists in complex statistical analyses within a multilevel modeling framework. It specifically performs the following:

    Multilevel Moderated Mediation: Investigates how relationships between an independent variable (X), a dependent variable (Y), and parallel mediating variables (M1, M2, M3) are influenced by moderators at the cluster level (level-2).
    Contrast Analysis: Examines and quantifies meaningful differences in indirect effects across various moderator levels.
    Bootstrapping: Offers robust statistical inference and estimation of confidence intervals for calculated effects.

## Dependencies

    pandas
    statsmodels
    scipy.stats
    patsy

## Usage

```
import boot_mlmem as bml
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import scipy.stats as stats # For bootstrap inferences
from patsy import dmatrices  # For formula creation 

# Load your dataset (example using Pandas)
data = pd.read_csv('your_data.csv') 

# Specify model parameters
results = bml.boot_mlmem(dataset=data,
                         x='independent_var',
                         m1='mediator1',
                         y='dependent_var',
                         cluster='cluster_id',
                         m2='mediator2',  # Optional
                         m3='mediator3',  # Optional
                         modS1='level2_moderator1', # Optional
                         modS2='level2_moderator2', # Optional
                         modS='level2_moderator3', # Optional
                         randx=[0, 1, 0], # Example: X random, M1 random
                         randm=[1, 0],    # Example: M1-Cluster random intercept
                         n_boot=2000,
                         centering_type='group_mean' 
                        )

# Access results 
model = results[0]
indirect_effects = results[1]
contrasts = results[2]

print(model.summary()) 
print(indirect_effects)
print(contrasts)
```

## Parameters

    dataset: Your input data object (statsmodels format or DataFrame).
    x: Independent variable name.
    m1, m2, m3: Mediator variable names.
    y: Dependent variable name.
    cluster: Variable indicating level-2 clustering.
    modS1, modS2, modS: Level-2 moderator variable names.
    randx: List controlling random effects for X paths.
    randm: List controlling random effects for mediator paths.
    n_boot: Number of bootstrap iterations.
    centering_type: 'group_mean' or 'grand_mean' centering.

## Output

    model: Fitted statsmodels mixed linear model object.
    indirect_effects: Dictionary containing calculated indirect effects under various moderator combinations.
    contrasts: Dictionary containing computed contrasts.

## Example

```
import boot_mlmem as bml
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import scipy.stats as stats # For bootstrap inferences
from patsy import dmatrices  # For formula creation 

# Set a seed for reproducibility of the random data
seed(123)

# **1. Simulate Sample Data** 
n_clusters = 50 # Number of level-2 clusters 
n_per_cluster = 30 # Observations per cluster

# Generate some random example data with relationships, moderators, and clustering
data = pd.DataFrame({
    'x': normal(loc=5, scale=1, size=n_clusters * n_per_cluster),
    'm1': normal(loc=3, scale=0.8, size=n_clusters * n_per_cluster),
    'm2': normal(loc=2, scale=0.5, size=n_clusters * n_per_cluster),
    'y': normal(loc=10, scale=2, size=n_clusters * n_per_cluster), 
    'cluster': [i for i in range(1, n_clusters + 1) for _ in range(n_per_cluster)],  
    'modS1': normal(loc=0, scale=2, size=n_clusters),
    'modS2': normal(loc=1, scale=1, size=n_clusters)
})

# Modify relationships based on variables:
data['m1'] = data['x'] * 0.6 + data['modS1'] * 0.3 + normal(loc=0, scale=0.5, size=n_clusters * n_per_cluster)
data['m2'] = data['x'] * 0.4 - data['modS1'] * 0.2 + normal(loc=0, scale=0.6, size=n_clusters * n_per_cluster)
data['y'] = data['x'] * 0.2 + data['m1'] * 0.7 + data['m2'] * 0.3 + data['modS2'] * 0.4 + normal(loc=0, scale=1, size=n_clusters * n_per_cluster)

# **2. Run the Multilevel Moderated Mediation Analysis**

results = bml.boot_mlmem(dataset=data,
                         x='x',
                         m1='m1',
                         m2='m2', 
                         y='y',
                         cluster='cluster',
                         modS1='modS1',
                         modS2='modS2', 
                         randx=[0, 1, 0],  # X (fixed), M1 (random), M2 (fixed)
                         randm=[1, 1],    # Both mediators with random intercepts
                         n_boot=1000,     # Reduced for a faster example
                         centering_type='group_mean' 
                        )

# **3. Interpret Results**

model = results[0]
indirect_effects = results[1]
contrasts = results[2]

print(model.summary())  # View fitted model information
print(indirect_effects) # Examine indirect effects under moderator combinations
print(contrasts)        # Examine calculated contrasts
```

## Key Considerations

    Carefully consider which random effects structures in your data you want to model and configure the randx and randm lists accordingly.
    Select an appropriate method of data centering. In most cases group mean centering is the correct method, but there is the option for grand mean centering for niche cases.

    Note:  This module assumes that the user has a good grasp of multilevel modeling concepts. Refer to Rockwood (2017) and the MLmem page (https://njrockwood.com/mlmed) for a deeper understanding.
