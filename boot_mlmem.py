import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import scipy.stats as stats # For bootstrap inferences
from patsy import dmatrices  # For formula creation 

def calculate_ml_mod_mediation(dataset, x, m1, y, cluster, 
                              m2=None, m3=None, 
                              modS1=None, modS2=None, modS=None,
                              randx=[0, 0, 0], randm=[0, 0],
                              n_boot=1000, centering_type="group_mean"):
    """
    Calculates multilevel moderated mediation effects with contrast analysis.

    Args:
        dataset: Input dataset (e.g., a statsmodels object or DataFrame).
        x: Name of the independent variable (X).
        m1: Name of the first mediator (M1).
        y: Name of the dependent variable (Y).
        cluster: Name of the variable indicating level-2 cluster membership.
        m2: Optional, name of the second mediator (M2).
        m3: Optional, name of the third mediator (M3).
        modS1: Optional, name of the A-path level-2 moderator.
        modS2: Optional, name of the B-path level-2 moderator.
        modS: Optional, name of the C'-path level-2 moderator.
        randx: List specifying random effects for X paths (default all fixed).
        randm: List specifying random effects for mediator paths (default all fixed)
	n_boot: Number of bootstrapping resamples.
	centering_type: Type of centering (grand mean or group mean) 

    Returns:
        Results with indirect effects, contrast effects, and significance. 
    """

    # 1. Data Preparation
    data = dataset.copy()  # Avoid modifying the original

    # Center mediators based on choice
    mediators = [m for m in [m1, m2, m3] if m is not None]
    data = center_variables(data, mediators, centering_type, cluster) 

    # 2. Formula Creation (using Patsy)
    formula_base = f' {y} ~ {x} + {m1} '   # Main part
    if m2: formula_base += f'+ {m2}'
    if m3: formula_base += f'+ {m3}'

    # Add interactions with moderators if given
    if modS1: 
        formula_base += f'+ {x}*{modS1}'
    if modS2: 
        formula_base += f'+ {m1}*{modS2}' # Interaction with mediator and a B-path moderator
    if modS: 
        formula_base += f'+ {x}*{modS}' # Interaction with the direct effect 

    # Build formula with random effects dynamically  
    formula = formula_base 

    if randx:
        for i, effect in enumerate(randx):
            if effect:  # If the effect is designated as random
                if i == 0:  # Random effect on X (c' path)
                    formula += f' + (1 + {x} | {cluster})' 
                else:  # Random effects on mediators
                    med = f'm{i}'  # Assuming mediators are m1, m2, etc.
                    formula += f' + (1 + {med} | {cluster})'

    if randm:
        for i, effect in enumerate(randm):
            if effect:
                med = f'm{i + 1}'  # Mediators starting with m1
                formula += f' + (1 | {cluster}:{med})'  # Example syntax

    # 3. Model Fitting
    model = smf.mixedlm(formula, data, groups=data[cluster]).fit()  

    # 4. Effect Calculation
    mod_values = {}  # For storing moderator values to use
    if modS1: 
        mod_values[modS1] = [data[modS1].mean() - data[modS1].std(),
                             data[modS1].mean() + data[modS1].std()] 
    if modS2:  # Add a section for the 'modS2'
        mod_values[modS2] = [data[modS2].mean() - data[modS2].std(),
                             data[modS2].mean() + data[modS2].std()] 
    if modS:  # Add similar logic for 'modS'
        mod_values[modS] = [data[modS].mean() - data[modS].std(),
                             data[modS].mean() + data[modS].std()] 

    indirect_effects = {}  # Store effects by condition
    # Calculate effects based on model parameters and store in 'indirect_effects'
	for modS1_val in mod_values.get(modS1, [None]):  # Iterate over modS1 if provided
        for modS2_val in mod_values.get(modS2, [None]):  # Iterate over modS2 if provided
        for modS_val in mod_values.get(modS, [None]): # Iterate over modS if provided

                # Calculate moderated coefficients
		a1_path = model.params[x] + model.params.get(f'{x}*{modS1}', 0) * modS1_val
                a2_path = model.params.get(f'{x}*{m2}', 0) + model.params.get(f'{x}*{m2}*{modS1}', 0) * modS1_val 
                a3_path = model.params.get(f'{x}*{m3}', 0) + model.params.get(f'{x}*{m3}*{modS1}', 0) * modS1_val

                b1_path = model.params[m1] + model.params.get(f'{m1}*{modS2}', 0) * modS2_val
                b2_path = model.params[m2] + model.params.get(f'{m2}*{modS2}', 0) * modS2_val
                b3_path = model.params[m3] + model.params.get(f'{m3}*{modS2}', 0) * modS2_val

                c_prime = model.params[x] + model.params.get(f'{x}*{modS}', 0) * modS_val

                # Indirect Effects (Including potential combinations)
                ind_effect1 = a1_path * b1_path 
                if m2:
                    ind_effect2 = a2_path * b2_path 
                if m3: 
                    ind_effect3 = a3_path * b3_path 

                # Various Combinations:
                if m2 and m3:
                    ind_effect12 = a1_path * b2_path  # M1 and M2 only
                    ind_effect13 = a1_path * b3_path  # M1 and M3 only
                    ind_effect23 = a2_path * b3_path  # M2 and M3 only
                    ind_effect123 = a1_path * b2_path * b3_path  # All three mediators 

                # Store with descriptive key (example)
                key = f'S1_{modS1_val}_S2_{modS2_val}_S_{modS_val}'  
                indirect_effects[key] = [ind_effect1, ind_effect2, ind_effect3, 
                                         ind_effect12, ind_effect13, ind_effect23,
                                         ind_effect123]  # Update list if relevant
		# Assuming a list structure

    direct_effects = {}  # Store direct effects (c' path) 
    for modS_val in mod_values.get(modS, [None]):
        c_prime = model.params[x] + model.params.get(f'{x}*{modS}', 0) * modS_val
        key = f'Direct_S_{modS_val}' 
        direct_effects[key] = c_prime

    # 5. Contrasts   
    contrasts = {}  # Store comparisons
    # Example Contrasts (Expanded)

    # ... Single Mediator Contrast with modS1
    key = f'M1_Contrast_{modS1}'  
    if modS1 in mod_values: 
        high_mod_effect = indirect_effects[f'S1_{mod_values[modS1][1]}_S2_None_S_None'][0] 
        low_mod_effect = indirect_effects[f'S1_{mod_values[modS1][0]}_S2_None_S_None'][0]
        contrasts[key] = high_mod_effect - low_mod_effect

    # ... Contrast with modS2 (B-path moderator)
    key = f'M1_Contrast_{modS2}'  
    if modS2 in mod_values: 
        high_mod_effect = indirect_effects[f'S1_None_S2_{mod_values[modS2][1]}_S_None'][0]
        low_mod_effect = indirect_effects[f'S1_None_S2_{mod_values[modS2][0]}_S_None'][0]
        contrasts[key] = high_mod_effect - low_mod_effect

    # ... Direct effect contrasts
    key = f'Cprime_Contrast_{modS}'  
    if modS in mod_values:  
        high_val = mod_values[modS][1]
        low_val = mod_values[modS][0]
        high_effect = direct_effects[f'Direct_S_{high_val}'] 
        low_effect = direct_effects[f'Direct_S_{low_val}'] 
        contrasts[key] = high_effect - low_effect 

    # ... Example combined Moderation Contrasts, not currently implemented
    # key = 'M1_Contrast_S1High_S2Low' 
    # if modS1 in mod_values and modS2 in mod_values:
        # effect_S1High_S2Low = indirect_effects[f'S1_{mod_values[modS1][1]}_S2_{mod_values[modS2][0]}_S_None'][0]
        # ... Similarly retrieve other relevant effects & compute contrasts   

    # 6. Inference  (Using Bootstrap)
    boot_indirect_effects = {}  
    boot_contrasts = {} 

    for _ in range(n_boot): # Uses the 'n_boot' parameter
        boot_sample = resample(dataset, replace=True)  
        boot_model = smf.mixedlm(formula, boot_sample, groups=boot_sample[cluster]).fit()

    # 7. Confidence Interval Calculation and Output
    print("INDIRECT EFFECTS WITH 95% CONFIDENCE INTERVALS:")
    for effect, values in boot_indirect_effects.items():
        lower_ci = stats.percentile(values, 2.5)
        upper_ci = stats.percentile(values, 97.5)
        print(f"  {effect}: ({lower_ci}, {upper_ci})")

    print("\nCONTRASTS WITH 95% CONFIDENCE INTERVALS:")
    for contrast, values in boot_contrasts.items():
        lower_ci = stats.percentile(values, 2.5)
        upper_ci = stats.percentile(values, 97.5)
        print(f"  {contrast}: ({lower_ci}, {upper_ci})")

    # 8. Return Results
    return model, indirect_effects, contrasts 

def center_variables(dataset, variables, centering_type="grand_mean", cluster=None):
    """
    Centers variables by group-mean or grand-mean.

    Args:
        dataset (pd.DataFrame): The dataset containing the variables.
        variables (list): List of variable names to center.
        centering_type (str):  Type of centering. Options: 
            * 'group_mean' (default): Subtracts group mean.
            * 'grand_mean':  Subtracts overall mean.
        cluster (str): If 'group_mean' centering, name of the clustering variable.

    Returns:
        pd.DataFrame: The modified dataset with centered variables.
    """

    data = dataset.copy()
    for var in variables:
        if centering_type == 'group_mean':
            if cluster is None:
                raise ValueError("For group-mean centering, 'cluster' variable name is required")
            data[var] = data[var] - data.groupby(cluster)[var].transform('mean')
        elif centering_type == 'grand_mean':
            data[var] = data[var] - data[var].mean()
        else:
            raise ValueError("Invalid centering type. Choose 'group_mean' or 'grand_mean'")

    return data
