import pandas as pd
import numpy as np
import re
import time
from tqdm import tqdm

from Person import Person

# the number of simulations universes we want to run to capture parameter 
# uncertainty in our estimated equations 
num_universes = 5

# import all members of the first wave of MCS
mcs_people = pd.read_excel('data/mcs_people.xls', header=0)

# drop any MCS members with missing variables
mcs_people = mcs_people.dropna()
num_people = mcs_people.shape[0]

# drop columns for reference values
mcs_people = mcs_people.drop(columns=['country', 'country1', 'ETHCM6', 'ethnicity1'])

# add constant column to pick up the regression constant
mcs_people["constant"] = 1

# fix column names
mcs_people.columns = [col.replace('Age', '_age') for col in mcs_people.columns]
mcs_people.columns = [col.replace('MP', '_mp') for col in mcs_people.columns]
mcs_people.columns = [re.sub(r'INCEQ(\d+)log', r'log_eqv_inc_sweep\1', col) for col in mcs_people.columns]
replace_dict = {'_sweep1': '_age0', '_sweep2': '_age3', '_sweep3': '_age5', '_sweep4': '_age7', '_sweep5': '_age11', '_sweep6': '_age14', '_sweep7': '_age17'}
mcs_people.columns = mcs_people.columns.to_series().replace(replace_dict, regex=True)
mcs_people.columns = [re.sub(r'truancy(\d+)', r'truancy_age\1', col) for col in mcs_people.columns]
mcs_people.columns = [re.sub(r'exclP(\d+)', r'excl_age\1', col) for col in mcs_people.columns]
mcs_people.columns = [re.sub(r'Cigregular(\d+)', r'cigregular_age\1', col) for col in mcs_people.columns]
mcs_people.columns = [re.sub(r'antisoc_age(\d+)b', r'antisoc_age\1', col) for col in mcs_people.columns]
mcs_people.columns = [re.sub(r'senP(\d+)state', r'sen_age\1', col) for col in mcs_people.columns]
mcs_people.columns = [re.sub(r'Obese(\d+)_UK90', r'Obeses_UK90_age\1', col) for col in mcs_people.columns]
mcs_people.columns = [re.sub(r'(\d+)\.country', r'country\1', col) for col in mcs_people.columns]
mcs_people.columns = [re.sub(r'(\d+)\.ETHCM6', r'ethnicity\1', col) for col in mcs_people.columns]
mcs_people.columns = [re.sub(r'^(?!.*_age\d+$)(.*)$', r'\1_age0', col) for col in mcs_people.columns]
mcs_people.columns = mcs_people.columns.str.lower()
mcs_people.sort_index(axis=1, inplace=True)

# load the regression betas for the model equations
betas = pd.read_excel('data/betas.xlsx', sheet_name='Coefs')
betas.set_index('variable', inplace=True)
betas.fillna(0.0, inplace=True)
betas.replace('.', 0.0, inplace=True)
betas = betas.apply(pd.to_numeric)

betas.columns = [col.replace('Age', '_age') for col in betas.columns]
betas.columns = [re.sub(r'Obese(\d+)_UK90', r'obeses_uk90_age\1', col) for col in betas.columns]
betas.columns = [re.sub(r'senP(\d+)state', r'sen_age\1', col) for col in betas.columns]
betas.columns = [re.sub(r'truancy(\d+)', r'truancy_age\1', col) for col in betas.columns]
betas.columns = [re.sub(r'exclP(\d+)', r'excl_age\1', col) for col in betas.columns]
betas.columns = [re.sub(r'Cigregular(\d+)', r'cigregular_age\1', col) for col in betas.columns]
betas.columns = [re.sub(r'antisoc_age(\d+)b', r'antisoc_age\1', col) for col in betas.columns]
betas.columns = [re.sub(r'Kessler(\d+)', r'kessler_age\1', col) for col in betas.columns]
betas.columns = [re.sub(r'MHconditions(\d+)', r'mh_conditions_age\1', col) for col in betas.columns]
betas.columns = [col.replace('badGCSE_ME', 'bad_gcse_age17') for col in betas.columns]
betas.columns = [re.sub(r'anxdepcurrent(\d+)', r'anxdepcurrent_age\1', col) for col in betas.columns]
betas.columns = [re.sub(r'Physicalcond(\d+)', r'physicalcond\1', col) for col in betas.columns]
betas.columns = [re.sub(r'Health(\d+)_poorfair', r'health_poorfair_age\1', col) for col in betas.columns]
betas.columns = betas.columns.str.lower()
betas.sort_index(axis=1, inplace=True)

betas.index = [row.replace('Age', '_age') for row in betas.index]
betas.index = [row.replace('MP', '_mp') for row in betas.index]
betas.index = [re.sub(r'INCEQ(\d+)log', r'log_eqv_inc_sweep\1', row) for row in betas.index]
replace_dict = {'_sweep1': '_age0', '_sweep2': '_age3', '_sweep3': '_age5', '_sweep4': '_age7', '_sweep5': '_age11', '_sweep6': '_age14', '_sweep7': '_age17'}
betas.index = betas.index.to_series().replace(replace_dict, regex=True)
betas.index = [re.sub(r'truancy(\d+)', r'truancy_age\1', row) for row in betas.index]
betas.index = [re.sub(r'exclP(\d+)', r'excl_age\1', row) for row in betas.index]
betas.index = [re.sub(r'Cigregular(\d+)', r'cigregular_age\1', row) for row in betas.index]
betas.index = [re.sub(r'antisoc_age(\d+)b', r'antisoc_age\1', row) for row in betas.index]
betas.index = [re.sub(r'senP(\d+)state', r'sen_age\1', row) for row in betas.index]
betas.index = [re.sub(r'Obese(\d+)_UK90', r'Obeses_UK90_age\1', row) for row in betas.index]
betas.index = [re.sub(r'(\d+)\.country', r'country\1', row) for row in betas.index]
betas.index = [re.sub(r'(\d+)\.ETHCM6', r'ethnicity\1', row) for row in betas.index]
betas.index = [re.sub(r'^(?!.*_age\d+$)(.*)$', r'\1_age0', row) for row in betas.index]
betas.index = betas.index.str.lower()
betas.sort_index(axis=0, inplace=True)

# add columns to mcs_people to match any extra rows in beta
new_col_names = betas.index.difference(mcs_people.columns)
new_cols = pd.DataFrame(0, index=mcs_people.index, columns=new_col_names)
mcs_people = pd.concat([mcs_people, new_cols], axis=1)
mcs_people.sort_index(axis=1, inplace=True)

# load the regression betas standard errors for the model equations
betas_se = pd.read_excel('data/betas.xlsx', sheet_name='SE')
betas_se.set_index('variable', inplace=True)
betas_se.fillna(0.0, inplace=True)
betas_se.replace('.', 0.0, inplace=True)
betas_se = betas_se.apply(pd.to_numeric) 

betas_se.columns = [col.replace('Age', '_age') for col in betas_se.columns]
betas_se.columns = [re.sub(r'Obese(\d+)_UK90', r'obeses_uk90_age\1', col) for col in betas_se.columns]
betas_se.columns = [re.sub(r'senP(\d+)state', r'sen_age\1', col) for col in betas_se.columns]
betas_se.columns = [re.sub(r'truancy(\d+)', r'truancy_age\1', col) for col in betas_se.columns]
betas_se.columns = [re.sub(r'exclP(\d+)', r'excl_age\1', col) for col in betas_se.columns]
betas_se.columns = [re.sub(r'Cigregular(\d+)', r'cigregular_age\1', col) for col in betas_se.columns]
betas_se.columns = [re.sub(r'antisoc_age(\d+)b', r'antisoc_age\1', col) for col in betas_se.columns]
betas_se.columns = [re.sub(r'Kessler(\d+)', r'kessler_age\1', col) for col in betas_se.columns]
betas_se.columns = [re.sub(r'MHconditions(\d+)', r'mh_conditions_age\1', col) for col in betas_se.columns]
betas_se.columns = [col.replace('badGCSE_ME', 'bad_gcse_age17') for col in betas_se.columns]
betas_se.columns = [re.sub(r'anxdepcurrent(\d+)', r'anxdepcurrent_age\1', col) for col in betas_se.columns]
betas_se.columns = [re.sub(r'Physicalcond(\d+)', r'physicalcond\1', col) for col in betas_se.columns]
betas_se.columns = [re.sub(r'Health(\d+)_poorfair', r'health_poorfair_age\1', col) for col in betas_se.columns]
betas_se.columns = betas_se.columns.str.lower()
betas_se.sort_index(axis=1, inplace=True)

betas_se.index = [row.replace('Age', '_age') for row in betas_se.index]
betas_se.index = [row.replace('MP', '_mp') for row in betas_se.index]
betas_se.index = [re.sub(r'INCEQ(\d+)log', r'log_eqv_inc_sweep\1', row) for row in betas_se.index]
replace_dict = {'_sweep1': '_age0', '_sweep2': '_age3', '_sweep3': '_age5', '_sweep4': '_age7', '_sweep5': '_age11', '_sweep6': '_age14', '_sweep7': '_age17'}
betas_se.index = betas_se.index.to_series().replace(replace_dict, regex=True)
betas_se.index = [re.sub(r'truancy(\d+)', r'truancy_age\1', row) for row in betas_se.index]
betas_se.index = [re.sub(r'exclP(\d+)', r'excl_age\1', row) for row in betas_se.index]
betas_se.index = [re.sub(r'Cigregular(\d+)', r'cigregular_age\1', row) for row in betas_se.index]
betas_se.index = [re.sub(r'antisoc_age(\d+)b', r'antisoc_age\1', row) for row in betas_se.index]
betas_se.index = [re.sub(r'senP(\d+)state', r'sen_age\1', row) for row in betas_se.index]
betas_se.index = [re.sub(r'Obese(\d+)_UK90', r'Obeses_UK90_age\1', row) for row in betas_se.index]
betas_se.index = [re.sub(r'(\d+)\.country', r'country\1', row) for row in betas_se.index]
betas_se.index = [re.sub(r'(\d+)\.ETHCM6', r'ethnicity\1', row) for row in betas_se.index]
betas_se.index = [re.sub(r'^(?!.*_age\d+$)(.*)$', r'\1_age0', row) for row in betas_se.index]
betas_se.index = betas_se.index.str.lower()
betas_se.sort_index(axis=0, inplace=True)

# set the seed to allow us to reproduce stochastic results
np.random.seed(786110)

# create the simulated betas combining the means and standard errors to create universes
sim_betas = []
for i in range(num_universes):
    sim_mean = betas + np.random.normal(scale=betas_se)
    sim_betas.append(sim_mean)

# create simulated probabilites for binary outcomes for each person in the
# MCS sample for each universe
# extract binary columns for which we need to generate probabilities
binary_cols = betas.columns[betas.columns.str.endswith(' b')].str.replace(' b$', '', regex=True)

# add these binary columns as probability draws in the mcs_sample
sim_probs = []
for i in range(num_universes):
    df = pd.DataFrame(data=np.random.rand(num_people, binary_cols.size) ,columns=binary_cols)
    sim_probs.append(df)


for n in range(num_universes):
    start_time = time.time()
    history = pd.DataFrame(columns=['mcsid', 'mcs_sweep', 'age', 'variable', 'value'])
    
    for i in tqdm(range(num_people)):
        person = Person(mcs_people.iloc[i], sim_betas[n], sim_probs[n].iloc[i])
        person.simulate_all_sweeps()
        history = pd.concat([history, person.history], ignore_index=True)
    
    history.to_csv("output/universe_" + str(n) + ".csv", index=False)
    # stop the timer and calculate the elapsed time
    elapsed_time = time.time() - start_time
    
    # print the elapsed time
    print(f"The simulation {n} took {elapsed_time:.2f} seconds to run.")

    
