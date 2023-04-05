import pandas as pd
import numpy as np
import re

class Person:
    def __init__(self, x, betas, probs):
        self.mcsid = x.mcsid_age0
        self.x = x.drop('mcsid_age0')
        self.betas = betas
        self.probs = probs
        history_list = []
        for name, value in x.iteritems():
            if name.endswith('_age0'):
                row = pd.DataFrame({'mcsid': [self.mcsid], 'mcs_sweep': [1], 'age': [0], 'variable': [name[:-5]], 'value': [value]})
                history_list.append(row)
        self.history = pd.concat(history_list, ignore_index=True)

    def simulate_sweep(self, sweep_num, sweep_age, sweep_age_prev):
        x2 = pd.Series(dtype=float)
        history_list = []

        #####################################################
        # calculate continuos outcome equations
        #####################################################
      
        # select columns that end with '_age' + str(sweep_age) + ' c'
        continuous_col_names = self.betas.columns[self.betas.columns.str.endswith('_age' + str(sweep_age) + ' c')]
        
        # compute dot product of betas with x
        continuous_outcomes = np.dot(self.betas[continuous_col_names].values.T, self.x.values)
        
        # fix the variable names
        continuous_outcomes = pd.Series(continuous_outcomes, index=continuous_col_names.str.rstrip(' c'))
        
        # save output
        if sweep_num < 7:
            self.x.update(continuous_outcomes)
        else:
            x2 = continuous_outcomes
            
        # set income equal to income in previous period
        if sweep_num < 7:
            self.x['log_eqv_inc_age' + str(sweep_age)] = self.x['log_eqv_inc_age' + str(sweep_age_prev)]
        else:
            x2['log_eqv_inc_age' + str(sweep_age)] = self.x['log_eqv_inc_age' + str(sweep_age_prev)]

   
        #####################################################
        # calculate binary outcome equations
        #####################################################

        # select columns that end with '_age' + str(sweep_age) + ' b'
        binary_col_names = self.betas.columns[self.betas.columns.str.endswith('_age' + str(sweep_age) + ' b')]

        # compute dot product of betas with x
        xb = np.dot(self.betas[binary_col_names].values.T, self.x.values).astype(float)
        
        # compute probability p using sigmoid function
        p = 1 / (1 + np.exp(-xb))
        
        # get list of output variable names
        output_variables = [col_name.replace(' b', '') for col_name in binary_col_names]
        
        # Create boolean mask of outcomes based on probability thresholds
        binary_outcomes = (p < self.probs[output_variables]).astype(int)
        
        # Update x or x2 based on sweep_num
        if sweep_num < 7:
            self.x.update(binary_outcomes)
        else:
            x2 = pd.concat([x2, binary_outcomes])

        outputs = self.x if sweep_num < 7 else x2
        if sweep_age > 9:
          trim = 6
        else:
          trim = 5
          
        for name, value in outputs.iteritems():
            if name.endswith('_age' + str(sweep_age)):
                row = pd.DataFrame({'mcsid': [self.mcsid], 'mcs_sweep': [sweep_num], 'age': [sweep_age], 'variable': [name[:-trim]], 'value': [value]})
                history_list.append(row)

        self.history = pd.concat([self.history] + history_list, ignore_index=True)
        
    def simulate_all_sweeps(self):
      self.simulate_sweep(2, 3, 0)
      self.simulate_sweep(3, 5, 3)
      self.simulate_sweep(4, 7, 5)
      self.simulate_sweep(5, 11, 7)
      self.simulate_sweep(6, 14, 11)
      self.simulate_sweep(7, 17, 14)
      

