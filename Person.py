import pandas as pd
import numpy as np
import re

class Person:
    """
    A class used to represent and simulate an MCS individual from birth to age 17

    ...

    Attributes
    ----------
    mcsid: str
        contains the unique MCSID of the individual to allow linking back to the raw data
    x : Series
        Contains all the person specific characteristics
    betas : DataFrame
        Contains the regression coefficients for the lifecourse trajectory equations
    probs : Series
        Contains random draws to compare the results of the binary equations 
        with in order to decide whether a binary outcome occurred
    history : DataFrame
        Keeps track of the lifecourse trajectory by capturing the evolution of key variables
    social_care : bool
        Records whether the person is currently in social care

    Methods
    -------
    simulate_sweep(sweep_num, sweep_age, sweep_age_prev)
        Simulates an MCS sweep updating person characteristics x and adding to history
    simulate_all_sweeps()
        Simulates all 7 sweeps of MCS taking an individual from birth to age 17
    """


    def __init__(self, x, betas, probs):
        """
        Parameters
        ----------
        x : Series
            Contains all person specific characteristics including placeholder
            for characteristics that will be simulated for MCS sweeps 2-7
        betas : DataFrame
            Contains the regression coefficients for the lifecourse trajectory 
            equations with rows representing independent variables coefficients 
            and columns representing dependent (outcome) variables
        probs : Series
            Contains random draws to compare the results of the binary equations 
            with in order to decide whether a binary outcome occurred
        """
        
        self.mcsid = x.mcsid_age0
        self.social_care = False
        self.x = x.drop('mcsid_age0')
        self.betas = betas
        self.probs = probs
        
        # save starting conditions to history
        history_list = []
        
        # manually add a variable to capture social care status intialised to 0 
        row = pd.DataFrame({'mcsid': [self.mcsid], 'mcs_sweep': [1], 'age': [0], 'variable': ['social_care'], 'value': [0]})
        history_list.append(row)
        
        # loop through x and add all the age0 variables as starting coditions
        for name, value in self.x.items():
            if name.endswith('_age0'):
                row = pd.DataFrame({'mcsid': [self.mcsid], 'mcs_sweep': [1], 'age': [0], 'variable': [name[:-5]], 'value': [value]})
                history_list.append(row)
        self.history = pd.concat(history_list, ignore_index=True)
        

    def simulate_sweep(self, sweep_num):
        """Simulates an MCS sweep for the individual

        The equations to simulate the given sweep are run combining x 
        characteristics and betas coefficients. Some equations are 
        continuous and so that is all that is required in order to produce
        outcomes whilst others are binary so the equation generates an 
        odds ratio that we convert to a probability threshold using the 
        sigmoid function. This probability threshold is then compared with 
        a random number passed in specifically for this outcome through 
        the probs instance variable to calculate whether the outcome 
        occurred. All new variables simulated in the sweep are updated in 
        x and saved to the history in a one row per variable format. We
        also simulate if individuals are taken into social care during the 
        sweep, if they are we decrement the cognitive, emotional and conduct
        skills for the current sweep. We assume this is a one-off decrement
        and that individuals do not leave social care once they enter.

        Parameters
        ----------
        sweep_num : int
            The sweep that you want to simulate
            sweep 1   2   3   4   5   6   7
            age   0   3   5   7   11  14  17
        """
        
        # calculate the sweep age and previous sweep age based on the sweep number
        if sweep_num == 2:
            sweep_age = 3
            sweep_age_prev = 0
        elif sweep_num == 3:
            sweep_age = 5
            sweep_age_prev = 3
        elif sweep_num == 4:
            sweep_age = 7
            sweep_age_prev = 5
        elif sweep_num == 5:
            sweep_age = 11
            sweep_age_prev = 7
        elif sweep_num == 6:
            sweep_age = 14
            sweep_age_prev = 11
        elif sweep_num == 7:
            sweep_age = 17
            sweep_age_prev = 14
        else:
            # handle invalid sweep_num values
            raise ValueError(f"Invalid sweep_num value: {sweep_num}\nNote: sweep_num must be between 2 and 7")

        # used to store sweep 7 outcomes as these are not currently recorded in x
        x2 = pd.Series(dtype=float)
        
        # used to retain the sweep level simulated variables that will be added to history
        history_list = []
        
        #####################################################
        # calculate continuous outcome equations
        #####################################################
      
        # select columns that end with '_age' + str(sweep_age) + ' c'
        continuous_col_names = self.betas.columns[self.betas.columns.str.endswith('_age' + str(sweep_age) + ' c')]
        
        # compute dot product of betas with x
        continuous_outcomes = np.dot(self.betas[continuous_col_names].values.T, self.x.values)
        
        # fix the variable names
        continuous_outcomes = pd.Series(continuous_outcomes, index=continuous_col_names.str.rstrip(' c'))
        
        # update continuous outcomes in x
        if sweep_num < 7:
            self.x.update(continuous_outcomes)
            # bound con and emo between 0 and 10
            self.x['con_age' + str(sweep_age)] = max(min(self.x['con_age' + str(sweep_age)], 10), 0)
            self.x['emo_age' + str(sweep_age)] = max(min(self.x['emo_age' + str(sweep_age)], 10), 0)
        else:
            x2 = continuous_outcomes
            # bound con and emo between 0 and 10
            x2['con_age' + str(sweep_age)] = max(min(x2['con_age' + str(sweep_age)], 10), 0)
            x2['emo_age' + str(sweep_age)] = max(min(x2['emo_age' + str(sweep_age)], 10), 0)
            
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

        # compute dot product of betas with x to get the log odds ration
        xb = np.dot(self.betas[binary_col_names].values.T, self.x.values).astype(float)
        
        # compute threshold probabilities thershold_p using the sigmoid function and the odds ratio
        threshold_p = 1 / (1 + np.exp(-xb))
        
        # get list of output variable names
        output_variables = [col_name.replace(' b', '') for col_name in binary_col_names]
        
        # create boolean mask of outcomes based on probability thresholds and random probabilities from probs
        binary_outcomes = (self.probs[output_variables] < threshold_p).astype(int)
        
        # update binary outcomes in x 
        if sweep_num < 7:
            self.x.update(binary_outcomes)
        else:
            x2 = pd.concat([x2, binary_outcomes])

        ######################################################################
        # calculate social care outcome and adjust other outcomes accordingly
        ######################################################################
        
        if self.social_care:
            # if already in social care just record to history and take no further action    
            row = pd.DataFrame({'mcsid': [self.mcsid], 'mcs_sweep': [sweep_num], 'age': [sweep_age], 'variable': ['social_care'], 'value': [1]})
            history_list.append(row)
        else:
            # probability of entering social care is determined by IMD
            # TODO: sepicify a more plausible social care equation
            if self.x['imd1_age0'] == 1:
                social_care_p = 0.00003097
            elif self.x['imd1_age0'] == 2:
                social_care_p = 0.00005488
            elif self.x['imd1_age0'] == 3:
                social_care_p = 0.00009727
            elif self.x['imd1_age0'] == 4:
                social_care_p = 0.00017235
            elif self.x['imd1_age0'] == 5:
                social_care_p = 0.00030540
            elif self.x['imd1_age0'] == 6:
                social_care_p = 0.00054138
            elif self.x['imd1_age0'] == 7:
                social_care_p = 0.00095921
            elif self.x['imd1_age0'] == 8:
                social_care_p = 0.00170117
            elif self.x['imd1_age0'] == 9:
                social_care_p = 0.00301494
            else:
                social_care_p = 0.00533833
            
            if self.probs['social_care_age'+str(sweep_age)] < social_care_p:
                # the individual enters social care this sweep
                # update the instance flag and the history
                self.social_care = True
                row = pd.DataFrame({'mcsid': [self.mcsid], 'mcs_sweep': [sweep_num], 'age': [sweep_age], 'variable': ['social_care'], 'value': [1]})
                
                # adjust zcog, emo and con to make them deteriorate in response to entering social care 
                # TODO: find some plausible values to adjust by
                if sweep_num < 7:
                    self.x['zcog_age' + str(sweep_age)] -= 0.3
                    self.x['con_age' + str(sweep_age)] += 2
                    self.x['con_age' + str(sweep_age)] = min(self.x['con_age' + str(sweep_age)], 10)
                    self.x['emo_age' + str(sweep_age)] += 2
                    self.x['emo_age' + str(sweep_age)] = min(self.x['emo_age' + str(sweep_age)], 10)
                else:
                    x2['con_age' + str(sweep_age)] += 2
                    x2['con_age' + str(sweep_age)] = min(x2['con_age' + str(sweep_age)], 10)
                    x2['emo_age' + str(sweep_age)] += 2
                    x2['emo_age' + str(sweep_age)] = min(x2['emo_age' + str(sweep_age)], 10)
            else:
                # the individual does not enter social care this sweep
                # update the history accordingly
                row = pd.DataFrame({'mcsid': [self.mcsid], 'mcs_sweep': [sweep_num], 'age': [sweep_age], 'variable': ['social_care'], 'value': [0]})
                history_list.append(row)
                
        #####################################################
        # save sweep level outcomes to history
        #####################################################
        
        outputs = self.x if sweep_num < 7 else x2
        
        # used to trim the _ageX from the variable name before saving in history
        trim = 6 if sweep_age > 9 else 5
          
        for name, value in outputs.items():
            if name.endswith('_age' + str(sweep_age)):
                row = pd.DataFrame({'mcsid': [self.mcsid], 'mcs_sweep': [sweep_num], 'age': [sweep_age], 'variable': [name[:-trim]], 'value': [value]})
                history_list.append(row)

        self.history = pd.concat([self.history] + history_list, ignore_index=True)
  
        
    def simulate_all_sweeps(self):
        """Simulates all MCS sweep for the individual

        This runs the simulation for sweeps 2 through to 7 sequentially.
        When simulating policies, it may be desirable to instead run the
        sweep simulations individually interspersed with the policy 
        interventions which will make changes to x and/or beta values 
        which will then be picked up in subsequent sweep simulations. 
        Similarly, you may want to calculate cohort specific context 
        factors e.g. peer effects that you update between sweeps and 
        pass in through updated x and or beta values for subsequent
        sweep simulations to use.
        """
        
        for sweep_num in range(2, 8):
            self.simulate_sweep(sweep_num)
            

