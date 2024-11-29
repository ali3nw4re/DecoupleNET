import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from numpy import random 
import pandas as pd
from tqdm import tqdm 

relaxation = 500
A = 0.5 
points = 512 #Number of points in each FID
sweep_width = 10000
duration = points / sweep_width
tpi = 2 * np.pi
reference_frequency = 500 * 10**6
number_of_spectra = 50000 #Number of training examples, 10-50k is reccomended
J_upper = 4000
J_lower = 2800
ppm_upper = 800
ppm_lower = -800
max_nuclei = 25
min_nuclei = 1
p_true = 0.75
p_false = 0.25


training_df = pd.DataFrame(columns=["Coupled", "FID"])

for i in tqdm(range(number_of_spectra)):
    number_of_nuclei = random.randint(min_nuclei, max_nuclei)
    full_FID = np.linspace(0, 0, points, dtype = "complex_")
    full_coupled_FID = np.linspace(0, 0, points, dtype = "complex_")
    for j in range(number_of_nuclei):    
        ppm = random.randint(ppm_lower, ppm_upper)
        desired_ppm = ppm / 100
        larmor = ((reference_frequency * desired_ppm) / 10**6)
        t = np.linspace(0, duration, points)
        FID = A * np.exp(relaxation * -t) * np.exp(1j * tpi * larmor * t)
        coupled_FID = FID
        coupling = random.choice([True, False], p=[p_true, p_false])
        if coupling == True:
            couple_degree = random.randint(1, 3)
            J = random.randint(J_lower, J_upper)
            coupled_FID = FID * (np.cos(0.5 * J * t))**couple_degree
        for k in range(len(FID)):
            full_FID[k] = full_FID[k] + FID[k]
            full_coupled_FID[k] = full_coupled_FID[k] + coupled_FID[k] + random.normal(loc=0, scale=0.01)

    training_df = training_df.append({"Coupled" : list(full_coupled_FID), "FID" : list(full_FID)}, ignore_index=True)
        
prefix = str(int(number_of_spectra/1000))
suffix = "_25multi_training_data.csv"
name = prefix + suffix
training_df.to_csv(name) 
print("File name: " + name)   
print("TRAINING DATA GENERATED")

