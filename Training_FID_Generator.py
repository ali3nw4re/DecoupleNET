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
number_of_spectra = 25000 #Number of training examples, 10-25k is reccomended
J_upper = 4000
J_lower = 2800
ppm_upper = 800
ppm_lower = -800

training_df = pd.DataFrame(columns=["Coupled", "FID"])

for i in tqdm(range(number_of_spectra)):
    ppm = random.randint(ppm_lower, ppm_upper)
    desired_ppm = ppm / 100
    larmor = ((reference_frequency * desired_ppm) / 10**6)
    t = np.linspace(0, duration, points)
    FID = A * np.exp(relaxation * -t) * np.exp(1j * tpi * larmor * t)
    for j in range(len(FID)):
        FID[j] = FID[j] + random.normal(loc=0, scale=0.01)
    couple_degree = random.randint(1,3) 
    J = random.randint(J_lower, J_upper)
    coupled_FID = FID * (np.cos(0.5 * J * t))**couple_degree
    training_df = training_df.append({"Coupled" : list(coupled_FID), "FID" : list(FID)}, ignore_index=True)
    
training_df.to_csv("training_data.csv")    
print("TRAINING DATA GENERATED")