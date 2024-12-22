import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)
import os
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
from numpy import random 
from scipy.fft import fft, fftfreq, fftshift
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tqdm import tqdm 
print("")
import title_screen
print("NO TIME EVOLUTION")
print("")
print("")

#Function to convert an array in the form [R1, R2, R3, I1, I2, I3] to [R1+I1, R2+I2, R3+I3]
def output_parse(array):
    final_prediction = []
    transpose = int(len(array)/2)
    for i in range(transpose):
        real = array[i]
        imag = array[i+transpose]
        add = complex(real, imag)
        final_prediction.append(add)
    return final_prediction

new_model = False
#Input spectra parameters
relaxation = 500
A = 0.5
points = 512
sweep_width = 10000
duration = points / sweep_width
tpi = 2 * np.pi
reference_frequency = 500 * 10**6
J_upper = 40
J_lower = 28
ppm_upper = 8
ppm_lower = -8
max_nuclei = 5
min_nuclei = 1 
p_true = 0.65 #Probability of a given nuclei being coupled
p_false = 0.35

training_q = input("Do you want to train a new model? Y/N ")

if training_q in ["Y", "y", "yes", "Yes", "YES"]:
    new_model = True
    number_of_spectra = int(input("How many spectra do you want in the training dataset? "))
    training_df = pd.DataFrame(columns=["Coupled", "FID"])
    print("")
    print("Generating training data:")
    for i in tqdm(range(number_of_spectra)):
        number_of_nuclei = random.randint(min_nuclei, max_nuclei)
        full_FID = np.linspace(0, 0, points, dtype = "complex_")
        full_coupled_FID = np.linspace(0, 0, points, dtype = "complex_")
        for j in range(number_of_nuclei):    
            desired_ppm = random.uniform(ppm_lower, ppm_upper)
            larmor = ((reference_frequency * desired_ppm) / 10**6)
            t = np.linspace(0, duration, points)
            FID = A * np.exp(relaxation * -t) * np.exp(1j * tpi * larmor * t)
            coupled_FID = FID
            coupling = random.choice([True, False], p=[p_true, p_false])
            if coupling == True:
                couple_degree = random.randint(1, 3)
                J = random.uniform(J_lower, J_upper)
                coupled_FID = FID * (np.cos(50 * J * t))**couple_degree
            for k in range(len(FID)):
                full_FID[k] = full_FID[k] + FID[k]
                full_coupled_FID[k] = full_coupled_FID[k] + coupled_FID[k] + random.normal(loc=0, scale=0.01)

        training_df = training_df.append({"Coupled" : list(full_coupled_FID), "FID" : list(full_FID)}, ignore_index=True)
    print("Training data generated. " + str(int(number_of_spectra)) + " examples generated.")
    print("")

    x = training_df["Coupled"]
    y = training_df["FID"]
    x_input = []
    y_output = []

    #This loop converts our training data from the form [R1+I1, R2+I2, R3+I3] to [R1, R2, R3, I1, I2, I3]
    #This is because the model cannot work with complex numbers, 
    #hence each complex value must be split into real and imaginary components.
    print("Converting complex FIDs to 2 real components:")
    for i in tqdm(range(len(x))):
        x_real_add = []
        x_imag_add = []
        y_real_add = []
        y_imag_add = []
        for j in range(len(x[i])):
            x_real_add.append(x[i][j].real)
            x_imag_add.append(x[i][j].imag)
            y_real_add.append(y[i][j].real)
            y_imag_add.append(y[i][j].imag)
        x_final_add = np.concatenate((x_real_add, x_imag_add), axis=0)
        y_final_add = np.concatenate((y_real_add, y_imag_add), axis=0)
        x_final_add = x_final_add.transpose()
        y_final_add = y_final_add.transpose()
        x_input.append(x_final_add)
        y_output.append(y_final_add)

    x_input = np.array(x_input)
    y_output = np.array(y_output)
    x_input = x_input.reshape(-1, 1024)
    y_output = y_output.reshape(-1, 1024)

    print("")
    print("")
    print("Training model:")
    print("")

    v2 = Sequential([
        Input(shape=(1024,)),  
        Dense(256, activation="relu"),                 
        Dropout(0.1),
        Dense(256, activation="relu"),  
        Dropout(0.1),
        Dense(256, activation="relu"),
        Dense(1024)                    
    ])

    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1) #Slows the learning rate when the decrease in loss slows down
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True) #Stops training when convergence is reached.
    v2.compile(optimizer="adam", loss="huber", metrics=["accuracy"])
    history = v2.fit(x_input, y_output, epochs=100, batch_size=32, validation_split=0.2, callbacks=[lr_scheduler, early_stopping])

    print("")
    print("**************************************************************")
    print("Training complete")
    loss_history = history.history["loss"]
    loss_avg = loss_history[-10:]
    print("Final Loss: " + str(np.mean(loss_avg)))
    print("")
    print("")
    loss_fig = plt.figure(figsize=(10, 6))
    loss_fig.canvas.manager.set_window_title("Change in loss")
    plt.plot(loss_history)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Change in loss")
    plt.show()
elif training_q in ["N", "n", "no", "No", "NO"]:
    loading_model_name = "Desktop/UCL/Y4/Python/Decoupling/saved_no_time_evolution_model/No_time_evolution_model_100k.keras"
    print("")
    print("Loading model: " + str(loading_model_name))
    v2 = tf.keras.models.load_model(loading_model_name)
else:
    print("Unknown input - quitting Decouple Net")
    quit()

print("")
print("")
v2.summary()
print("")
print("")

#Saving a newly trained model
if new_model == True:
    save_q = input("Do you want to save the model? Y/N ")
    if save_q in ["Y", "y", "yes", "Yes", "YES"]:
        model_name = "Desktop/UCL/Y4/Python/Decoupling/saved_no_time_evolution_model/No_time_evolution_model_" + str(int(number_of_spectra/1000)) + "k.keras"
        directory = os.path.dirname(model_name)
        os.makedirs(directory, exist_ok=True)
        v2.save(model_name)
        print("")
        print("Model saved as " + model_name)

#Verification
#Plots multiple spectra, showing the input and expected outputs - followed by the real output for comparison.
def verify():
    while True:
        print("")
        verify_q = input("Do you want to test the model? Y/N ")
        if verify_q not in ["Y", "y", "yes", "Yes", "YES"]:
            print("")
            print("Quitting DecoupleNET")
            print("")
            quit()
        for i in range(int(input("How many verifcation examples do you want? "))):
            print("")
            print("*************************** Verification example " + str(i + 1) + " ***************************")
            number_of_nuclei = random.randint(min_nuclei, max_nuclei)
            full_FID = np.linspace(0, 0, points, dtype = "complex_")
            full_coupled_FID = np.linspace(0, 0, points, dtype = "complex_")
            for j in range(number_of_nuclei):
                desired_ppm = random.uniform(ppm_lower, ppm_upper)
                larmor = ((reference_frequency * desired_ppm) / 10**6)
                t = np.linspace(0, duration, points)
                FID = A * np.exp(relaxation * -t) * np.exp(1j * tpi * larmor * t)
                coupled_FID = A * np.exp(relaxation * -t) * np.exp(1j * tpi * larmor * t)
                coupling = random.choice([True, False], p=[p_true, p_false])
                if coupling == True:
                    couple_degree = random.randint(1, 3)
                    J = random.uniform(J_lower, J_upper)
                    coupled_FID = coupled_FID * (np.cos(50 * J * t))**couple_degree
                for k in range(len(FID)):
                    full_FID[k] = full_FID[k] + FID[k]
                    full_coupled_FID[k] = full_coupled_FID[k] + coupled_FID[k] + random.normal(loc=0, scale=0.01)
    
            spectrum = fftshift(fft(full_FID))
            spectrum2 = fftshift(fft(full_coupled_FID))
            x_axis = fftshift(fftfreq(points, duration / points))
            ppm = ((x_axis) / (reference_frequency / 10**6))
            print("INPUT: ")
            print("Peak PPM: "  + str(ppm[spectrum.argmax()]))

            model_input = []
            for j in range(len(full_coupled_FID)):
                model_input.append(full_coupled_FID[j].real)
            for k in range((len(full_coupled_FID))):
                model_input.append(full_coupled_FID[k].imag)
            model_input = np.array(model_input)
            model_input = model_input.reshape(1, -1)
            prediction = v2.predict(model_input)
            output = []
            for l in range(len(prediction[0])):
                output.append(prediction[0][l])
            final_output = output_parse(output)
            final_spectrum = fftshift(fft(final_output))
            
            fig, axs = plt.subplots(2, 2, figsize=(10, 6))
            fig.canvas.manager.set_window_title("Verification example " + str(int(i+1)))
            axs[0][0].plot(np.real(ppm), np.real(spectrum2))
            axs[0][0].set_title("Coupled (Input)")
            axs[0][0].set_xlabel("PPM")
            axs[0][1].plot(np.real(ppm), np.real(spectrum))
            axs[0][1].set_title("Uncoupled (Target)")
            axs[0][1].set_xlabel("PPM")
            axs[1][0].plot(np.real(ppm), np.real(spectrum), label="Uncoupled")
            axs[1][0].plot(np.real(ppm), np.real(spectrum2), label="Coupled")
            axs[1][0].set_title("Coupled + Uncoupled (Overlaid)")
            axs[1][0].set_xlabel("PPM")
            axs[1][0].legend()
            axs[1][1].plot(np.real(ppm), np.real(final_spectrum))
            axs[1][1].set_title("Prediction")
            axs[1][1].set_xlabel("PPM")
            plt.tight_layout()
            plt.show()
            
            print("PREDICTION: ")
            print("Peak PPM: "  + str(ppm[final_spectrum.argmax()]))
            print("")
            print("")

verify()