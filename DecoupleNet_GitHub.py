import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)
print("")
print("""
$$$$$$$\                                                    $$\                 $$\   $$\ $$$$$$$$\ $$$$$$$$\ 
$$  __$$\                                                   $$ |                $$$\  $$ |$$  _____|\__$$  __|
$$ |  $$ | $$$$$$\   $$$$$$$\  $$$$$$\  $$\   $$\  $$$$$$\  $$ | $$$$$$\        $$$$\ $$ |$$ |         $$ |   
$$ |  $$ |$$  __$$\ $$  _____|$$  __$$\ $$ |  $$ |$$  __$$\ $$ |$$  __$$\       $$ $$\$$ |$$$$$\       $$ |   
$$ |  $$ |$$$$$$$$ |$$ /      $$ /  $$ |$$ |  $$ |$$ /  $$ |$$ |$$$$$$$$ |      $$ \$$$$ |$$  __|      $$ |   
$$ |  $$ |$$   ____|$$ |      $$ |  $$ |$$ |  $$ |$$ |  $$ |$$ |$$   ____|      $$ |\$$$ |$$ |         $$ |   
$$$$$$$  |\$$$$$$$\ \$$$$$$$\ \$$$$$$  |\$$$$$$  |$$$$$$$  |$$ |\$$$$$$$\       $$ | \$$ |$$$$$$$$\    $$ |   
\_______/  \_______| \_______| \______/  \______/ $$  ____/ \__| \_______|      \__|  \__|\________|   \__|   
                                                  $$ |                                                        
                                                  $$ |                                                        
                                                  \__|                                                              
""")
import os
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
from numpy import random 
from scipy.fft import fft, fftfreq, fftshift
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tqdm import tqdm 
import datetime
print("")

class minuk_gen_FID:
    #Spectra parameters
    R2_upper = 60.
    R2_lower = 5.
    A_mean = 1.
    A_SD = 0.5
    points = 512
    tpi = 2 * np.pi
    magnet_strength_hz = 800 * 10**6 #In Hz
    reference_frequency = magnet_strength_hz * (1/4) #Because C13 NMR
    ppm_upper = 180             
    ppm_lower = 165
    sweep_width = (ppm_upper - ppm_lower) * (reference_frequency / 10**6) # = 3000
    duration = points / sweep_width # = 0.1706
    J_upper = 40
    J_lower = 28
    max_nuclei = 3
    min_nuclei = 1 
    time_period = 0.0035
    noise_magnitude = 0.1

    def normal_A(self):
        return random.normal(loc=self.A_mean, scale=self.A_SD)
    
    def __init__(self):
        J = 0
        A = 1
        omega = 0
        couple_degree = 0
        number_of_nuclei = random.randint(self.min_nuclei, self.max_nuclei)
        full_FID = np.zeros(self.points, dtype="complex_")
        full_coupled_FID = np.zeros(self.points, dtype="complex_")
        full_coupled_T_FID = np.zeros(self.points, dtype="complex_")
        for i in range(number_of_nuclei):    
            omega = random.uniform(-self.sweep_width/2, self.sweep_width/2)
            t = np.linspace(0, self.duration, self.points)
            A = self.normal_A()
            while A > 2 or A < 0:
                A = self.normal_A()
            R2 = random.uniform(self.R2_lower, self.R2_upper)
            FID = A * np.exp(1j * t * omega - R2 * t)
            coupled_FID = FID
            coupled_T_FID = FID
            couple_degree = random.randint(0, 3)
            J = random.uniform(self.J_lower, self.J_upper)
            coupled_FID = FID * (np.cos(np.pi * J * t))**couple_degree
            t += self.time_period
            coupled_T_FID = FID * (np.cos(np.pi * J * t))**couple_degree
            noise_real = random.normal(loc=0, scale=self.noise_magnitude, size=self.points)
            noise_imag = random.normal(loc=0, scale=self.noise_magnitude, size=self.points)
            noise = noise_real + 1j * noise_imag
            full_FID = full_FID + FID
            full_coupled_FID = full_coupled_FID + coupled_FID + noise
            full_coupled_T_FID = full_coupled_T_FID + coupled_T_FID + noise

        self.out_full_FID = full_FID
        self.out_coupled_FID = full_coupled_FID
        self.out_coupled_T_FID = full_coupled_T_FID
        self.out_omega = omega
        self.out_couple_degree = couple_degree
        self.out_J = J
        self.out_A = A
        self.out_no_nuclei = number_of_nuclei

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

def gen_FID():
    return minuk_gen_FID()

new_model = False
pre_trained_model = "Desktop/UCL/Y4/Python/Decoupling/saved_model/DecoupleNet_25k.keras"
training_q = input("Do you want to train a new model? Y/N ")

if training_q in ["Y", "y", "yes", "Yes", "YES"]:
    new_model = True
    number_of_spectra = int(input("How many spectra do you want in the training dataset? "))
    training_df = pd.DataFrame(columns=["Coupled", "Coupled+T", "FID"])
    print("")
    print("Generating training data:")
    for i in tqdm(range(number_of_spectra)):
        m = gen_FID()
        full_FID = m.out_full_FID
        full_coupled_FID = m.out_coupled_FID
        full_coupled_T_FID = m.out_coupled_T_FID
        training_df = training_df.append({"Coupled" : list(full_coupled_FID), "Coupled+T" : list(full_coupled_T_FID), "FID" : list(full_FID)}, ignore_index=True)
    print("Training data generated. " + str(int(len(training_df.index))) + " examples generated.")
    print("")

    x_coupled = training_df["Coupled"]
    x_coupled_t = training_df["Coupled+T"]
    y = training_df["FID"]
    x_input = []
    x_input_t = []
    y_output = []

    #This loop converts our training data from the form [R1+I1, R2+I2, R3+I3] to [R1, R2, R3, I1, I2, I3]
    #This is because the model cannot work with complex numbers, 
    #hence each complex value must be split into real and imaginary components.
    print("Converting complex FIDs to real components:")
    for i in tqdm(range(len(x_coupled))):
        x_coupled_real_add = []
        x_coupled_imag_add = []
        x_coupled_t_real_add = []
        x_coupled_t_imag_add = []
        y_real_add = []
        y_imag_add = []
        for j in range(len(x_coupled[i])):
            x_coupled_real_add.append(x_coupled[i][j].real)
            x_coupled_imag_add.append(x_coupled[i][j].imag)
            x_coupled_t_real_add.append(x_coupled_t[i][j].real)
            x_coupled_t_imag_add.append(x_coupled_t[i][j].imag)
            y_real_add.append(y[i][j].real)
            y_imag_add.append(y[i][j].imag)
        x_final_add = np.concatenate((x_coupled_real_add, x_coupled_imag_add), axis=0)
        x_final_t_add = np.concatenate((x_coupled_t_real_add, x_coupled_t_imag_add), axis=0)
        y_final_add = np.concatenate((y_real_add, y_imag_add), axis=0)
        x_input.append(x_final_add)
        x_input_t.append(x_final_t_add)
        y_output.append(y_final_add)

    x_input = np.array(x_input)
    x_input_t = np.array(x_input_t)
    y_output = np.array(y_output)
    x_input = x_input.reshape(-1, 1024)
    x_input_t = x_input_t.reshape(-1, 1024)
    y_output = y_output.reshape(-1, 1024)
    
    print("")
    print("")
    print("Training model:")
    print("")
    
    #Model architecture:
    layer_size = 1024

    input_layer_x = Input(shape=(x_input.shape[1],))
    input_layer_x_t = Input(shape=(x_input_t.shape[1],))

    x_branch = Dense(layer_size, activation="relu")(input_layer_x)
    t_branch = Dense(layer_size, activation="relu")(input_layer_x_t)

    combiner = Concatenate(name="Combiner")([x_branch, t_branch])

    hidden1 = Dense(layer_size, activation="relu")(combiner)
    hidden2 = Dense(int(layer_size/2), activation="relu")(hidden1)
    hidden3 = Dense(layer_size, activation="relu")(hidden2)
    hidden4 = Dense(int(layer_size/2), activation="relu")(hidden3)
    hidden5 = Dense(layer_size, activation="relu")(hidden4)
    output_layer = Dense(1024)(hidden5)
    #Model architecture^


    model = Model(inputs=[input_layer_x, input_layer_x_t], outputs=output_layer)
    model.compile(optimizer=Adam(), loss="mse", metrics=["mae"])

    log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1) #Slows the learning rate when the decrease in loss slows down
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1) #Stops training when convergence is reached.
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True)
    
    history = model.fit(
        [x_input, x_input_t], 
        y_output, 
        epochs=200, 
        batch_size=32, 
        validation_split=0.2, 
        callbacks=[lr_scheduler, early_stopping, tensorboard])
    
    print("")
    print("**************************************************************")
    print("Training complete")
    print("")

    loss_history = history.history["loss"]
    log_loss = np.log(loss_history)
    loss_avg = loss_history[-10:]
    print("Final Loss: " + str(np.mean(loss_avg)))
    print("")
    print("Gradients: ")
    with tf.GradientTape() as tape:
        predictions = model([x_input, x_input_t])
        loss = tf.reduce_mean(tf.square(y_output - predictions))
    gradients = tape.gradient(loss, model.trainable_variables)
    for g in gradients:
        print(tf.norm(g).numpy())
    loss_fig = plt.figure(figsize=(15, 6))
    loss_fig.canvas.manager.set_window_title("Change in loss")
    plt.plot(log_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Change in loss (log)")
    plt.show()
elif training_q in ["N", "n", "no", "No", "NO"]:
    print("")
    print("Loading model: " + str(pre_trained_model))
    model = tf.keras.models.load_model(pre_trained_model)
else:
    print("Unknown input - quitting Decouple Net")
    quit()

print("")
print("")
model.summary()
print("")
print("")

#Saving a newly trained model
if new_model == True:
    save_q = input("Do you want to save the model? Y/N ")
    if save_q in ["Y", "y", "yes", "Yes", "YES"]:
        model_name = "Desktop/UCL/Y4/Python/Decoupling/saved_model/DecoupleNet_" + str(int(number_of_spectra/1000)) + "k.keras"
        directory = os.path.dirname(model_name)
        os.makedirs(directory, exist_ok=True)
        model.save(model_name)
        print("")
        print("Model saved as " + model_name)
        print("")

#Verification
def verify():
    while True:
        verify_q = input("Do you want to test the model? Y/N ")
        if verify_q not in ["Y", "y", "yes", "Yes", "YES"]:
            print("")
            print("Quitting DecoupleNET")
            print("")
            quit()
        for i in range(int(input("How many verifcation examples do you want? "))):
            print("")
            print("*************************** Verification example " + str(i + 1) + " ***************************")
            m = gen_FID()
            full_FID = m.out_full_FID
            full_coupled_FID = m.out_coupled_FID
            full_coupled_T_FID = m.out_coupled_T_FID
            
            spectrum = fftshift(fft(full_FID))
            spectrum2 = fftshift(fft(full_coupled_FID))
            spectrum3 = fftshift(fft(full_coupled_T_FID))
            x_axis = fftshift(fftfreq(minuk_gen_FID.points, minuk_gen_FID.duration / minuk_gen_FID.points))
            ppm_center = (minuk_gen_FID.ppm_lower + minuk_gen_FID.ppm_lower) / 2
            ppm = ((x_axis) / (minuk_gen_FID.reference_frequency / 10**6)) + ppm_center

            verify_no_nuclei = m.out_no_nuclei
            verify_J = m.out_J
            verify_A = m.out_A
            verify_omega = m.out_omega
            verify_coupling_degree = m.out_couple_degree
            
            print("Number of nuclei / peaks: " + str(verify_no_nuclei))
            print("J: " + str(verify_J))
            print("A: " + str(verify_A))
            print("Omega: " + str(verify_omega))
            print("Coupling degree: " + str(verify_coupling_degree))
            print("")
            print("")
            print("INPUT: ")
            print("Peak PPM: "  + str(ppm[spectrum.argmax()]))

            model_input = []
            model_input_t = []
            for j in range(len(full_coupled_FID)):
                model_input.append(full_coupled_FID[j].real)
            for k in range((len(full_coupled_FID))):
                model_input.append(full_coupled_FID[k].imag)
            for l in range((len(full_coupled_T_FID))):
                model_input_t.append(full_coupled_T_FID[l].real)
            for m in range((len(full_coupled_T_FID))):
                model_input_t.append(full_coupled_T_FID[m].imag)
            model_input = np.array(model_input)
            model_input_t = np.array(model_input_t)
            model_input = model_input.reshape(1, -1)
            model_input_t = model_input_t.reshape(1, -1)
            prediction = model.predict([model_input, model_input_t])
            output = []
            for n in range(len(prediction[0])):
                output.append(prediction[0][n])
            final_output = output_parse(output)
            final_spectrum = fftshift(fft(final_output))

            print("PREDICTION: ")
            print("Peak PPM: "  + str(ppm[final_spectrum.argmax()]))
            print("Difference between peaks: " + str(np.abs((ppm[final_spectrum.argmax()]) - (ppm[spectrum.argmax()]))))
            
            fig, axs = plt.subplots(2, 3, figsize=(10, 6))
            fig.canvas.manager.set_window_title("Verification example " + str(int(i+1)))
            axs[0][0].plot(np.real(ppm), np.real(spectrum2))
            axs[0][0].set_title("Coupled (Input)")
            axs[0][0].set_xlabel("PPM")
            axs[0][1].plot(np.real(ppm), np.real(spectrum3))
            axs[0][1].set_title("Coupled + T (Input)")
            axs[0][1].set_xlabel("PPM")
            axs[0][2].plot(np.real(ppm), np.real(spectrum))
            axs[0][2].set_title("Uncoupled (Target)")
            axs[0][2].set_xlabel("PPM")
            axs[1][0].plot(np.real(ppm), np.real(spectrum), label="Uncoupled")
            axs[1][0].plot(np.real(ppm), np.real(spectrum2), label="Coupled")
            axs[1][0].set_title("Coupled + Uncoupled (Overlaid)")
            axs[1][0].set_xlabel("PPM")
            axs[1][0].legend()
            axs[1][1].plot(np.real(ppm), np.real(spectrum2), label="Coupled")
            axs[1][1].plot(np.real(ppm), np.real(spectrum3), label="Coupled + T")
            axs[1][1].set_title("Coupled + Coupled+T (Overlaid)")
            axs[1][1].set_xlabel("PPM")
            axs[1][1].legend()
            axs[1][2].plot(np.real(ppm), np.real(final_spectrum))
            axs[1][2].set_title("Prediction")
            axs[1][2].set_xlabel("PPM")
            plt.tight_layout()
            plt.show()
            print("")
            print("")

verify()