from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import model_from_json

data_input_x = np.linspace(-5, 5, 100000) #valores a serem calculados
data_output_y = np.sin(data_input_x) #respostas dos valores de entrada

#Plots
plt.plot(data_input_x)
plt.plot(data_output_y)
activation = 'sigmoid' #'tanh' ou 'sigmoid'

model = Sequential()
#Layers
model.add(Dense(units=5, activation='{}'.format(activation), input_shape=(1,))) #units(quantidade de saidas), input_shape(quantidade de entradas)
model.add(Dense(units=1, activation='linear')) #Permite numeros negativos

#Definicoes
model.compile(loss='mse', optimizer=Adam())

#Print summary
model.summary()

#Trainning
model.fit(data_input_x, data_output_y, epochs=200)

# serialize model to JSON
model_json = model.to_json()
with open("model_final_{}/model_and_weight/model_final_{}.json".format(activation, activation), "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model_final_{}/model_and_weight/model_final_{}.h5".format(activation, activation))
print("Saved model to disk")

#prediction
data_output_predicted_y = model.predict(data_input_x)

#Prints e plots
model.get_weights()
plt.plot(model.predict(data_input_x))
plt.plot(data_output_y)


#Saving the predicted outputs
with open("model_final_{}/data/predicted_output.txt".format(activation), "w") as predicted_file:
    for i in range(len(data_output_predicted_y)):
        save_predicted = str(float(data_output_predicted_y[i]))
        predicted_file.write("{}\n".format(save_predicted))

with open("model_final_{}/data/data_output.txt".format(activation), "w") as output_file:
    for j in range(len(data_output_y)):
        save_output = str(float(data_output_y[j]))
        output_file.write("{}\n".format(save_output))

with open("model_final_{}/data/data_input.txt".format(activation), "w") as input_file:
    for k in range(len(data_input_x)):
        save_input = str(float(data_input_x[k]))
        input_file.write("{}\n".format(save_input))

