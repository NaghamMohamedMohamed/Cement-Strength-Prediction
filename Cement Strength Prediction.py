##################### 1.Importing libraries #####################

import pandas as pd
import numpy as np
# For data splitting
from sklearn.model_selection import train_test_split
# For Feature Scaling
from sklearn.preprocessing import StandardScaler



##################### 2.Loading the data #####################

df = pd.read_excel("concrete_data.xlsx")

print( "\nThe dataset info : " )
print ("\n" , df.info())



##################### 3.Storing the inputs , outputs #####################

features = df.drop( "concrete_compressive_strength" , axis = 1)
target = df["concrete_compressive_strength"] 

print ( "\nThe features columns : \n" , features )
print ( "\nThe target column : \n" , target )



##################### 4. Data Splitting #####################

# Split the data into training and testing sets
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.25, random_state=42)



##################### 5.Data Scaling #####################

scaler = StandardScaler()
features_train = scaler.fit_transform(features_train)
features_test = scaler.transform(features_test)



print( "\nFeatures :" )
print (" Train data type : " , type(features_train))
print(" Test data type : " , type(features_test))

print( "\nTarget :" )
print (" Train data type : " , type(target_train))
print(" Test data type : " , type(target_test))

###################### 6. NN class #####################

# For setting the needed parameters , training , testing the NN model , predicting the target value of a new input record , calc. the error of the NN model 
class NeuralNetwork :

    # Default constructor
    def __init__(self):
        # Initialize attributes

        # Rrepresents the number of neurons (or features) in the input layer. It corresponds to the number of features in the input data.
        self.input_size = None

        # Represents the number of neurons in the hidden layer
        self.hidden_size = None

        # Represents the number of neurons in the output layer. The output layer produces the final output of the network
        self.output_size = None

        # Number of iterartion ( feet forward propagation , back propagation )
        self.epochs = None

        # Which is ETA symbol ( η )in the equation of the error calculation
        self.learning_rate = None

        # Initialize weights and biases
        self.weights_input_hidden = None
        self.bias_hidden = None
        self.weights_hidden_output = None
        self.bias_output = None
        # Some variables for intermediate values during forward propagation
        self.hidden_input = None
        self.hidden_output = None
        self.final_input = None
        self.final_output = None


    # Constructor
    def set_NN_architecture(self , input_size, hidden_size, output_size , n_epochs,learning_rate):

        # The number of neurons in each layer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate

        # Initialize weights and biases
        # Initializes the weights connecting each layer with the other one
        np.random.seed(42)
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))



    # Sigmoid activation function to get the hidden layer neurons output , output neurons values
    # Used in feet forward propagation
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    # Derivative sigmoid activation function to get the hidden layer neurons output , output neurons values
    # Used in back propagation
    def sigmoid_derivative(self, a):
        return a * (1 - a)


    # A function to perform feet-froward propagation 
    def forward_propagation ( self ,input_data ) :
        # Calculate the input of hidden layers by taking the dot product ( multiplying ) each input value by the weight value for each connection between them 
        self.hidden_input = np.dot(input_data, self.weights_input_hidden) + self.bias_hidden

        # Apply activation function to hidden layer input ( sigmoid function )
        self.hidden_output = self.sigmoid(self.hidden_input)

         # Calculate final layer input ( output layer )
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output

        # Step 4: Apply activation function to final layer input ( output layer )
        #self.final_output = self.sigmoid(self.final_input)
        self.final_output = self.final_input

        return self.final_output


    # A function to perform back propagation
    def back_propagation ( self, error, learning_rate , input_data ):

        # Calculate the derivative of the final output layer with respect to its input
        output_delta = error 

        # Calculate the error in the hidden layer
        # It uses the dot product of the output_delta and the transpose of the weights connecting the hidden layer to the output layer
        # This calculation represents how much each neuron in the hidden layer contributed to the error in the final output
        # .T --> to transpose weight matrices during backpropagation
        hidden_error = output_delta.dot(self.weights_hidden_output.T)

        # Calculate the derivative of the hidden layer with respect to its input
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases ( follow the gradient descent principle )

        # To update the weights connecting the hidden layer to the output layer
        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
        #print ( self.weights_hidden_output)
        # The result is a row vector where each element is the sum of the corresponding column in output_delta.
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

        # To update the weights connecting the input layer to the hidden layer.
        # reshape --> to ensure that input_data is treated as a column vector & that the reshaped array has one column , to make sure the matrix multiplication 
        # with hidden_delta is valid
        self.weights_input_hidden += np.dot(input_data.reshape(-1, 1), hidden_delta) * learning_rate
        #print(self.weights_input_hidden)
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate


    # A function to calculate the error at the output layer for back propagtaion 
    def calculate_error(self, target, prediction):
        return target - prediction
    



##################### 7.Training the NN model #####################
    
def train ( NN, features_train, target_train):
    for epoch in range(NN.n_epochs):
        total_error = 0.0

        # Loop over each training example
        for i in range(len(features_train)):

            # a. Forward propagation
            input_data = features_train[i]
            target_value = target_train.iloc[i]

            # Call the forward propagation function 
            prediction = NN.forward_propagation(input_data)

            # b. Backpropagation and weight updates
            # Calculate the error between the output values from the NN and the actual ones
            error = NN.calculate_error(target_value, prediction)
            # Call the backpropagation function 
            NN.back_propagation(error, learning_rate , input_data)
            
            # Track total error for this epoch by calculating the mse
            total_error += pow(error,2)

        #print(total_error , " " , len(features_train))

        print(f"\nEpoch {epoch+1} --> Total Error: {total_error}")

        # Calculate mean square error for this epoch
        mean_square_error = total_error / len(features_train)

        # Output mean square error for monitoring
        print(f"\nTrain Mean Squared Error: {mean_square_error}")

        # Stop training if the error is below a certain threshold ( it can be changed )
        if mean_square_error < 0.05 :
            print("Acceptable error reached. Training stopped.")
            break
    


##################### 8.Testing the NN model #####################
        
def test ( NN, features_test, target_test):
    total_error = 0.0
    # Loop over each testing example
    for i in range(len(features_test)):

        # Forward propagation
        input_data = features_test[i]
        target_value = target_test.iloc[i]

        # Call the forward propagation function 
        prediction = NN.forward_propagation(input_data)

        # Calculate the error between the output values from the NN and the actual ones
        error = NN.calculate_error(target_value, prediction)
        total_error += pow(error,2)

    # Calculate mean square error for this epoch
    mean_square_error = total_error / len(features_test)

    # Output mean square error for monitoring
    print(f"\nTest Mean Squared Error: {mean_square_error}")



##################### 9.Predicting the target value ( cement length ) which has new values of features which are input from the user #####################
def predict ( NN , scaled_inputs) :
    # Call the forward propagation function to predict the final output ( predicted value --> concrete  strength )
    output = NN.forward_propagation(scaled_inputs)
    print ( f"\nThe predicted concrete compressive strength is : {output}")



# Creating an object from the NN class
NN = NeuralNetwork()
input_size = 4
hidden_size = 5 
output_size = 1 
n_epochs = 25
learning_rate = 0.01

NN.set_NN_architecture( input_size , hidden_size , output_size , n_epochs , learning_rate)
train( NN , features_train , target_train )

# Print weights and biases after training
#print("\nWeights Input-Hidden:\n", NN.weights_input_hidden)
#print("\nBias Hidden:\n", NN.bias_hidden)
#print("\nWeights Hidden-Output:\n", NN.weights_hidden_output)
#print("\nBias Output:\n", NN.bias_output)


test( NN , features_test , target_test)

# Take the input from the user
n_records = int(input( "\nEnter the number of records : ") )
for i in range(n_records) :
    f1 = float( input( "Enter the amount of cement : ") )
    f2 = float( input( "Enter the amount of water required : ") )
    f3 = float( input( "Enter the rigidity of cement after drying ( Superplasticizer ) : ") )
    f4 = float( input( "Enter the age before repairing : ") )

    # Converting them to numpy array
    inputs = np.array([f1, f2, f3, f4])
    # Scaling the input
    scaled_inputs = scaler.transform([inputs])

    # Print intermediate values for debugging
    #print("\nHidden Input:\n", NN.hidden_input)
    #print("\nHidden Output:\n", NN.hidden_output)
    #print("\nFinal Input:\n", NN.final_input)
    #print("\nFinal Output:\n", NN.final_output)

    predict( NN , scaled_inputs )
