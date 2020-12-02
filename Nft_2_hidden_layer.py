import cv2
import numpy as np
import sys

sys.stdout = open("TwoHiddenLayers.txt", "w")

def binsig(x):
    return (1/(1+np.exp(-x)))

def initialise_weights(x, y):
    vec = np.random.rand(x, y)
    return np.multiply(vec,0.01)

def initialise_bias(x):
    vec = np.random.rand(x)
    return np.multiply(vec, 0.01)

def create_input_vector():
    
    # Reading image
    img_files = ["./Nft/a1.jpg", "./Nft/j1.jpg", "./Nft/p1.jpg", "./Nft/a2.jpg", "./Nft/j2.jpg", 
                 "./Nft/p2.jpg", "./Nft/a3.jpg", "./Nft/j3.jpg", "./Nft/p3.jpg", "./Nft/a4.jpg", 
                 "./Nft/j4.jpg", "./Nft/p4.jpg", "./Nft/a5.jpg", "./Nft/j5.jpg", "./Nft/p5.jpg"]
    X = []
    for file in img_files:
        img = cv2.imread(file,0)
        
        for i in range(10):
            for j in range(10):
                if img[i][j]==255:
                    img[i][j]=1
                else:
                    img[i][j]=0
                    
        A=np.asarray(img).reshape(-1)
        X.append(A)
    return X

def create_target_vector():
    T=np.zeros((15, 3),dtype=int)
    for i in range(15):
        T[i][(i%3)] = 1    
    return T
 
def create_test_input_vector():
    img_files = ["./Nft/a6.jpg", "./Nft/j6.jpg", "./Nft/p6.jpg", "./Nft/a1.jpg", "./Nft/j7.jpg", 
                 "./Nft/p7.jpg", "./Nft/k.jpg" ]
    X = []
    for file in img_files:
        img = cv2.imread(file,0)
        
        for i in range(10):
            for j in range(10):
                if img[i][j]==255:
                    img[i][j]=1
                else:
                    img[i][j]=0
                    
        A=np.asarray(img).reshape(-1)
        X.append(A)
    return X
    
def create_test_expected_output():
    T=np.zeros((7, 3),dtype=int)
    for i in range(6):
        T[i][(i%3)] = 1
        
    T[6] = [1, 1, 1]
    return T

def train_images_data(x, t, v1, b1, v2, b2, w, b3, alpha):
    
    num_input_nodes=100
    num_first_hidden_nodes=70
    num_second_hidden_nodes=40
    num_output_nodes=3
    
    connection=1;
    epoch=0;
    while connection:
        e=0
        for I in range(15):
            #Feed forward
            
            # Input layer to 1st Hidden layer
            for j in range(num_first_hidden_nodes):
                z1in[j] = b1[j]
                for i in range(num_input_nodes):
                    z1in[j] = z1in[j] + (x[I][i]*v1[i][j])
            
                z1[j]=binsig(z1in[j])
            
            # 1st Hidden to 2nd Hidden layer
            for j in range(num_second_hidden_nodes):
                z2in[j] = b2[j]
                for i in range(num_first_hidden_nodes):
                    z2in[j] = z2in[j] + (z1[i]*v2[i][j])
            
                z2[j]=binsig(z2in[j])
    
            # 2nd Hidden layer to Output
            for k in range(3):
                yin[k] = b3[k]
                for l in range(num_second_hidden_nodes):
                    yin[k] = yin[k] + (w[l][k]*z2[l])
                    
                y[I][k] = binsig(yin[k])
                
            #Backpropagation of Error
            
            #For output nodes
            for k in range(3):
                delk[k] = (t[I][k]-y[I][k])*y[I][k]*(1-y[I][k])
                #Error calculation
                e = e+((t[I][k]-y[I][k])**2)
                
            
            #For hidden nodes
            for i in range(num_second_hidden_nodes):
                delw2[i]=0
                for j in range(3):
                    delw2[i] += delk[j]*w[i][j]
                    
            for i in range(num_second_hidden_nodes):
                delw2[i] = delw2[i]*z2[i]*(1-z2[i])
                
            for i in range(num_first_hidden_nodes):
                delw1[i]=0
                for j in range(num_second_hidden_nodes):
                    delw1[i] += delw2[j]*v2[i][j]
                    
            for i in range(num_first_hidden_nodes):
                delw1[i] = delw1[i]*z1[i]*(1-z1[i])
            
            #Input to output weights updation
            
            #Input layer to 1st Hidden Layer 
            for i in range(100):
                for j in range(num_first_hidden_nodes):
                    v1[i][j] += alpha*delw1[j]*x[I][i]
                    
            #1st Hidden nodes' bias updation
            for i in range(num_first_hidden_nodes):
                b1[i] += alpha*delw1[i]
            
            # 1st Hidden to 2nd Hidden Layer
            for i in range(num_first_hidden_nodes):
                for j in range(num_second_hidden_nodes):
                    v2[i][j] += alpha*delw2[j]*z1[i]
                    
            #2nd Hidden nodes' bias updation
            for i in range(num_second_hidden_nodes):
                b2[i] += alpha*delw2[i]    
            
            #2nd Hidden to output weights updation
            for i in range(num_second_hidden_nodes):
                for j in range(3):
                    w[i][j] += alpha*delk[j]*z2[i]
               
            #Output nodes' bias updation
            for i in range(3):
                b3[i] += alpha*delk[i]
                
        epoch = epoch+1
        if e<0.01 or epoch>=10000:
            connection=0
        #else:
         #   print(epoch)
          #  print(e)
    
    #disp('BPN for XOR funtion with Binary input and Output');
    print(f"Total Epochs Performed: {epoch}")
    print(f"Error: {e}")
    print("The Weights are: ")
    print(w)    
  
def test_images_data(v1, b1, v2, b2, w, b3, x, t):
    Y=[0,0,0]    
    
    num_input_nodes=100
    num_first_hidden_nodes=70
    num_second_hidden_nodes=40
    num_output_nodes=3

    for I in range(7):
        #Feed forward
        
        Y[0]=Y[1]=Y[2]=0
        
        # Input layer to 1st Hidden layer
        for j in range(num_first_hidden_nodes):
            z1in[j] = b1[j]
            for i in range(num_input_nodes):
                z1in[j] = z1in[j] + (x[I][i]*v1[i][j])
        
            z1[j]=binsig(z1in[j])
        
        # 1st Hidden to 2nd Hidden layer
        for j in range(num_second_hidden_nodes):
            z2in[j] = b2[j]
            for i in range(num_first_hidden_nodes):
                z2in[j] = z2in[j] + (z1[i]*v2[i][j])
        
            z2[j]=binsig(z2in[j])

        # 2nd Hidden layer to Output
        for k in range(3):
            yin[k] = b3[k]
            for l in range(num_second_hidden_nodes):
                yin[k] = yin[k] + (w[l][k]*z2[l])
                
            y[I][k] = binsig(yin[k])
                
            
            if y[I][k] > 0.5:
                Y[k] = 1
            else:
                Y[k] = 0
        
        if Y[0]==1 and Y[1]==0 and Y[2]==0:        #output cases
            result = "A"
        elif Y[0]==0 and Y[1]==1 and Y[2]==0: 
            result = "J"
        elif Y[0]==0 and Y[1]==0 and Y[2]==1:
            result = "P"
        else :
            result = "Ambiguous Character"
        
        if I!=6:
            print(f"Decimal output: {y[I]}   Resultant Output: {Y} = {result}  Expected Output: {t[I]}")
        else:
            print(f"Decimal output: {y[I]}   Resultant Output: {Y} = {result}  Expected Output: Ambiguous Character")


np.set_printoptions(suppress=True)
#MAIN FUNCTION INTERFACE

# Different values of alpha
alpha=[0.8, 0.4, 0.2, 0.1, 0.05, 0.01]

# Declarations
num_input_nodes=100
num_first_hidden_nodes=70
num_second_hidden_nodes=40
num_output_nodes=3

#Creating input vector
x = create_input_vector()
#Creating target vector
t = create_target_vector()

#Creating input data for testing
testX = create_test_input_vector()
#Creating expected output data for test data
testT = create_test_expected_output()
#Training & Testing for all the alpha values
for i in range(6):
    
    #Input layer to 1st Hidden layer weights and bias
    v1=initialise_weights(num_input_nodes, num_first_hidden_nodes)
    b1=initialise_bias(num_first_hidden_nodes)

    # 1st Hidden to 2nd hidden layer weights and bias 
    v2=initialise_weights(num_first_hidden_nodes, num_second_hidden_nodes)
    b2=initialise_bias(num_second_hidden_nodes)
    
    #2nd Hidden to Output layer weights and bias
    w=initialise_weights(num_second_hidden_nodes, num_output_nodes)
    b3=initialise_bias(num_output_nodes)
    
    #Input to 1st hidden layer
    z1in=np.zeros(num_first_hidden_nodes, dtype=float)
    #Converted input to 1st hidden layer
    z1=np.zeros(num_first_hidden_nodes, dtype=float)
    
    #Input to 2nd hidden layer
    z2in=np.zeros(num_second_hidden_nodes, dtype=float)
    #Converted input to 2nd hidden layer
    z2=np.zeros(num_second_hidden_nodes, dtype=float)
    
    #Input given to output layer
    yin=np.zeros(3, dtype=float)
    #Output result
    y=np.zeros((15,3), dtype=float)
    
    delk=np.zeros((3), dtype=float)
    delw2=np.zeros((num_second_hidden_nodes), dtype=float)
    delw1=np.zeros((num_first_hidden_nodes), dtype=float)

    print(f"\nLearning Rate: {alpha[i]}\n")
    train_images_data(x, t, v1, b1, v2, b2, w, b3, alpha[i])
    test_images_data(v1, b1, v2, b2, w, b3, testX, testT)
    


