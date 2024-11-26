/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package finalcode;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

/**
 *
 * @author USER
 */
public class StudentNetwork {
    private int inputSize;
    private int outputSize;
    private int[] hiddenLayers;
    private double[][][] weights;
    private double[] learningRates; // Array to store learning rates for each layer
    private Random rand;

    // Represent skip connections as a set of layer pairs (fromLayer, toLayer)
    private Set<SkipConnection> skipConnections;
    private double skipConnectionPercentage;
    
    private static final int CONVERGENCE_EPOCHS = 5; // Number of epochs to check for stability
    private static final double ERROR_CONVERGENCE_THRESHOLD = 0.01;
    private static final double ERROR_THRESHOLD = 0.05; // Example threshold value for layer error
    private static final double GRADIENT_THRESHOLD = 0.01; // Example value, adjust based on experimentation

    public StudentNetwork(int inputSize, int outputSize, int[] hiddenLayers, double skipConnectionPercentage, int seed) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.hiddenLayers = hiddenLayers;
        this.skipConnectionPercentage = skipConnectionPercentage;
        this.rand = new Random(seed);
        this.weights = new double[hiddenLayers.length + 1][][]; // Initialize layers, including the output
        this.skipConnections = new HashSet<>();
        this.learningRates = new double[hiddenLayers.length + 1]; // Array for learning rates
        initializeLearningRates(); // Initialize learning rates for layers
        printLearningRates(); // Print learning rates for layers
        initializeFullyConnectedWeights(); // Use fully connected layer weights
        initializeSkipConnections();
    }
    
    
    private void printLearningRates() {
        System.out.println("Learning Rates for each layer:");
        for (int i = 0; i < learningRates.length; i++) {
            System.out.println("Layer " + i + ": " + learningRates[i]);
        }
    }
    
        // Initialize skip connections between non-adjacent layers
    private void initializeSkipConnections() {
        for (int fromLayer = 0; fromLayer < hiddenLayers.length; fromLayer++) {
            for (int toLayer = fromLayer + 2; toLayer < hiddenLayers.length + 1; toLayer++) { // Skip at least one layer
                if (rand.nextDouble() < skipConnectionPercentage) {
                    skipConnections.add(new SkipConnection(fromLayer, toLayer));
                }
            }
        }
    }

     private void initializeFullyConnectedWeights() {
        int previousLayerSize = inputSize;

        // Initialize fully connected weights for hidden layers
        for (int l = 0; l < hiddenLayers.length; l++) {
            weights[l] = new double[hiddenLayers[l]][previousLayerSize];
            for (int i = 0; i < hiddenLayers[l]; i++) {
                for (int j = 0; j < previousLayerSize; j++) {
                    weights[l][i][j] = rand.nextGaussian();
                }
            }
            previousLayerSize = hiddenLayers[l];
        }

        // Initialize fully connected weights for output layer
        weights[hiddenLayers.length] = new double[outputSize][previousLayerSize];
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < previousLayerSize; j++) {
                weights[hiddenLayers.length][i][j] = rand.nextGaussian();
            }
        }
    }

private void adjustSkipConnections(Network teacherNetwork, int epoch) {
    for (int fromLayer = 0; fromLayer < hiddenLayers.length; fromLayer++) {
        for (int toLayer = fromLayer + 2; toLayer < hiddenLayers.length + 1; toLayer++) {
            SkipConnection skipConnection = new SkipConnection(fromLayer, toLayer);
            if (shouldAddSkipConnection(fromLayer, toLayer, teacherNetwork)) {
                if (!skipConnections.contains(skipConnection)) {
                    skipConnections.add(skipConnection);
                    System.out.println("Epoch " + epoch + ": Added skip connection from layer " + fromLayer + " to layer " + toLayer);
                }
            } else if (shouldRemoveSkipConnection(fromLayer, toLayer, teacherNetwork)) {
                if (skipConnections.remove(skipConnection)) {
                    System.out.println("Epoch " + epoch + ": Removed skip connection from layer " + fromLayer + " to layer " + toLayer);
                }
            }
        }
    }
}

// Modify the train method to pass the epoch to adjustSkipConnections
public void train(double[][] inputs, double[][] targets, int epochs, Network teacherNetwork, StringBuilder log) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        double totalLoss = 0;
        for (int i = 0; i < inputs.length; i++) {
            double[] input = inputs[i];
            double[] target = targets[i];
            double[] output = predict(input);
            totalLoss += computeLoss(output, target);
            updateWeights(input, output, target);
        }

        // Adjust skip connections every 5 epochs
        if (epoch % 5 == 0) {
            adjustSkipConnections(teacherNetwork, epoch);
        }

        // Log the average loss for this epoch
        log.append("Epoch ").append(epoch).append(", Loss: ").append(totalLoss / inputs.length).append("\n");
    }
}



    private double computeLoss(double[] output, double[] target) {
        double loss = 0.0;
        for (int i = 0; i < output.length; i++) {
            loss += Math.pow(output[i] - target[i], 2);
        }
        return loss / output.length;
    }

    // Predict method that incorporates skip connections
    public double[] predict(double[] input) {
        double[] previousLayerOutput = input;
        double[][] layerOutputs = new double[hiddenLayers.length + 1][];
        layerOutputs[0] = input;

        // Forward pass through hidden layers
        for (int l = 0; l < hiddenLayers.length; l++) {
            double[] currentLayerOutput = new double[hiddenLayers[l]];
            for (int i = 0; i < hiddenLayers[l]; i++) {
                for (int j = 0; j < previousLayerOutput.length; j++) {
                    currentLayerOutput[i] += previousLayerOutput[j] * weights[l][i][j];
                }
                currentLayerOutput[i] = sigmoid(currentLayerOutput[i]);
            }

            // Apply skip connections if any target this layer
            for (SkipConnection sc : skipConnections) {
                if (sc.toLayer == l + 1) { // Current layer is a target of a skip connection
                    double[] skipInput = layerOutputs[sc.fromLayer];
                    for (int i = 0; i < currentLayerOutput.length; i++) {
                        currentLayerOutput[i] += skipInput[i % skipInput.length] * 0.5; // Example scaling
                    }
                }
            }

            layerOutputs[l + 1] = currentLayerOutput;
            previousLayerOutput = currentLayerOutput;
        }

        // Forward pass through output layer
        double[] output = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < previousLayerOutput.length; j++) {
                output[i] += previousLayerOutput[j] * weights[hiddenLayers.length][i][j];
            }
            output[i] = sigmoid(output[i]);
        }

        return output;
    }


    // Helper function to compute sigmoid
    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    
        private void initializeLearningRates() {
        // Set learning rates for each layer (can be adjusted as needed)
        for (int i = 0; i < learningRates.length; i++) {
            learningRates[i] = (i < hiddenLayers.length) ? 0.01 / (i + 1) : 0.01; // Example decreasing rates for hidden layers
        }
    }
    
    private void updateWeights(double[] input, double[] output, double[] target) {
        double[][] layerOutputs = new double[hiddenLayers.length + 1][];
        double[][] layerInputs = new double[hiddenLayers.length + 1][];

        // Forward pass through hidden layers
        layerInputs[0] = input;
        for (int l = 0; l < hiddenLayers.length; l++) {
            double[] currentLayerOutput = new double[hiddenLayers[l]];
            for (int i = 0; i < hiddenLayers[l]; i++) {
                for (int j = 0; j < layerInputs[l].length; j++) {
                    currentLayerOutput[i] += layerInputs[l][j] * weights[l][i][j];
                }
                currentLayerOutput[i] = sigmoid(currentLayerOutput[i]);
            }
            layerOutputs[l] = currentLayerOutput;
            layerInputs[l + 1] = currentLayerOutput;
        }

        // Forward pass through output layer
        double[] outputLayerOutput = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < layerInputs[hiddenLayers.length].length; j++) {
                outputLayerOutput[i] += layerInputs[hiddenLayers.length][j] * weights[hiddenLayers.length][i][j];
            }
            outputLayerOutput[i] = sigmoid(outputLayerOutput[i]);
        }
        layerOutputs[hiddenLayers.length] = outputLayerOutput;

        // Backward pass and weight update
        double[][] errors = new double[hiddenLayers.length + 1][];
        errors[hiddenLayers.length] = new double[outputSize];
        for (int i = 0; i < outputSize; i++) {
            errors[hiddenLayers.length][i] = output[i] - target[i];
        }

        for (int l = hiddenLayers.length; l >= 0; l--) {
            int currentLayerSize = (l == hiddenLayers.length) ? outputSize : hiddenLayers[l];
            int previousLayerSize = (l == 0) ? inputSize : hiddenLayers[l - 1];

            if (l > 0) {
                errors[l - 1] = new double[hiddenLayers[l - 1]];
            }

            for (int i = 0; i < currentLayerSize; i++) {
                double error = errors[l][i];
                for (int j = 0; j < previousLayerSize; j++) {
                    double previousLayerOutput = (l == 0) ? input[j] : layerOutputs[l - 1][j];
                    // Update weights using the specific learning rate for this layer
                    weights[l][i][j] -= learningRates[l] * error * previousLayerOutput;
                }
                if (l > 0) {
                    for (int j = 0; j < previousLayerSize; j++) {
                        errors[l - 1][j] += error * weights[l][i][j];
                    }
                }
            }
        }
    }
    
    // Inner class to represent a skip connection



// Print skip connections
public void printSkipConnections() {
    System.out.println("Initialized Skip Connections:");
    for (SkipConnection sc : skipConnections) {
        System.out.println("From Layer " + sc.fromLayer + " to Layer " + sc.toLayer);
    }
}

// Helper method to get the output for a specific layer
private double[] getLayerOutput(double[] input, int layer) {
    double[] currentOutput = input;
    for (int l = 0; l <= layer; l++) {
        currentOutput = forwardPass(currentOutput, l);
    }
    return currentOutput;
}

// Computes squared error between layer output and target
private double computeLayerError(double[] output, double[] target) {
    int minLength = Math.min(output.length, target.length);
    if (output.length != target.length) {
       // System.out.println("Warning: Mismatched array lengths. Using the first " + minLength + " elements.");
    }
    
    double error = 0.0;
    for (int i = 0; i < minLength; i++) {
        error += Math.pow(output[i] - target[i], 2);
    }
    return error / minLength;
}



//getGradientForLayer
private double[] generateSampleInput() {
    int inputSize = this.inputSize; // Assuming inputSize is the number of input features
    double[] input = new double[inputSize];
    for (int i = 0; i < inputSize; i++) {
        input[i] = rand.nextDouble(); // Generate random values between 0 and 1
    }
    return input;
}

private double[] generateSampleTarget(double[] input, Network teacherNetwork) {
    return teacherNetwork.predict(input); // Call predict on the passed instance
}




private double[][] computeGradients(double[] input, double[] target) {
    int numLayers = hiddenLayers.length + 1; // Number of layers, including output
    double[][] gradients = new double[numLayers][]; // To store gradients for each layer

    // Step 1: Forward Pass to get layer outputs
    double[][] layerOutputs = new double[numLayers][];
    layerOutputs[0] = input;
    for (int l = 0; l < hiddenLayers.length; l++) {
        layerOutputs[l + 1] = forwardPass(layerOutputs[l], l);
    }
    double[] output = forwardPass(layerOutputs[hiddenLayers.length - 1], hiddenLayers.length);

    // Step 2: Compute Loss Gradient (output layer)
    double[] outputLayerGradient = new double[outputSize];
    for (int i = 0; i < outputSize; i++) {
        outputLayerGradient[i] = 2 * (output[i] - target[i]); // Assuming Mean Squared Error loss
    }
    gradients[hiddenLayers.length] = outputLayerGradient;

    // Step 3: Backward Pass to calculate gradients for each layer
    for (int l = hiddenLayers.length - 1; l >= 0; l--) {
        double[] currentGradients = new double[hiddenLayers[l]];

        // Propagate gradients through weights
        for (int i = 0; i < hiddenLayers[l]; i++) {
            double sumGradient = 0.0;
            for (int j = 0; j < layerOutputs[l].length; j++) {
                sumGradient += outputLayerGradient[j] * weights[l][i][j];
            }
            currentGradients[i] = sumGradient * sigmoidDerivative(layerOutputs[l][i]);
        }
        gradients[l] = currentGradients;

        // Update outputLayerGradient for next layer (backpropagate gradient)
        outputLayerGradient = currentGradients;
    }
    
    return gradients;
}

// Helper method to perform a forward pass through a specific layer
private double[] forwardPass(double[] input, int layer) {
    double[] output = new double[hiddenLayers[layer]];
    for (int i = 0; i < hiddenLayers[layer]; i++) {
        for (int j = 0; j < input.length; j++) {
            output[i] += input[j] * weights[layer][i][j];
        }
        output[i] = sigmoid(output[i]); // Activation function
    }
    return output;
}

// Helper for sigmoid derivative
private double sigmoidDerivative(double x) {
    return x * (1 - x); // Derivative of sigmoid(x) = x * (1 - x)
}


// Heuristics



private double getErrorForLayer(int layer, Network teacherNetwork) {
    double totalError = 0.0;
    int numSamples = 100; // Sample size for computing error
    for (int i = 0; i < numSamples; i++) {
        double[] input = generateSampleInput();
        double[] target = generateSampleTarget(input, teacherNetwork); // Pass teacherNetwork
        double[] layerOutput = getLayerOutput(input, layer);
        totalError += computeLayerError(layerOutput, target);
    }
    return totalError / numSamples;
}

private double getGradientForLayer(int layer, Network teacherNetwork) {
    double totalGradient = 0.0;
    int numSamples = 100; // Sample size for computing gradients (tune this as needed)
    for (int i = 0; i < numSamples; i++) {
        double[] input = generateSampleInput();
        double[] target = generateSampleTarget(input, teacherNetwork);
        double[][] gradients = computeGradients(input, target);
        totalGradient += calculateAverageGradient(gradients[layer]);
    }
    return totalGradient / numSamples;
}

// Helper to calculate average gradient magnitude for a layer
private double calculateAverageGradient(double[] gradients) {
    double sum = 0.0;
    for (double grad : gradients) {
        sum += Math.abs(grad);
    }
    return sum / gradients.length;
}

private boolean isConverged(int layer, Network teacherNetwork) {
    boolean isStable = true;
    for (int i = 0; i < CONVERGENCE_EPOCHS; i++) {
        double layerError = getErrorForLayer(layer, teacherNetwork);
        if (layerError > ERROR_CONVERGENCE_THRESHOLD) {
            isStable = false;
            break;
        }
    }
    return isStable;
}

private boolean shouldAddSkipConnection(int fromLayer, int toLayer, Network teacherNetwork) {
    // Example: Add a connection if error or gradient threshold is met
    return getErrorForLayer(fromLayer, teacherNetwork) > ERROR_THRESHOLD || 
           getGradientForLayer(fromLayer, teacherNetwork) < GRADIENT_THRESHOLD;
}

private boolean shouldRemoveSkipConnection(int fromLayer, int toLayer, Network teacherNetwork) {
    // Example: Remove a connection if convergence is stable
    return isConverged(fromLayer, teacherNetwork) && 
           skipConnections.contains(new SkipConnection(fromLayer, toLayer));
}


// Inner class to represent a skip connection
private static class SkipConnection {
    int fromLayer;
    int toLayer;

    SkipConnection(int fromLayer, int toLayer) {
        this.fromLayer = fromLayer;
        this.toLayer = toLayer;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof SkipConnection)) return false;
        SkipConnection sc = (SkipConnection) obj;
        return this.fromLayer == sc.fromLayer && this.toLayer == sc.toLayer;
    }

    @Override
    public int hashCode() {
        return fromLayer * 31 + toLayer;
    }
}

}