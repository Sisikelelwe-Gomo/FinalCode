/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/Classes/Class.java to edit this template
 */
package finalcode;

import java.util.Random;

/**
 *
 * @author jimmy
 */
public class Network {
    private int inputSize;
    private int outputSize;
    private int[] hiddenLayers;
    private double[][][] weights;
    private Random rand;

    public Network(int inputSize, int outputSize, int[] hiddenLayers, int seed) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.hiddenLayers = hiddenLayers;
        this.rand = new Random(seed);
        this.weights = new double[hiddenLayers.length + 1][][];  // Initialize for hidden layers + output layer
        initializeWeights();
    }

    private void initializeWeights() {
        int previousLayerSize = inputSize;

        // Initialize weights for connections from one layer to the next (no skip connections)
        for (int l = 0; l < hiddenLayers.length; l++) {
            weights[l] = new double[hiddenLayers[l]][previousLayerSize];
            for (int i = 0; i < hiddenLayers[l]; i++) {
                for (int j = 0; j < previousLayerSize; j++) {
                    weights[l][i][j] = rand.nextGaussian();  // Fully connected to only the next layer
                }
            }
            previousLayerSize = hiddenLayers[l];
        }

        // Initialize weights for the output layer
        weights[hiddenLayers.length] = new double[outputSize][previousLayerSize];
        for (int i = 0; i < outputSize; i++) {
            for (int j = 0; j < previousLayerSize; j++) {
                weights[hiddenLayers.length][i][j] = rand.nextGaussian();
            }
        }
    }

    public double[] predict(double[] input) {
        double[] previousLayerOutput = input;

        // Forward pass through hidden layers
        for (int l = 0; l < hiddenLayers.length; l++) {
            double[] currentLayerOutput = new double[hiddenLayers[l]];
            for (int i = 0; i < hiddenLayers[l]; i++) {
                for (int j = 0; j < previousLayerOutput.length; j++) {
                    currentLayerOutput[i] += previousLayerOutput[j] * weights[l][i][j];
                }
                currentLayerOutput[i] = sigmoid(currentLayerOutput[i]);
            }
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

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
}
