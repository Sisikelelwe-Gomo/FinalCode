package finalcode;

import java.io.FileWriter;
import java.io.IOException;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;



//dont create matrix everytime


public class FinalCode { 
    
    // Fixed seed for reproducibility
    private static final int SEED = 42;
    private static final Random RANDOM = new Random(System.currentTimeMillis());

    public static void main(String[] args) {
        // Generate fixed random inputs and get targets from the teacher network
        double[][] inputData = generateFixedRandomInputs(10000, 4);
        
        int[] thiddenLayers = {5, 5, 5, 5, 5, 5, 5, 5, 5, 5,5 ,5 ,5, 5, 5,5, 5, 5, 5, 5, 5, 5, 5, 5, 5,5 ,5 ,5, 5, 5,5, 5, 5, 5, 5, 5, 5, 5, 5, 5,5 ,5 ,5, 5, 5,5, 5, 5, 5, 5, 5, 5, 5, 5, 5,5 ,5 ,5, 5, 5,5, 5, 5, 5, 5, 5, 5, 5, 5, 5,5 ,5 ,5, 5, 5,5, 5, 5, 5, 5, 5, 5, 5, 5, 5,5 ,5 ,5, 5, 5,5, 5, 5, 5, 5, 5, 5, 5, 5, 5,5 ,5 ,5, 5, 5,5, 5, 5, 5, 5, 5, 5, 5, 5, 5,5 ,5 ,5, 5, 5,5, 5, 5, 5, 5, 5, 5, 5, 5, 5,5 ,5 ,5, 5, 5,5, 5, 5, 5, 5, 5, 5, 5, 5, 5,5 ,5 ,5, 5, 5, }; // Example hidden layers configuration
        Network teacherNetwork = new Network(4, 3, thiddenLayers ,  SEED); // Example dimensions
        double[][] teacherTargets = new double[inputData.length][];
        for (int i = 0; i < inputData.length; i++) {
            teacherTargets[i] = teacherNetwork.predict(inputData[i]);
        }

        //public StudentNetwork(int inputSize, int outputSize, int[] hiddenLayers, double skipConnectionPercentage, int seed)
        
        int[] ShiddenLayers = {5};
        // Initialize student network with hidden layers and skip connections
        StudentNetwork studentNetwork = new StudentNetwork(4, 3, ShiddenLayers, 0.4, SEED); // Example dimensions and 50% skip connections
        studentNetwork.printSkipConnections();  // Verifying skip connections

        // Train the student network using the teacher's targets
        StringBuilder log = new StringBuilder();
        int epochs = 100; // Change this value to the desired number of epochs
        studentNetwork.train(inputData, teacherTargets, epochs, teacherNetwork, log);

        // Print training log
        System.out.println(log.toString());

        // Evaluation with new random inputs
        double[][] testInputs = generateFixedRandomInputs(1000, 4); // Generate new test inputs
        double totalError = 0;
        for (int i = 0; i < testInputs.length; i++) {
            double[] teacherOutput = teacherNetwork.predict(testInputs[i]);
            double[] studentOutput = studentNetwork.predict(testInputs[i]);
            totalError += computeError(teacherOutput, studentOutput);
        }
        double averageError = totalError / testInputs.length;
        System.out.println("Average Error: " + averageError);

        // Calculate and print accuracy
        double accuracy = calculateAccuracy(testInputs, teacherNetwork, studentNetwork);
        System.out.println("Accuracy: " + accuracy);

        // Save results
        saveResults("results.txt", averageError, accuracy);
    }

    private static double computeError(double[] teacherOutput, double[] studentOutput) {
        double error = 0.0;
        for (int i = 0; i < teacherOutput.length; i++) {
            error += Math.pow(teacherOutput[i] - studentOutput[i], 2);
        }
        return error;
    }

    private static double[][] generateFixedRandomInputs(int numSamples, int numFeatures) {
        double[][] inputs = new double[numSamples][numFeatures];
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                inputs[i][j] = RANDOM.nextDouble();
            }
        }
        return inputs;
    }

    private static void saveResults(String filename, double error, double accuracy) {
        try (FileWriter writer = new FileWriter(filename)) {
            writer.write("Average Error between Teacher and Student Network: " + error + "\n");
            writer.write("Accuracy: " + accuracy);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static double calculateAccuracy(double[][] inputs, Network teacherNetwork, StudentNetwork studentNetwork) {
        int correctPredictions = 0;
        for (double[] input : inputs) {
            double[] teacherOutput = teacherNetwork.predict(input);
            double[] studentOutput = studentNetwork.predict(input);
            if (isPredictionCorrect(teacherOutput, studentOutput)) {
                correctPredictions++;
            }
        }
        return (double) correctPredictions / inputs.length;
    }

    private static boolean isPredictionCorrect(double[] teacherOutput, double[] studentOutput) {
        int teacherMaxIndex = 0;
        int studentMaxIndex = 0;
        for (int i = 1; i < teacherOutput.length; i++) {
            if (teacherOutput[i] > teacherOutput[teacherMaxIndex]) {
                teacherMaxIndex = i;
            }
            if (studentOutput[i] > studentOutput[studentMaxIndex]) {
                studentMaxIndex = i;
            }
        }
        return teacherMaxIndex == studentMaxIndex;
    }
}