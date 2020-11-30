package com.rakib.deeplearning.cnn;

import java.io.File;

public class ConvolutionalNeuralNetwork {
    int numberOfChannel = 1;
    int width =28;
    int height =28;
    int numberOfOutPut =2;
    int batchSize =15;
    int numberOfEpochs =200;
    int numberOfIteration =1;
    int numberOfSeed =123;

    public ConvolutionalNeuralNetwork() {
        
        File trainingData = new File("src/main/resources/catanddog/training_set");
        File testingData = new File("src/main/resources/catanddog/test_set");


    }
}
