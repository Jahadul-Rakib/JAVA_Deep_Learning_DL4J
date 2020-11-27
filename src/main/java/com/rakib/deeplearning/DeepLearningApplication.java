package com.rakib.deeplearning;
import com.rakib.deeplearning.ann.MultiLayerNeuralNetwork;
import com.rakib.deeplearning.ann.MultiLayerNeuralNetworkWithDataSet;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.io.IOException;

@SpringBootApplication
public class DeepLearningApplication {

    public static void main(String[] args) throws IOException {
        SpringApplication.run(DeepLearningApplication.class, args);
        new MultiLayerNeuralNetwork();
        new MultiLayerNeuralNetworkWithDataSet();
    }
}
