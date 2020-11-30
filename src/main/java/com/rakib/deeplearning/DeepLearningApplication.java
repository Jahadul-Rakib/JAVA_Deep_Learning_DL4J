package com.rakib.deeplearning;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.io.IOException;

@SpringBootApplication
public class DeepLearningApplication {

    public static void main(String[] args) throws IOException {
        SpringApplication.run(DeepLearningApplication.class, args);
//        new MultiLayerNeuralNetwork();
//        new MultiLayerNeuralNetworkWithDataSet();
    }
}
