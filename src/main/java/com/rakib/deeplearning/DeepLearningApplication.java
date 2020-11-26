package com.rakib.deeplearning;
import com.rakib.deeplearning.ann.MultiLayerNeuralNetwork;
import com.rakib.deeplearning.ann.MultiLayerNeuralNetworkWithDataSet;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class DeepLearningApplication {

    public static void main(String[] args) {
        SpringApplication.run(DeepLearningApplication.class, args);
        new MultiLayerNeuralNetwork();
        new MultiLayerNeuralNetworkWithDataSet();
    }
}
