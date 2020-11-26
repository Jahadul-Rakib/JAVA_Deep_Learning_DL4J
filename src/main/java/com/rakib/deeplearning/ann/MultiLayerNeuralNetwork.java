package com.rakib.deeplearning.ann;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;


public class MultiLayerNeuralNetwork {

    public MultiLayerNeuralNetwork() {
        //1. Configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .miniBatch(false)
                .updater(new Sgd(0.1))
                .list()
                .layer(new DenseLayer
                        .Builder()
                        .nIn(2)
                        .nOut(4)
                        .weightInit(new UniformDistribution(0, 1))
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(new DenseLayer
                        .Builder()
                        .nIn(4)
                        .nOut(4)
                        .weightInit(new UniformDistribution(0, 1))
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(new OutputLayer
                        .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(4).nOut(2)
                        .activation(Activation.SOFTMAX)
                        .weightInit(new UniformDistribution(0, 1))
                        .build())
                .build();

        //2. Network
        MultiLayerNetwork network = new MultiLayerNetwork(conf);

        //3. prepare dataset
        double[][] features = new double[][]{{0, 1}, {1, 0}, {0, 0}, {1, 1}};
        double[][] labels = new double[][]{{0, 1}, {1, 0}, {1, 0}, {1, 0}};
        DataSet dataSet = new DataSet(Nd4j.create(features), Nd4j.create(labels));

        //4. fit model and epoch
        for (int i = 1; i <= 1000; i++) {
            network.fit(dataSet);
        }
        //5. evaluation
        Evaluation evaluation = new Evaluation();
        evaluation.eval(dataSet.getLabels(), network.output(dataSet.getFeatures()));
        System.out.println(evaluation.stats());

        //6. Predict
        System.out.println(network.output(Nd4j.create(new double[][]{{1, 0}})));
    }
}
