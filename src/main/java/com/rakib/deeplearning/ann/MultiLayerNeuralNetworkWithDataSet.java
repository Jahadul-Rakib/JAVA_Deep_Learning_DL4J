package com.rakib.deeplearning.ann;

import com.rakib.deeplearning.service.CSVReader;
import org.apache.commons.csv.CSVRecord;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;


public class MultiLayerNeuralNetworkWithDataSet {

    public MultiLayerNeuralNetworkWithDataSet() throws IOException {

        List<CSVRecord> recordList = CSVReader.parse(new File("src/main/resources/iris.data"));
        recordList = recordList.stream().filter(v -> v.size() == 5).collect(Collectors.toList());
        System.out.println(recordList.size());

        double[][] features = recordList.stream().map(val -> new double[]{
                Double.parseDouble(val.get(0)),
                Double.parseDouble(val.get(1)),
                Double.parseDouble(val.get(2)),
                Double.parseDouble(val.get(3))}
        ).toArray(value -> new double[value][]);

        double[][] labels = recordList.stream().map(val -> {
            String label = val.get(4);
            if (Objects.equals(label, "Iris-setosa")) {
                return new double[]{1, 0, 0};
            } else if (Objects.equals(label, "Iris-versicolor")) {
                return new double[]{0, 1, 0};
            } else {
                return new double[]{0, 0, 0};
            }
        }).toArray(value -> new double[value][]);

        INDArray featureArray = Nd4j.create(features);
        INDArray labelsArray = Nd4j.create(labels);

        DataSet dataSet = new DataSet(featureArray, labelsArray);
        dataSet.shuffle();
        SplitTestAndTrain testAndTrain = dataSet.splitTestAndTrain(0.7);
        DataSet train = testAndTrain.getTrain();
        DataSet test = testAndTrain.getTest();

        DataNormalization normalization = new NormalizerStandardize();
        normalization.fit(train);
        normalization.transform(train);
        normalization.transform(test);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Sgd(0.1))
                .l2(1e-4)
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.TANH)
                .list()
                .layer(new DenseLayer
                        .Builder()
                        .nIn(4)
                        .nOut(3)
                        .build())
                .layer(new DenseLayer
                        .Builder()
                        .nIn(3)
                        .nOut(3)
                        .build())
                .layer(new OutputLayer
                        .Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nIn(3)
                        .nOut(3)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();


        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        for (int i = 1; i <= 1000; i++) {
            network.fit(train);
        }

        Evaluation evaluation = new Evaluation(3);
        evaluation.eval(test.getLabels(), network.output(test.getFeatures()));
        System.out.println(evaluation.stats());

        System.out.println(network.output(Nd4j.create(new double[][]{{1, 0, 1.5, 3}})));
    }
}
