
public class ActivationFunction()
{
    public double[,] ReLUCache;
    public double[,] dInputs;
    void ReLUCaching(double[,] x)
    {
        ReLUCache = new double[x.GetLength(0), x.GetLength(1)];
        for (int i = 0; i < x.GetLength(0); i++)
        {
            for (int j = 0; j < x.GetLength(1); j++)
            {
                ReLUCache[i,j] = x[i,j];
            }
        }
    }
    public double[,] Forward(double[,] x)
    {
        ReLUCaching(x);
        //ReLU activation function
        for (int i = 0; i < x.GetLength(0); i++)
        {
            for (int j = 0; j < x.GetLength(1); j++)
            {
                x[i,j] = x[i,j] > 0 ? x[i,j] : 0;
            }
        }
        return x;
    }
    public double[,] Backward(double[,] dValues)
    {
        dInputs = new double[dValues.GetLength(0), dValues.GetLength(1)];
        for (int i = 0; i < dValues.GetLength(0);i++)
        {
            for (int j = 0; j < dValues.GetLength(1);j++)
            {
                if (ReLUCache[i,j] <= 0)
                {
                    dInputs[i,j] = 0;
                }
                else
                {
                    dInputs[i,j] = dValues[i,j];
                }
            }
        }
        return dInputs;
    }
}
public class ActivationSoftmax()
{
    public double[,] softmax(double[,] outputs)
    {
        double[,] exp_values = new double[outputs.GetLength(0), outputs.GetLength(1)];
        double[,] max_values = new double[outputs.GetLength(0), 1];
        for (int i = 0; i < outputs.GetLength(0); i++)
        {
            max_values[i, 0] = outputs[i, 0];
            for (int j = 1; j < outputs.GetLength(1); j++)
            {
                if (outputs[i, j] > max_values[i, 0])
                {
                    max_values[i, 0] = outputs[i, j];
                }
            }
        }
        for (int i = 0; i < outputs.GetLength(0); i++)
        {
            for (int j = 0; j < outputs.GetLength(1); j++)
            {
                exp_values[i, j] = Math.Exp(outputs[i, j] - max_values[i, 0]);
            }
        }
        double[,] norm_values = new double[exp_values.GetLength(0), exp_values.GetLength(1)];
        double[,] sum_values = new double[exp_values.GetLength(0), 1];
        for (int i = 0; i < exp_values.GetLength(0); i++)
        {
            for (int j = 0; j < exp_values.GetLength(1); j++)
            {
                sum_values[i, 0] += exp_values[i, j];
            }
        }
        for (int i = 0; i < exp_values.GetLength(0); i++)
        {
            for (int j = 0; j < exp_values.GetLength(1); j++)
            {
                norm_values[i, j] = exp_values[i, j] / sum_values[i, 0];
            }
        }
        
        return norm_values;
    }
    public double[,] Backward(double[,] Probabilites, int[] lables)
    {
        int samples = Probabilites.GetLength(0);
        int classes = Probabilites.GetLength(1);
        double[,] dInputs = new double[samples, classes];
        int correctClass;
        for (int i = 0; i < samples; i++)
        {
            for (int j = 0; j < classes; j++)
            {
                dInputs[i, j] = Probabilites[i, j];
            }
        }

        for(int i = 0;i< samples;i++)
        {
            correctClass = lables[i];
            dInputs[i,correctClass] -= 1;
        }

        for (int i = 0; i < samples; i++)
        {
            for (int j = 0; j < classes; j++)
            {
                dInputs[i, j] /= samples;
            }
        }
        return dInputs;
    }
}

public class Loss()
{
    public double Calculate(double[,] y_pred, int[] labels)
    {
        LossCategoricalCrossentropy loss = new LossCategoricalCrossentropy();
        double[] sampleLosses = loss.forward(y_pred, labels);
        double TotalLoss = 0;

        for (int i = 0; i < sampleLosses.Length; i++)
        {
            TotalLoss += sampleLosses[i];
        }
        return TotalLoss / sampleLosses.Length;
    }
}
public class LossCategoricalCrossentropy : Loss
{
    public double[] forward(double[,] y_pred, int[] labels)
    {
        int samples = y_pred.GetLength(0);
        int classes = y_pred.GetLength(1);
        double[] negativeLogLikelihoods = new double[samples];
    
        for (int i = 0; i < samples; i++)
        {
            int correctClass = labels[i];
            double correct_confidence = y_pred[i, correctClass];
            double clipped_confidence = Math.Clamp(correct_confidence, 1e-7, 1 - 1e-7);
            negativeLogLikelihoods[i] = -Math.Log(clipped_confidence);
        }
        return negativeLogLikelihoods;
    }
}

public class Layer
{
    public double[,] LayerCache;
    Random rand = new Random();
    public double[,] weights;
    public double[] biases;
    public double[,] dWeights;
    public double[] dBiases;
    public double[,] dInputs;
    public Layer(int n_inputs, int n_neurons)
    {
        double limit = Math.Sqrt(6.0 / (n_inputs + n_neurons));

        this.weights = new double[n_inputs,n_neurons];
        for (int i = 0; i < n_inputs; i++)
        {
            for (int j = 0; j < n_neurons; j++)
            {
                weights[i, j] = (rand.NextDouble() * 2 - 1) * limit;

            }
        }
        this.biases = new double[n_neurons];
        for (int i = 0; i < n_neurons; i++)
        {
            biases[i] = 0;
        }
    }
    public double[,] forward(double[,] inputs)
    {
        double[,] outputs = new double[inputs.GetLength(0), weights.GetLength(1)];
        for (int i = 0; i < inputs.GetLength(0); i++)
        {
            for (int j = 0; j < weights.GetLength(1); j++)
            {
                double sum = 0;
                for (int k = 0; k < weights.GetLength(0); k++)
                {
                    sum += inputs[i, k] * weights[k, j];
                }
                sum += biases[j];
                outputs[i, j] = sum;
            }
        }
        this.LayerCache = inputs;
        return outputs;
    }
    public double[,] backward(double[,] dValues)
    {
        dWeights = new double[LayerCache.GetLength(1), dValues.GetLength(1)];
        dBiases = new double[dValues.GetLength(1)];
        dInputs = new double[dValues.GetLength(0), LayerCache.GetLength(1)];

        double total;
        for (int i = 0; i < LayerCache.GetLength(1); i++)
        {
            for (int j = 0; j < dValues.GetLength(1); j++)
            {
                total = 0;
                for (int k = 0; k < dValues.GetLength(0); k++)
                {
                    total += LayerCache[k, i] * dValues[k, j];
                }
                dWeights[i, j] = total;
            }
        }
        for(int i = 0; i < dValues.GetLength(1); i++)
        {
            total = 0;
            for(int j = 0; j < dValues.GetLength(0); j++)
            {
                total += dValues[j, i];
            }
            dBiases[i] = total;
        } 
        for(int i = 0; i < dValues.GetLength(0); i++)
        {
            for (int j = 0; j < LayerCache.GetLength(1); j++)
            {
                total = 0;
                for (int k = 0; k < dValues.GetLength(1); k++)
                {
                    total += dValues[i, k] * weights[j, k];
                }
                dInputs[i, j] = total;
            }
        }
        return dInputs;
    }
    public void UpdateParameters(double learningRate)
    {
        for (int i = 0;i < weights.GetLength(0);i++)
        {
            for (int j = 0;j < weights.GetLength(1);j++)
            {
                weights[i,j] -= learningRate * dWeights[i,j];
            }
        }
        for(int i = 0;i < biases.Length;i++)
        {
            biases[i] -= learningRate * dBiases[i];
        }
    }
}
public sealed record KanjiDataSplit(
    double[,] TrainInputs,
    int[] TrainLabels,
    double[,] ValidationInputs,
    int[] ValidationLabels,
    double[,] TestInputs,
    int[] TestLabels
);


public static class KanjiSplit
{
    public static KanjiDataSplit Create(
        KanjiImageDataset allData,
        int trainPerClass,
        int validationPerClass,
        int seed
    )
    {
        int classCount = allData.Labels.Max() + 1;


        Random random = new Random(seed);

        List<int> trainIndices = new List<int>();
        List<int> testIndices = new List<int>();
        List<int> validationIndices = new List<int>();

        for (int label = 0; label < classCount; label++)
        {
            List<int> classIndices = new List<int>();

            for (int sample = 0; sample < allData.Labels.Length; sample++)
            {
                if (allData.Labels[sample] == label)
                {
                    classIndices.Add(sample);
                }
            }

            if (classIndices.Count <= trainPerClass)
            {
                throw new InvalidOperationException(
                    $"Label {label} has only {classIndices.Count} samples; " +
                    $"it needs more than {trainPerClass}."
                );
            }

            Shuffle(classIndices, random);

            for (int i = 0; i < classIndices.Count; i++)
            {
                if (i < trainPerClass)
                {
                    trainIndices.Add(classIndices[i]);
                    }
else if (i < trainPerClass + validationPerClass)
{
    validationIndices.Add(classIndices[i]);
}
else
{
    testIndices.Add(classIndices[i]);
}

            }
        }

return new KanjiDataSplit(
    CopyRows(allData.Inputs, allData.Labels, trainIndices),
    CopyLabels(allData.Labels, trainIndices),

    CopyRows(allData.Inputs, allData.Labels, validationIndices),
    CopyLabels(allData.Labels, validationIndices),

    CopyRows(allData.Inputs, allData.Labels, testIndices),
    CopyLabels(allData.Labels, testIndices)
);

    }

    private static void Shuffle(List<int> values, Random random)
    {
        for (int i = values.Count - 1; i > 0; i--)
        {
            int swapIndex = random.Next(i + 1);

            int temporary = values[i];
            values[i] = values[swapIndex];
            values[swapIndex] = temporary;
        }
    }

    private static double[,] CopyRows(
        double[,] sourceInputs,
        int[] sourceLabels,
        List<int> sourceIndices
    )
    {
        int featureCount = sourceInputs.GetLength(1);

        double[,] result = new double[
            sourceIndices.Count,
            featureCount
        ];

        for (int row = 0; row < sourceIndices.Count; row++)
        {
            int sourceRow = sourceIndices[row];

            for (int feature = 0; feature < featureCount; feature++)
            {
                result[row, feature] = sourceInputs[sourceRow, feature];
            }
        }

        return result;
    }

    private static int[] CopyLabels(
        int[] sourceLabels,
        List<int> sourceIndices
    )
    {
        int[] result = new int[sourceIndices.Count];

        for (int i = 0; i < sourceIndices.Count; i++)
        {
            result[i] = sourceLabels[sourceIndices[i]];
        }

        return result;
    }
}

public static class BatchBuilder
{
    public static int[] CreateShuffledIndices(
        int sampleCount,
        Random random
    )
    {
        int[] indices = Enumerable.Range(0, sampleCount).ToArray();

        for (int index = indices.Length - 1; index > 0; index--)
        {
            int swapIndex = random.Next(index + 1);

            int temporary = indices[index];
            indices[index] = indices[swapIndex];
            indices[swapIndex] = temporary;
        }

        return indices;
    }

    public static (double[,] Inputs, int[] Labels) CreateBatch(
        double[,] sourceInputs,
        int[] sourceLabels,
        int[] shuffledIndices,
        int startIndex,
        int batchSize
    )
    {
        int featureCount = sourceInputs.GetLength(1);

        double[,] batchInputs = new double[batchSize, featureCount];
        int[] batchLabels = new int[batchSize];

        for (int batchRow = 0; batchRow < batchSize; batchRow++)
        {
            int sourceRow = shuffledIndices[startIndex + batchRow];

            for (int feature = 0;
                 feature < featureCount;
                 feature++)
            {
                batchInputs[batchRow, feature] =
                    sourceInputs[sourceRow, feature];
            }

            batchLabels[batchRow] =
                sourceLabels[sourceRow];
        }

        return (batchInputs, batchLabels);
    }
}

class program
{
    static void Main(string[] args)
    {
if (args.Length == 1 &&
    args[0].Equals(
        "scan-kanji",
        StringComparison.OrdinalIgnoreCase
    ))
{
    kanjiDataset.PrintSummaryFromEnvironment();
    return;
}


        if (args.Length == 1 &&
    args[0].Equals(
        "load-model",
        StringComparison.OrdinalIgnoreCase
    ))
{
    LoadedKanjiModel loadedModel =
        ModelPersistence.LoadFromEnvironment();

    Console.WriteLine("Model loaded successfully.");
    Console.WriteLine(
        $"Architecture: " +
        $"{loadedModel.Metadata.InputFeatureCount} → " +
        $"{loadedModel.Metadata.HiddenNeuronCount1} → " +
        $"{loadedModel.Metadata.HiddenNeuronCount2} → " +
        $"{loadedModel.Metadata.OutputClassCount}"
    );

    Console.WriteLine(
        $"Classes: {string.Join(", ", loadedModel.Metadata.ClassKanji)}"
    );

string[] activeKanji =
    kanjiDataset.GetClassKanji();

string[] activeFolders =
    kanjiDataset.GetClassFolders();

if (loadedModel.Metadata.OutputClassCount !=
    kanjiDataset.ClassCount ||
    !loadedModel.Metadata.ClassKanji.SequenceEqual(activeKanji) ||
    !loadedModel.Metadata.ClassFolders.SequenceEqual(activeFolders))
{
    throw new InvalidOperationException(
        "The saved model class mapping does not match " +
        "the active dataset manifest."
    );
}


    KanjiImageDataset loadedDataset =
    kanjiDataset.LoadAllFromEnvironment();

KanjiDataSplit loadedSplit = KanjiSplit.Create(
    loadedDataset,
    trainPerClass: 144,
    validationPerClass: 16,
    seed: 12345
);


if (loadedSplit.TestInputs.GetLength(1) !=
    loadedModel.Metadata.InputFeatureCount)
{
    throw new InvalidOperationException(
        "The test image feature count does not match " +
        "the saved model."
    );
}

ActivationFunction loadedActivationFunction1 =
    new ActivationFunction();

ActivationFunction loadedActivationFunction2 =
    new ActivationFunction();

ActivationSoftmax loadedActivationSoftmax =
    new ActivationSoftmax();

Loss loadedLoss = new Loss();

double[,] loadedTestOutputs =
    loadedModel.Layer1.forward(loadedSplit.TestInputs);

loadedTestOutputs =
    loadedActivationFunction1.Forward(loadedTestOutputs);

loadedTestOutputs =
    loadedModel.Layer2.forward(loadedTestOutputs);

loadedTestOutputs =
    loadedActivationFunction2.Forward(loadedTestOutputs);

loadedTestOutputs =
    loadedModel.Layer3.forward(loadedTestOutputs);

loadedTestOutputs =
    loadedActivationSoftmax.softmax(loadedTestOutputs);

double loadedTestLoss = loadedLoss.Calculate(
    loadedTestOutputs,
    loadedSplit.TestLabels
);

double loadedTestAccuracy = calculateAccuracy(
    loadedTestOutputs,
    loadedSplit.TestLabels
);

Console.WriteLine();
Console.WriteLine(
    $"Saved model test loss: {loadedTestLoss}"
);

Console.WriteLine(
    $"Saved model test accuracy: {loadedTestAccuracy}"
);

return;
}

        KanjiImageDataset dataset =
        kanjiDataset.LoadAllFromEnvironment();
        KanjiDataSplit split = KanjiSplit.Create(
    dataset,
    trainPerClass: 144,
    validationPerClass: 16,
    seed: 12345
);
Console.WriteLine(
    $"Training: {split.TrainInputs.GetLength(0)} x " +
    $"{split.TrainInputs.GetLength(1)}"
);

Console.WriteLine(
    $"Validation: {split.ValidationInputs.GetLength(0)} x " +
    $"{split.ValidationInputs.GetLength(1)}"
);

Console.WriteLine(
    $"Test: {split.TestInputs.GetLength(0)} x " +
    $"{split.TestInputs.GetLength(1)}"
);



double[,] StartData = split.TrainInputs;
int[] labels = split.TrainLabels;
int outputClassCount = labels.Max() + 1;
        
        Layer layer1 = new Layer(StartData.GetLength(1), 128);
        Layer layer2 = new Layer(128, 64);
        Layer layer3 = new Layer(64,outputClassCount);
   


        ActivationFunction activationFunction1 = new ActivationFunction();
        ActivationFunction activationFunction2 = new ActivationFunction();
        ActivationSoftmax activationSoftmax = new ActivationSoftmax();
        Loss loss = new Loss();
        double[,] outputs = new double[StartData.GetLength(0), StartData.GetLength(1)];
        double lossValue;
        double[,] dInputs;
        double accuracy;

        double calculateAccuracy(double[,] outputs, int[] labels)
        {
            int correctPredictions = 0;
            for (int i = 0; i < outputs.GetLength(0); i++)
            {
                int predictedClass = 0;
                double maxProbability = outputs[i, 0];
                for (int j = 1; j < outputs.GetLength(1); j++)
                {
                    if (outputs[i, j] > maxProbability)
                    {
                        maxProbability = outputs[i, j];
                        predictedClass = j;
                    }
                }
                if (predictedClass == labels[i])
                {
                    correctPredictions++;
                }
            }
            return (double)correctPredictions / outputs.GetLength(0);
        }

        const int batchSize = 64;
        Random batchRandom = new Random(67890);

foreach (int epoch in Enumerable.Range(0, 100))
{
    int[] shuffledIndices =
        BatchBuilder.CreateShuffledIndices(
            StartData.GetLength(0),
            batchRandom
        );

    double totalLoss = 0.0;
    double totalCorrect = 0.0;

    for (int startIndex = 0;
         startIndex < StartData.GetLength(0);
         startIndex += batchSize)
    {
        (double[,] batchInputs, int[] batchLabels) =
            BatchBuilder.CreateBatch(
                StartData,
                labels,
                shuffledIndices,
                startIndex,
                batchSize
            );

        outputs = layer1.forward(batchInputs);
        outputs = activationFunction1.Forward(outputs);

        outputs = layer2.forward(outputs);
        outputs = activationFunction2.Forward(outputs);

        outputs = layer3.forward(outputs);
        outputs = activationSoftmax.softmax(outputs);

        lossValue = loss.Calculate(outputs, batchLabels);
        accuracy = calculateAccuracy(outputs, batchLabels);

        totalLoss += lossValue * batchSize;
        totalCorrect += accuracy * batchSize;

        dInputs = activationSoftmax.Backward(
            outputs,
            batchLabels
        );

        dInputs = layer3.backward(dInputs);

        dInputs = activationFunction2.Backward(dInputs);
        dInputs = layer2.backward(dInputs);

        dInputs = activationFunction1.Backward(dInputs);
        dInputs = layer1.backward(dInputs);

        layer1.UpdateParameters(0.2);
        layer2.UpdateParameters(0.2);
        layer3.UpdateParameters(0.2);
    }

    double epochLoss =
        totalLoss / StartData.GetLength(0);

    double epochAccuracy =
        totalCorrect / StartData.GetLength(0);

    Console.WriteLine(
        $"Epoch: {epoch}, " +
        $"Loss: {epochLoss}, " +
        $"Accuracy: {epochAccuracy}"
    );
}


 
        double[,] testOutputs = layer1.forward(split.TestInputs);
        testOutputs = activationFunction1.Forward(testOutputs);
        testOutputs = layer2.forward(testOutputs);
        testOutputs = activationFunction2.Forward(testOutputs);
        testOutputs = layer3.forward(testOutputs);
        testOutputs = activationSoftmax.softmax(testOutputs);
        double testLoss = loss.Calculate(testOutputs,split.TestLabels);
        double testAccuracy = calculateAccuracy(testOutputs,split.TestLabels);
        Console.WriteLine();
        Console.WriteLine($"Test loss: {testLoss}");
        Console.WriteLine($"Test accuracy: {testAccuracy}");

        string modelPath = Path.Combine(
    "models",
    "kanji-n5-32x32-v1.json"
);

ModelPersistence.Save(
    modelPath,
    layer1,
    layer2,
    layer3,
    kanjiDataset.GetClassKanji(),
    kanjiDataset.GetClassFolders()
);


Console.WriteLine(
    $"Saved model: {Path.GetFullPath(modelPath)}"
);

    }
}
