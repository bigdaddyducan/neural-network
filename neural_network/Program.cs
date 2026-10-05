
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
        this.weights = new double[n_inputs,n_neurons];
        for (int i = 0; i < n_inputs; i++)
        {
            for (int j = 0; j < n_neurons; j++)
            {
                weights[i, j] = rand.NextDouble() * 2 - 1;
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
    double[,] TestInputs,
    int[] TestLabels
);

public static class KanjiSplit
{
    public static KanjiDataSplit Create(
        KanjiImageDataset allData,
        int trainPerClass,
        int seed
    )
    {
        const int classCount = 4;

        Random random = new Random(seed);

        List<int> trainIndices = new List<int>();
        List<int> testIndices = new List<int>();

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
                else
                {
                    testIndices.Add(classIndices[i]);
                }
            }
        }

        return new KanjiDataSplit(
            CopyRows(allData.Inputs, allData.Labels, trainIndices),
            CopyLabels(allData.Labels, trainIndices),
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

class program
{
    static void Main(string[] args)
    {
        KanjiImageDataset dataset =
        kanjiDataset.LoadAllFromEnvironment();
        KanjiDataSplit split = KanjiSplit.Create(
    dataset,
    trainPerClass: 160,
    seed: 12345
);

double[,] StartData = split.TrainInputs;
int[] labels = split.TrainLabels;




        
        Layer layer1 = new Layer(StartData.GetLength(1), 100);
        Layer layer2 = new Layer(100, 4);

        ActivationFunction activationFunction = new ActivationFunction();
        ActivationSoftmax activationSoftmax = new ActivationSoftmax();
        LossCategoricalCrossentropy Loss_Function = new LossCategoricalCrossentropy();
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
        foreach (int epoch in Enumerable.Range(0, 11))
        {
            outputs = layer1.forward(StartData); 
            outputs = activationFunction.Forward(outputs);
            outputs = layer2.forward(outputs); 
            outputs = activationSoftmax.softmax(outputs);
            lossValue = loss.Calculate(outputs, labels);
            accuracy = calculateAccuracy(outputs, labels);
            dInputs = activationSoftmax.Backward(outputs, labels);
            dInputs = layer2.backward(dInputs);
            dInputs = activationFunction.Backward(dInputs);
            dInputs = layer1.backward(dInputs);
            layer1.UpdateParameters(0.2);
            layer2.UpdateParameters(0.2);      
            
            if (epoch % 1 == 0)
            {
                Console.WriteLine($"Epoch: {epoch}, Loss: {lossValue}, Accuracy: {accuracy}");
            }
        }  
    }
}
