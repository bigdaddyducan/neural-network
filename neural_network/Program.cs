//creating a neuron
public class Neuron
{
    double[] X;
    double[] weights;
    double bias;
    double output;

    double calculateOutput()
    {
        double sum = 0;
        for (int i = 0; i < X.Length; i++)
        {
            sum += X[i] * weights[i];
        }
        sum += bias;
        return sum;
    }

}
public class ActivationFunction()
{
    public double[,] Forward(double[,] x)
    {
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
        /*
        Console.WriteLine("Sum Values:\n\n\n");
        for (int i = 0; i < sum_values.GetLength(0); i++)
        {
            for (int j = 0; j < sum_values.GetLength(1); j++)
            {
                Console.Write(sum_values[i, j] + " ");
            }
            Console.WriteLine();
        }
        
        Console.WriteLine("After Softmax:\n\n\n");
        for (int i = 0; i < norm_values.GetLength(0); i++)
        {
            for (int j = 0; j < norm_values.GetLength(1); j++)
            {
                Console.Write(norm_values[i, j] + " ");
            }
            Console.WriteLine();
        }
        */
        return norm_values;
    }
}

public class Loss()
{
    double forward(double[,] y_pred, double[] y_true)
    {
        int samples = y_pred.GetLength(0);
        double[,] y_pred_clipped = new double[y_pred.GetLength(0), y_pred.GetLength(1)];
        for(int i = 0; i < y_pred.GetLength(0); i++)
        {
            for(int j = 0; j < y_pred.GetLength(1); j++)
            {
                y_pred_clipped[i,j] = Math.Clamp(y_pred[i,j], 1e-7, 1 - 1e-7);
            }
        }
        double[] correct_confidences = new double[Enumerable.Range(),];
    }


}

public class Layer
{
    public double [,] outputs;
    Random rand = new Random();
    public double[,] weights;
    public double[] biases;
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
        return outputs;
    }
}

class program
{
    static void Main(string[] args)
    {
        
         SpiralDataset dataset = SpiralData.Generate(
            pointsPerClass: 100,
            classes: 3,
            random: new Random(0));
            
            double[,] StartData = dataset.Inputs;
            int[] labels = dataset.Labels;


        Layer layer1 = new Layer(2, 3);
        ActivationFunction activationFunction = new ActivationFunction();
        Layer layer2 = new Layer(3, 3);
        ActivationSoftmax activationSoftmax = new ActivationSoftmax();
        double[,] outputs = new double[StartData.GetLength(0), StartData.GetLength(1)];
        outputs = layer1.forward(StartData); 
        outputs = activationFunction.Forward(outputs);
        outputs = layer2.forward(outputs); 
        outputs = activationSoftmax.softmax(outputs);

        for (int i = 0; i < outputs.GetLength(0); i++)
        {
            for (int j = 0; j < outputs.GetLength(1); j++)
            {
                Console.Write(outputs[i, j] + " ");
            }
            Console.WriteLine();
        }
        
    }
}
