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
public class Layer
{
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
        double[,] StartData = new double[,] { { 1, 2, 3, 1.5 }, { -0.5, -0.75, 2, -1.0}, { -1.5, 2.7, 3.3, -0.8} };

        Layer layer1 = new Layer(4, 2);
        ActivationFunction activationFunction = new ActivationFunction();
        double[,] output = layer1.forward(StartData);

        for(int i =0; i < layer1.forward(StartData).GetLength(0); i++)
        {
            for(int j = 0; j < layer1.forward(StartData).GetLength(1); j++)
            {
                Console.Write(layer1.forward(StartData)[i, j] + " ");
            }
            Console.WriteLine();
        }
        Console.WriteLine("After Activation Function:");
        activationFunction.Forward(layer1.forward(StartData));

        output = activationFunction.Forward(layer1.forward(StartData));

        for (int i = 0; i < output.GetLength(0); i++)
        {
            for (int j = 0; j < output.GetLength(1); j++)
            {
                Console.Write(output[i, j] + " ");
            }
            Console.WriteLine();
        }
        double E = 2.71828182846;
        double[,] exp_values = new double[output.GetLength(0), output.GetLength(1)];

        for (int i = 0; i < output.GetLength(0); i++)
        {
            for (int j = 0; j < output.GetLength(1); j++)
            {
                exp_values[i, j] = Math.Pow(E, output[i, j]);
            }
        }
        Console.WriteLine("After softmax Function:");
        for (int i = 0; i < exp_values.GetLength(0); i++)
        {
            for (int j = 0; j < exp_values.GetLength(1); j++)
            {
                Console.Write(exp_values[i, j] + " ");
            }
            Console.WriteLine();
        }
        double [,] Normalisation(double[,] exp_values)
        {
            double norm_base = 0;
            double[,] norm_values = new double[exp_values.GetLength(0), exp_values.GetLength(1)];
            for (int i = 0; i < exp_values.GetLength(0); i++)
            {
                for (int j = 0; j < exp_values.GetLength(1); j++)
                {
                    norm_base += exp_values[i, j];
                }
            }
            for (int i = 0; i < exp_values.GetLength(0); i++)
            {
                for (int j = 0; j < exp_values.GetLength(1); j++)
                {
                    norm_values[i, j] = exp_values[i, j] / norm_base;
                }
            }
            return norm_values;
        }
        output = Normalisation(exp_values);
        for (int i = 0; i < output.GetLength(0); i++)
        {
            for (int j = 0; j < output.GetLength(1); j++)
            {
                Console.Write(output[i, j] + " ");
            }
            Console.WriteLine();
        }
    }
}
