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

        Layer layer1 = new Layer(4, 3);
        ActivationFunction activationFunction = new ActivationFunction();
        ActivationSoftmax softMaxFunction = new ActivationSoftmax();
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
        output = softMaxFunction.softmax(output);
        
        double sum = 0;
        for (int i = 0; i < output.GetLength(0); i++)
        {
            sum = 0;
            for (int j = 0; j < output.GetLength(1); j++)
            {
                sum += output[i, j];
            }
            Console.WriteLine(sum);
        }
        
        
    }
}
