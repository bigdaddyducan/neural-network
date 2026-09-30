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
        //return output = activationFunction(sum);
    }
}

public class Layer
{
    Random rand = new Random();
    public double[,] weights;
    public Layer(int n_inputs, int n_neurons)
    {
        this.weights = new double[n_inputs,n_neurons];
        for (int i = 0; i < n_inputs; i++)
        {
            for (int j = 0; j < n_neurons; j++)
            {
                weights[i, j] = rand.NextDouble();
            }
        }
    }
}

class program
{
    static void Main(string[] args)
    {
        Layer layer1 = new Layer(3, 4);

        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                Console.Write(layer1.weights[i, j] + " ");
            }
            Console.WriteLine();
        }
    }
}
