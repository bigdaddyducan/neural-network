//creating a neuron
class neuron
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
        return output = activationFunction(sum);
    }
}

class Layer
{
    
}
