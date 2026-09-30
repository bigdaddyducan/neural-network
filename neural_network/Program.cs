class perceptron
{
    Random random = new Random();
    double[] weights;
    double learningRateConstant;
    perceptron()
    {
        this.weights = [];
    }

    public perceptron(int n)
    {
        this.weights = new double[n];
        this.learningRateConstant = 0.01;
        for (int i = 0; i < n; i++)
        {
            this.weights[i] = random.NextDouble() * 2 - 1;
        }
    }
    double activate(double sum)
    {
        return sum > 0 ? 1 : -1;
    }

    public double feedForward(double[] inputs)
    {
        double sum = 0;
        for(int i = 0; i < this.weights.Length; i++)
        {
            sum += inputs[i] * this.weights[i];
        }
        return activate(sum);
    }

    public void train(double[] inputs, double desired)
    {
        double guess = feedForward(inputs);
        double error = desired - guess;
        for(int i = 0; i < this.weights.Length; i++)
        {
            this.weights[i] += error * inputs[i] * this.learningRateConstant;
        }
    }

}
class Program{
    static void Main(string[] args)
    {
        perceptron p = new perceptron(3);
        double[] inputs = {50,-12,1};
        double guess = p.feedForward(inputs);

        Console.WriteLine(guess);
    }
}

class getTrainingData
{
    Random random = new Random();
     double f(int x)
    {
        return 0.5 * x - 1;
    }
    int x = 0;
    int y= 0;
    int desired = -1;
    double yline = 0;
    int[] trainingInputs = new int[3];

    public getTrainingData()
    {
        x = random.Next(-100, 100);
        y = random.Next(-100, 100);
        yline = f(x);
        if(y > yline)
        {
            desired = 1;
        }
        trainingInputs[0] = x;
        trainingInputs[1] = y;
        trainingInputs[2] = desired;
    }
    
}