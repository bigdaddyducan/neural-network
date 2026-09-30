class perceptron
{
    Random random = new Random();
    double[] weights;
    perceptron()
    {
        this.weights = [];
    }

    public perceptron(int n)
    {
        this.weights = new double[n];
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