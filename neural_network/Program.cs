double[] inputs = {12,4};
double[] weights = {0.5,-1};
double bias = -0.5;

double sum = 0;

for(int i =0; i<inputs.Length; i++)
{
    sum += inputs[i] * weights[i];
}

sum += bias;

double activate(double sum)
{
    if (sum > 0)
    {
        return sum;
    }
    else
    {
        return 0;
    }
}


Console.WriteLine(activate(sum));
