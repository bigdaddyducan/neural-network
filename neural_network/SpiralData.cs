using System;

/// <summary>
/// A two-dimensional, labelled spiral dataset equivalent to Sentdex's
/// Python <c>spiral_data(points, classes)</c> helper.
/// </summary>
public sealed record SpiralDataset(double[,] Inputs, int[] Labels);

public static class SpiralData
{
    /// <summary>
    /// Creates <paramref name="classes"/> interleaving spiral classes.
    /// Each input row contains <c>[x, y]</c>; Labels[row] is its class index.
    /// </summary>
    public static SpiralDataset Generate(
        int pointsPerClass,
        int classes,
        Random? random = null)
    {
        if (pointsPerClass <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(pointsPerClass), "Points per class must be positive.");

        if (classes <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(classes), "Class count must be positive.");

        random ??= Random.Shared;

        int totalPoints = checked(pointsPerClass * classes);
        double[,] inputs = new double[totalPoints, 2];
        int[] labels = new int[totalPoints];

        for (int classNumber = 0; classNumber < classes; classNumber++)
        {
            for (int point = 0; point < pointsPerClass; point++)
            {
                int row = classNumber * pointsPerClass + point;

                // Equivalent to NumPy linspace(0.0, 1.0, points).
                double radius = pointsPerClass == 1
                    ? 0.0
                    : (double)point / (pointsPerClass - 1);

                // Equivalent to:
                // linspace(class * 4, (class + 1) * 4, points)
                // + randomNormal(0, 0.2)
                double classAngleStart = classNumber * 4.0;
                double classAngleEnd = (classNumber + 1) * 4.0;
                double baseAngle = pointsPerClass == 1
                    ? classAngleStart
                    : classAngleStart
                      + (classAngleEnd - classAngleStart) * point / (pointsPerClass - 1.0);
                double angle = baseAngle + NextGaussian(random) * 0.2;

                inputs[row, 0] = radius * Math.Sin(angle * 2.5);
                inputs[row, 1] = radius * Math.Cos(angle * 2.5);
                labels[row] = classNumber;
            }
        }

        return new SpiralDataset(inputs, labels);
    }

    // Box-Muller transform: returns one normally distributed value (mean 0, SD 1).
    private static double NextGaussian(Random random)
    {
        double u1 = 1.0 - random.NextDouble(); // Never allow log(0).
        double u2 = 1.0 - random.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
