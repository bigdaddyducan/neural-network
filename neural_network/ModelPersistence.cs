using System.Text.Json;

public sealed class MatrixData
{
    public int Rows { get; set; }
    public int Columns { get; set; }
    public double[] Values { get; set; } = new double[0];
}

public sealed class ModelCheckpoint
{
    public int FormatVersion { get; set; }

    public int InputFeatureCount { get; set; }
    public int HiddenNeuronCount { get; set; }
    public int OutputClassCount { get; set; }

    public int SourceImageWidth { get; set; }
    public int SourceImageHeight { get; set; }
    public int TargetImageWidth { get; set; }
    public int TargetImageHeight { get; set; }

    public string PixelConvention { get; set; } = "";
    public string FlatteningOrder { get; set; } = "";

    public string[] ClassKanji { get; set; } = new string[0];
    public string[] ClassFolders { get; set; } = new string[0];

    public MatrixData Layer1Weights { get; set; } = new MatrixData();
    public double[] Layer1Biases { get; set; } = new double[0];

    public MatrixData Layer2Weights { get; set; } = new MatrixData();
    public double[] Layer2Biases { get; set; } = new double[0];
}

public static class ModelPersistence
{
    private static readonly JsonSerializerOptions JsonOptions =
        new JsonSerializerOptions
        {
            WriteIndented = true
        };

    public static void Save(
        string modelPath,
        Layer layer1,
        Layer layer2
    )
    {
        if (File.Exists(modelPath))
        {
            throw new InvalidOperationException(
                $"Refusing to overwrite existing model: {modelPath}"
            );
        }

        if (layer1.weights.GetLength(1) != layer2.weights.GetLength(0))
        {
            throw new InvalidOperationException(
                "The two layer dimensions do not connect correctly."
            );
        }

        ModelCheckpoint checkpoint = new ModelCheckpoint
        {
            FormatVersion = 1,

            InputFeatureCount = layer1.weights.GetLength(0),
            HiddenNeuronCount = layer1.weights.GetLength(1),
            OutputClassCount = layer2.weights.GetLength(1),

            SourceImageWidth = 128,
            SourceImageHeight = 127,
            TargetImageWidth = 64,
            TargetImageHeight = 64,

            PixelConvention =
                "grayscale converted to ink strength: 1.0 - grayscale",

            FlatteningOrder =
                "row-major: featureIndex = y * 64 + x",

            ClassKanji = new[] { "雨", "山", "川", "大" },
            ClassFolders = new[]
            {
                "0x96e8",
                "0x5c71",
                "0x5ddd",
                "0x5927"
            },

            Layer1Weights = ToMatrixData(layer1.weights),
            Layer1Biases = (double[])layer1.biases.Clone(),

            Layer2Weights = ToMatrixData(layer2.weights),
            Layer2Biases = (double[])layer2.biases.Clone()
        };

        string? directory = Path.GetDirectoryName(modelPath);

        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        string json = JsonSerializer.Serialize(
            checkpoint,
            JsonOptions
        );

        File.WriteAllText(modelPath, json);
    }

    private static MatrixData ToMatrixData(double[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int columns = matrix.GetLength(1);

        double[] values = new double[rows * columns];

        for (int row = 0; row < rows; row++)
        {
            for (int column = 0; column < columns; column++)
            {
                int index = row * columns + column;
                values[index] = matrix[row, column];
            }
        }

        return new MatrixData
        {
            Rows = rows,
            Columns = columns,
            Values = values
        };
    }
}
