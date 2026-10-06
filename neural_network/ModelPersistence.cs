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
    public int HiddenNeuronCount1 { get; set; }
    public int HiddenNeuronCount2 { get; set; }
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

    public MatrixData Layer3Weights { get; set; } = new MatrixData();
    public double[] Layer3Biases { get; set; } = new double[0];
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
        Layer layer2,
        Layer layer3,
        string[] classKanji,
        string[] classFolders
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
        if (layer2.weights.GetLength(1) !=
    layer3.weights.GetLength(0))
        {
            throw new InvalidOperationException(
                "Layer 2 and layer 3 dimensions do not connect correctly."
            );
        }


        if (classKanji.Length != layer3.weights.GetLength(1) ||
    classFolders.Length != layer3.weights.GetLength(1))
        {
            throw new InvalidOperationException(
                "The saved class mapping does not match " +
                "the output layer width."
            );
        }



        ModelCheckpoint checkpoint = new ModelCheckpoint
        {
            FormatVersion = 2,

            InputFeatureCount = layer1.weights.GetLength(0),
            HiddenNeuronCount1 = layer1.weights.GetLength(1),
            HiddenNeuronCount2 = layer2.weights.GetLength(1),
            OutputClassCount = layer3.weights.GetLength(1),

            SourceImageWidth = 128,
            SourceImageHeight = 127,
            TargetImageWidth = 32,
            TargetImageHeight = 32,

            PixelConvention =
                "grayscale converted to ink strength: 1.0 - grayscale",

            FlatteningOrder =
                "row-major: featureIndex = y * 32 + x",
            ClassKanji = (string[])classKanji.Clone(),
            ClassFolders = (string[])classFolders.Clone(),


            Layer1Weights = ToMatrixData(layer1.weights),
            Layer1Biases = (double[])layer1.biases.Clone(),

            Layer2Weights = ToMatrixData(layer2.weights),
            Layer2Biases = (double[])layer2.biases.Clone(),

            Layer3Weights = ToMatrixData(layer3.weights),
            Layer3Biases = (double[])layer3.biases.Clone()
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
    public static LoadedKanjiModel LoadFromEnvironment()
    {
        string? modelPath =
            Environment.GetEnvironmentVariable("KANJI_MODEL_PATH");

        if (string.IsNullOrWhiteSpace(modelPath))
        {
            throw new InvalidOperationException(
                "KANJI_MODEL_PATH is not set."
            );
        }

        return Load(modelPath);
    }

    public static LoadedKanjiModel Load(string modelPath)
    {
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException(
                $"Saved model was not found: {modelPath}"
            );
        }

        string json = File.ReadAllText(modelPath);

        ModelCheckpoint? checkpoint =
            JsonSerializer.Deserialize<ModelCheckpoint>(json);

        if (checkpoint is null)
        {
            throw new InvalidOperationException(
                "The model checkpoint could not be read."
            );
        }

        ValidateCheckpoint(checkpoint);

        Layer layer1 = new Layer(
            checkpoint.InputFeatureCount,
            checkpoint.HiddenNeuronCount1
        );

        Layer layer2 = new Layer(
            checkpoint.HiddenNeuronCount1,
            checkpoint.HiddenNeuronCount2
        );
        Layer layer3 = new Layer(
        checkpoint.HiddenNeuronCount2,
        checkpoint.OutputClassCount
    );

        layer1.weights = FromMatrixData(checkpoint.Layer1Weights);
        layer1.biases = (double[])checkpoint.Layer1Biases.Clone();

        layer2.weights = FromMatrixData(checkpoint.Layer2Weights);
        layer2.biases = (double[])checkpoint.Layer2Biases.Clone();

        layer3.weights = FromMatrixData(checkpoint.Layer3Weights);
        layer3.biases = (double[])checkpoint.Layer3Biases.Clone();

        return new LoadedKanjiModel(
            layer1,
            layer2,
            layer3,
            checkpoint
        );
    }

    private static void ValidateCheckpoint(
        ModelCheckpoint checkpoint
    )
    {
        if (checkpoint.FormatVersion != 2)
        {
            throw new InvalidOperationException(
                $"Unsupported checkpoint version: " +
                $"{checkpoint.FormatVersion}"
            );
        }

        if (checkpoint.Layer1Weights.Rows !=
            checkpoint.InputFeatureCount ||
            checkpoint.Layer1Weights.Columns !=
            checkpoint.HiddenNeuronCount1)
        {
            throw new InvalidOperationException(
                "Layer 1 weight dimensions do not match " +
                "the saved architecture."
            );
        }

        if (checkpoint.Layer2Weights.Rows !=
            checkpoint.HiddenNeuronCount1 ||
            checkpoint.Layer2Weights.Columns !=
            checkpoint.HiddenNeuronCount2)
        {
            throw new InvalidOperationException(
                "Layer 2 weight dimensions do not match " +
                "the saved architecture."
            );
        }
        if (checkpoint.Layer3Weights.Rows !=
        checkpoint.HiddenNeuronCount2 ||
        checkpoint.Layer3Weights.Columns !=
        checkpoint.OutputClassCount)
        {
            throw new InvalidOperationException(
                "Layer 3 weight dimensions do not match " +
                "the saved architecture."
            );
        }

        if (checkpoint.Layer1Biases.Length !=
            checkpoint.HiddenNeuronCount1)
        {
            throw new InvalidOperationException(
                "Layer 1 bias count is invalid."
            );
        }

        if (checkpoint.Layer2Biases.Length !=
            checkpoint.HiddenNeuronCount2)
        {
            throw new InvalidOperationException(
                "Layer 2 bias count is invalid."
            );
        }
        if (checkpoint.Layer3Biases.Length !=
        checkpoint.OutputClassCount)
        {
            throw new InvalidOperationException(
                "Layer 3 bias count is invalid."
            );
        }
        if (checkpoint.ClassKanji.Length !=
            checkpoint.OutputClassCount)
        {
            throw new InvalidOperationException(
                "The saved class mapping does not match " +
                "the number of output neurons."
            );
        }
    }

    private static double[,] FromMatrixData(MatrixData matrixData)
    {
        if (matrixData.Rows <= 0 || matrixData.Columns <= 0)
        {
            throw new InvalidOperationException(
                "Saved matrix dimensions must be positive."
            );
        }

        int expectedValueCount =
            matrixData.Rows * matrixData.Columns;

        if (matrixData.Values.Length != expectedValueCount)
        {
            throw new InvalidOperationException(
                "Saved matrix value count does not match " +
                "its dimensions."
            );
        }

        double[,] matrix = new double[
            matrixData.Rows,
            matrixData.Columns
            ];

        for (int row = 0; row < matrixData.Rows; row++)
        {
            for (int column = 0;
                 column < matrixData.Columns;
                 column++)
            {
                int index = row * matrixData.Columns + column;

                matrix[row, column] =
                    matrixData.Values[index];
            }
        }

        return matrix;
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
public sealed record LoadedKanjiModel(
    Layer Layer1,
    Layer Layer2,
    Layer Layer3,
    ModelCheckpoint Metadata
);
