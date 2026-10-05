using System.Drawing;
using System.Drawing.Drawing2D;
public static class kanjiDataset
{
    private sealed record KanjiClass(int Label,string Kanji,string FolderName);
    private static readonly KanjiClass[] Classes = [
        new KanjiClass(0, "雨", "0x96e8"),
        new KanjiClass(1, "山", "0x5c71"),
        new KanjiClass(2, "川", "0x5ddd"),
        new KanjiClass(3, "大", "0x5927")
    ];
    public static void PrintSummaryFromEnvironment()
    {
        string? root = Environment.GetEnvironmentVariable("KANJI_DATA_ROOT");

        if (string.IsNullOrWhiteSpace(root))
        {
            throw new InvalidOperationException(
                "KANJI_DATA_ROOT is not set. " +
                "Set it to your ETL9G images folder before running."
            );
        }

        PrintSummary(root);
    }

    public static void PrintSummary(string root)
    {
        if (!Directory.Exists(root))
        {
            throw new DirectoryNotFoundException(
                $"Kanji data folder was not found: {root}"
            );
        }

        int total = 0;

        foreach (KanjiClass kanjiClass in Classes)
        {
            string classFolder = Path.Combine(root, kanjiClass.FolderName);

            if (!Directory.Exists(classFolder))
            {
                throw new DirectoryNotFoundException(
                    $"Missing folder for {kanjiClass.Kanji}: {classFolder}"
                );
            }

            string[] pngFiles = Directory.GetFiles(
                classFolder,
                "*.png",
                SearchOption.TopDirectoryOnly
            );

            total += pngFiles.Length;

            Console.WriteLine(
                $"Label {kanjiClass.Label} | " +
                $"{kanjiClass.Kanji} | " +
                $"{kanjiClass.FolderName} | " +
                $"{pngFiles.Length} PNGs"
            );
        }

        Console.WriteLine($"Total: {total} PNGs");
        PrintFirstImageDetails(root);
    }
    private static void PrintFirstImageDetails(string root)
    {
    KanjiClass firstClass = Classes[0];

    string classFolder = Path.Combine(root, firstClass.FolderName);

    string firstImage = Directory
        .GetFiles(classFolder, "*.png", SearchOption.TopDirectoryOnly)
        .OrderBy(path => path, StringComparer.Ordinal)
        .First();

    using Bitmap image = new Bitmap(firstImage);

    Color topLeft = image.GetPixel(0, 0);

    double grayscale =
        (0.299 * topLeft.R +
         0.587 * topLeft.G +
         0.114 * topLeft.B) / 255.0;

    Console.WriteLine();
    Console.WriteLine($"Example image: {Path.GetFileName(firstImage)}");
    Console.WriteLine($"Class: {firstClass.Kanji} (label {firstClass.Label})");
    Console.WriteLine($"Dimensions: {image.Width} x {image.Height}");
    Console.WriteLine(
        $"Pixel [0, 0]: RGB({topLeft.R}, {topLeft.G}, {topLeft.B}), " +
        $"grayscale: {grayscale}"
    );

    double[] features = LoadImageAsFeatures(firstImage);

Console.WriteLine($"Resized dimensions: 64 x 64");
Console.WriteLine($"Feature count: {features.Length}");
Console.WriteLine($"Ink range: {features.Min()} to {features.Max()}");
Console.WriteLine(
    $"Non-zero ink features: {features.Count(value => value > 0.0)}"
);

    }
    private static double[] LoadImageAsFeatures(string imagePath)
    {
    const int targetWidth = 64;
    const int targetHeight = 64;

    using Bitmap original = new Bitmap(imagePath);
    using Bitmap resized = new Bitmap(targetWidth, targetHeight);
    using Graphics graphics = Graphics.FromImage(resized);

    graphics.Clear(Color.White);
    graphics.InterpolationMode = InterpolationMode.HighQualityBicubic;
    graphics.DrawImage(
        original,
        new Rectangle(0, 0, targetWidth, targetHeight)
    );

    double[] features = new double[targetWidth * targetHeight];

    for (int y = 0; y < targetHeight; y++)
    {
        for (int x = 0; x < targetWidth; x++)
        {
            Color pixel = resized.GetPixel(x, y);

            double grayscale =
                (0.299 * pixel.R +
                 0.587 * pixel.G +
                 0.114 * pixel.B) / 255.0;

            double ink = 1.0 - grayscale;

            int featureIndex = y * targetWidth + x;
            features[featureIndex] = ink;
        }
    }

    return features;
    }


}