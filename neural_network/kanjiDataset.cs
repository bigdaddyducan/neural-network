using System.Drawing;
using System.Drawing.Drawing2D;
public sealed record KanjiImageDataset(
    double[,] Inputs,
    int[] Labels
);

public static class kanjiDataset
{
private sealed record KanjiClass(
    int Label,
    string Kanji
)
{
    public string FolderName =>
        $"0x{char.ConvertToUtf32(Kanji, 0):x4}";
}

private static readonly KanjiClass[] Classes =
[
    // Numbers and currency
    new KanjiClass(0, "一"),
    new KanjiClass(1, "二"),
    new KanjiClass(2, "三"),
    new KanjiClass(3, "四"),
    new KanjiClass(4, "五"),
    new KanjiClass(5, "六"),
    new KanjiClass(6, "七"),
    new KanjiClass(7, "八"),
    new KanjiClass(8, "九"),
    new KanjiClass(9, "十"),
    new KanjiClass(10, "百"),
    new KanjiClass(11, "千"),
    new KanjiClass(12, "万"),
    new KanjiClass(13, "円"),

    // Time and calendar
    new KanjiClass(14, "日"),
    new KanjiClass(15, "月"),
    new KanjiClass(16, "火"),
    new KanjiClass(17, "水"),
    new KanjiClass(18, "木"),
    new KanjiClass(19, "金"),
    new KanjiClass(20, "土"),
    new KanjiClass(21, "年"),
    new KanjiClass(22, "時"),
    new KanjiClass(23, "分"),
    new KanjiClass(24, "半"),

    // Position and direction
    new KanjiClass(25, "上"),
    new KanjiClass(26, "下"),
    new KanjiClass(27, "中"),
    new KanjiClass(28, "外"),
    new KanjiClass(29, "左"),
    new KanjiClass(30, "右"),
    new KanjiClass(31, "前"),
    new KanjiClass(32, "後"),
    new KanjiClass(33, "東"),
    new KanjiClass(34, "西"),
    new KanjiClass(35, "南"),
    new KanjiClass(36, "北"),

    // People
    new KanjiClass(37, "人"),
    new KanjiClass(38, "子"),
    new KanjiClass(39, "女"),
    new KanjiClass(40, "男"),
    new KanjiClass(41, "父"),
    new KanjiClass(42, "母"),
    new KanjiClass(43, "友"),
    new KanjiClass(44, "私"),

    // Education and language
    new KanjiClass(45, "学"),
    new KanjiClass(46, "校"),
    new KanjiClass(47, "生"),
    new KanjiClass(48, "先"),
    new KanjiClass(49, "何"),
    new KanjiClass(50, "本"),
    new KanjiClass(51, "名"),
    new KanjiClass(52, "語"),
    new KanjiClass(53, "文"),
    new KanjiClass(54, "字"),

    // Common actions
    new KanjiClass(55, "食"),
    new KanjiClass(56, "飲"),
    new KanjiClass(57, "見"),
    new KanjiClass(58, "聞"),
    new KanjiClass(59, "読"),
    new KanjiClass(60, "書"),
    new KanjiClass(61, "話"),
    new KanjiClass(62, "買"),
    new KanjiClass(63, "行"),
    new KanjiClass(64, "来"),
    new KanjiClass(65, "帰"),
    new KanjiClass(66, "入"),
    new KanjiClass(67, "出"),

    // Descriptions
    new KanjiClass(68, "大"),
    new KanjiClass(69, "小"),
    new KanjiClass(70, "高"),
    new KanjiClass(71, "安"),
    new KanjiClass(72, "新"),
    new KanjiClass(73, "古"),
    new KanjiClass(74, "多"),
    new KanjiClass(75, "少"),
    new KanjiClass(76, "長"),

    // Transport
    new KanjiClass(77, "電"),
    new KanjiClass(78, "車"),
    new KanjiClass(79, "駅")
];

public static int ClassCount => Classes.Length;

public static string[] GetClassKanji()
{
    return Classes
        .OrderBy(kanjiClass => kanjiClass.Label)
        .Select(kanjiClass => kanjiClass.Kanji)
        .ToArray();
}

public static string[] GetClassFolders()
{
    return Classes
        .OrderBy(kanjiClass => kanjiClass.Label)
        .Select(kanjiClass => kanjiClass.FolderName)
        .ToArray();
}

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
    public static KanjiImageDataset LoadAllFromEnvironment()
{
    string? root = Environment.GetEnvironmentVariable("KANJI_DATA_ROOT");

    if (string.IsNullOrWhiteSpace(root))
    {
        throw new InvalidOperationException(
            "KANJI_DATA_ROOT is not set. " +
            "Set it to your ETL9G images folder before running."
        );
    }

    if (!Directory.Exists(root))
    {
        throw new DirectoryNotFoundException(
            $"Kanji data folder was not found: {root}"
        );
    }

    List<double[]> featureRows = new List<double[]>();
    List<int> labels = new List<int>();

    foreach (KanjiClass kanjiClass in Classes)
    {
        string classFolder = Path.Combine(root, kanjiClass.FolderName);

        if (!Directory.Exists(classFolder))
        {
            throw new DirectoryNotFoundException(
                $"Missing folder for {kanjiClass.Kanji}: {classFolder}"
            );
        }

        string[] pngFiles = Directory
            .GetFiles(classFolder, "*.png", SearchOption.TopDirectoryOnly)
            .OrderBy(path => path, StringComparer.Ordinal)
            .ToArray();

        foreach (string imagePath in pngFiles)
        {
            double[] features = LoadImageAsFeatures(imagePath);

            if (features.Length != 1024)
            {
                throw new InvalidOperationException(
                    $"Expected 1024 features, got {features.Length}: {imagePath}"
                );
            }

            featureRows.Add(features);
            labels.Add(kanjiClass.Label);
        }
    }

    double[,] inputs = new double[featureRows.Count, 1024];

    for (int sample = 0; sample < featureRows.Count; sample++)
    {
        for (int feature = 0; feature < 1024; feature++)
        {
            inputs[sample, feature] = featureRows[sample][feature];
        }
    }

    return new KanjiImageDataset(inputs, labels.ToArray());
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
    const int targetWidth = 32;
    const int targetHeight = 32;

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