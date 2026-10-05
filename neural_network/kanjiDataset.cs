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
    }

}