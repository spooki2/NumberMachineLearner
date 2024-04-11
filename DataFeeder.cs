namespace numberRecogniser;

using Microsoft.VisualBasic.FileIO;
using System.Collections.Generic;

public static class DataFeeder
{
    public static string CsvFilePath =
        @"C:\Users\spook\RiderProjects\numberRecogniser\numberRecogniser\DATA\mnist_train.csv";

    public static String[] getCsv(int id)
    {
        int label = 0;
        String[] img = new String[784];
        using (var reader = new StreamReader(CsvFilePath))
        {
            // Read the first line of the file
            var line = reader.ReadLine();
            line = reader.ReadLine();
            for (int i = 0; i < id; i++)
            {
                line = reader.ReadLine();
            }

            var values = line.Split(',');
            //skip label + get image id

            int index = 0;
            label = int.Parse(values[0]);
            for (int i = 0; i < 784; i++)
            {
                img[index] = values[i];
                index++;
            }
        }

        return img;
    }


    public static (int, double[]) getImage(int id)
    {
        int label = 0;
        double[] img = new double[784];
        using (var reader = new StreamReader(CsvFilePath))
        {
            // Read the first line of the file
            var line = reader.ReadLine();
            line = reader.ReadLine();
            for (int i = 0; i < id; i++)
            {
                line = reader.ReadLine();
            }

            var values = line.Split(',');
            //skip label + get image id

            int index = 0;
            label = int.Parse(values[0]);
            for (int i = 0; i < 784; i++)
            {
                img[index] = double.Parse(values[i]) / 255;
                index++;
            }
        }

        return (label, img);
    }
}