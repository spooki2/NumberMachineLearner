namespace numberRecogniser;

using MathNet.Numerics.LinearAlgebra;
using System;
using System.Drawing;

public static class Model
{
    public static void initRandom(Neuron[] hiddenLayer1, Neuron[] hiddenLayer2, Neuron[] outputLayer)
    {
        Random _random = new Random();
        for (int i = 0; i < 16; i++)
        {
            hiddenLayer1[i] = new Neuron
            {
                Weights = GenerateRandomVector(784, _random, -0.5, 0.5),
                Inputs = GenerateRandomVector(784, _random, -0.5, 0.5),
                Bias = 0
            };
        }

        for (int i = 0; i < 16; i++)
        {
            hiddenLayer2[i] = new Neuron
            {
                Weights = GenerateRandomVector(16, _random, -0.5, 0.5),
                Inputs = GenerateRandomVector(16, _random, -0.5, 0.5),
                Bias = 0
            };
        }

        for (int i = 0; i < 10; i++)
        {
            outputLayer[i] = new Neuron
            {
                Weights = GenerateRandomVector(16, _random, -0.5, 0.5),
                Inputs = GenerateRandomVector(16, _random, -0.5, 0.5),
                Bias = 0
            };
        }
    }

    private static Vector<double> GenerateRandomVector(int size, Random random, double minValue, double maxValue)
    {
        var vector = Vector<double>.Build.Dense(size);
        for (int i = 0; i < size; i++)
        {
            vector[i] = random.NextDouble() * (maxValue - minValue) + minValue;
        }

        return vector;
    }


    public static double[] labelToArr(double label)
    {
        double[] arr = new double[10];
        double index = label - 1;
        for (int i = 0; i < 10; i++)
        {
            if (i == index)
            {
                arr[i] = 1;
            }
            else
            {
                arr[i] = 0;
            }
        }

        return arr;
    }

    public static void randomWeights(DepNeuron[][] network)
    {
        Random random = new Random();
        for (int i = 0; i < network.Length; i++) // For each layer
        {
            for (int j = 0; j < network[i].Length; j++) // For each neuron
            {
                for (int k = 0; k < network[i][j].Weights.Length; k++)
                {
                    network[i][j].Weights[k] = random.NextDouble();
                }
            }
        }
    }


    public static void NeuronConnector(DepNeuron[][] network)
    {
        // Make each neuron's inputs be the outputs of neurons in the previous layer
        for (int i = 0; i < network.Length - 1; i++) // Iterate over each layer
        {
            DepNeuron[] nextLayer = network[i + 1];
            DepNeuron[] currLayer = network[i];
            for (int n = 0; n < nextLayer.Length; n++) // Iterate over each neuron in the next layer
            {
                for (int j = 0; j < currLayer.Length; j++) // Iterate over each neuron in the current layer
                {
                    nextLayer[n].Inputs[j] = currLayer[j].Calculate();
                }
            }
        }
    }


    public static void DrawImageFromDoubles(double[] pixels, String name)
    {
        Bitmap image = new Bitmap(28, 28);

        for (int i = 0; i < pixels.Length; i++)
        {
            int x = i % 28;
            int y = i / 28;

            double pixelValue = pixels[i];
            int colorValue = (int)(pixelValue * 255); // Map pixel value from range [0, 1] to [0, 255]
            //clamp color
            if (colorValue > 255)
            {
                colorValue = 255;
            }
            else if (colorValue < 0)
            {
                colorValue = 0;
            }

            Color color = Color.FromArgb(colorValue, colorValue, colorValue);
            image.SetPixel(x, y, color);
        }

        image.Save(name, System.Drawing.Imaging.ImageFormat.Png);
        //throw excep
    }
}