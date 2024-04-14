namespace numberRecogniser;

using System;
using System.Drawing;
using MathNet.Numerics.LinearAlgebra;
using Vector = MathNet.Numerics.LinearAlgebra.Vector<double>;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;

public class singularNeuronTrain
{
    public static void initRandom(Neuron neuron)
    {
        Random random = new Random();
        for (int i = 0; i < neuron.Weights.Length; i++)
        {
            neuron.Weights[i] = random.NextDouble();
        }
    }

    public static void runSingle(Neuron neuron, double[] img)
    {

        for (int i = 0; i < img.Length; i++)
        {
            neuron.Inputs[i] = img[i];
        }

        double cost = 0;
        for (int i = 0; i < img.Length; i++)
        {
            cost += Math.Pow(neuron.Weights[i] - img[i], 2);
        }

        cost /= img.Length;
        Console.WriteLine("cost: {0}", cost.ToString("F12"));


        Vector newWeights = Vector.Build.Dense(neuron.Weights.Length);
        double newBias = 0;
        for (int i = 0; i < neuron.Weights.Length; i++)
        {
            double costBefore = 0;
            double costAfter = 0;
            for (int j = 0; j < neuron.Weights.Length; j++)
            {
                costBefore += Math.Pow(neuron.Weights[i] - img[i], 2);
                costAfter += Math.Pow((neuron.Weights[i] + MathFunctions.LIM0) - img[i], 2);
            }

            costBefore /= neuron.Weights.Length;
            costAfter /= neuron.Weights.Length;
            double weightDer = (costAfter - costBefore) / MathFunctions.LIM0;
            newBias += MathFunctions.applyDirectionToLearningStep(weightDer);
            newWeights[i] = MathFunctions.applyDirectionToLearningStep(weightDer);
        }

        newBias /= neuron.Weights.Length;

        //apply weights
        for (int i = 0; i < neuron.Weights.Length; i++)
        {
            neuron.Weights[i] -= newWeights[i];
        }
        //apply bias
        neuron.Bias -= newBias;

        double[] weights = new double[neuron.Weights.Length];
        //draw this to png
        for (int i = 0; i < neuron.Weights.Length; i++)
        {
            weights[i] = neuron.Weights[i];
        }

        Model.DrawImageFromDoubles(weights);
    }
}