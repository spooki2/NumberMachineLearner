namespace numberRecogniser;

using System;
using System.Drawing;
using MathNet.Numerics.LinearAlgebra;
using Vector = MathNet.Numerics.LinearAlgebra.Vector<double>;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;

public class singularNeuronTrain
{
    public static void initRandom(DepNeuron depNeuron)
    {
        Random random = new Random();
        for (int i = 0; i < depNeuron.Weights.Length; i++)
        {
            depNeuron.Weights[i] = random.NextDouble();
        }
    }

    public static void runSingle(DepNeuron depNeuron, double[] img)
    {

        for (int i = 0; i < img.Length; i++)
        {
            depNeuron.Inputs[i] = img[i];
        }

        double cost = 0;
        for (int i = 0; i < img.Length; i++)
        {
            cost += Math.Pow(depNeuron.Weights[i] - img[i], 2);
        }

        cost /= img.Length;
        Console.WriteLine("cost: {0}", cost.ToString("F12"));


        Vector newWeights = Vector.Build.Dense(depNeuron.Weights.Length);
        double newBias = 0;
        for (int i = 0; i < depNeuron.Weights.Length; i++)
        {
            double costBefore = 0;
            double costAfter = 0;
            for (int j = 0; j < depNeuron.Weights.Length; j++)
            {
                costBefore += Math.Pow(depNeuron.Weights[i] - img[i], 2);
                costAfter += Math.Pow((depNeuron.Weights[i] + MathFunctions.LIM0) - img[i], 2);
            }

            costBefore /= depNeuron.Weights.Length;
            costAfter /= depNeuron.Weights.Length;
            double weightDer = (costAfter - costBefore) / MathFunctions.LIM0;
            newBias += MathFunctions.applyDirectionToLearningStep(weightDer);
            newWeights[i] = MathFunctions.applyDirectionToLearningStep(weightDer);
        }

        newBias /= depNeuron.Weights.Length;

        //apply weights
        for (int i = 0; i < depNeuron.Weights.Length; i++)
        {
            depNeuron.Weights[i] -= newWeights[i];
        }
        //apply bias
        depNeuron.Bias -= newBias;

        double[] weights = new double[depNeuron.Weights.Length];
        //draw this to png
        for (int i = 0; i < depNeuron.Weights.Length; i++)
        {
            weights[i] = depNeuron.Weights[i];
        }

        //Model.DrawImageFromDoubles(weights);
    }
}