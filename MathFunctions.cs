using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex32;

namespace numberRecogniser;

public class MathFunctions
{
    const double LEARNING_RATE = 0.01;

    public const double LIM0 = 0.0000001;

    public static Func<double, double> ReLU = x => Math.Max(0, x);

    public static Func<double, double> ReLUtag = x => x > 0 ? 1 : 0;


    public static Vector<double> softMax(Vector<double> output)
    {
        double maxInput = output.Max();
        Vector<double> normalizedInput = output.Subtract(maxInput);
        Vector<double> expValues = normalizedInput.PointwiseExp();
        double expSum = expValues.Sum();
        Vector<double> softmaxOutput = expValues.Divide(expSum);

        return softmaxOutput;
    }

    public static double derivative(double costBefore, double costAfter)
    {
        return (costAfter - costBefore) / (LIM0);
    }

    public static Matrix<double> newWeight(Neuron[] currLayer, Vector<double> desireVector)
    {
        Matrix<double> sharedWeightMatrix = Matrix<double>.Build.Dense(currLayer[0].Weights.Count, currLayer.Length);
        for (int l = 0; l < currLayer.Length; l++)
        {
            Neuron curr = currLayer[l];
            Vector<double> newWeights = Vector<double>.Build.Dense(curr.Weights.Count);
            newWeights = Vector<double>.Build.Dense(curr.Weights.Count);
            for (int i = 0; i < curr.Weights.Count; i++)
            {
                double costBefore = 0;
                double costAfter = 0;
                for (int j = 0; j < curr.Weights.Count; j++)
                {
                    costBefore += Math.Pow(curr.Weights[i] - curr.Inputs[i], 2);
                    costAfter += Math.Pow((curr.Weights[i] + MathFunctions.LIM0) - curr.Inputs[i], 2);
                }

                costBefore /= curr.Weights.Count;
                costAfter /= curr.Weights.Count;
                double weightDer = (costAfter - costBefore) / MathFunctions.LIM0;
                newWeights[i] = MathFunctions.applyDirectionToLearningStep(weightDer);
            }

            //Console.WriteLine("made new");
            sharedWeightMatrix = sharedWeightMatrix.InsertColumn(l, newWeights);
        }


        return sharedWeightMatrix;
    }


    public static Vector<double> newBias(Neuron[] currLayer, Vector<double> desireVector)
    {
        Vector<double> newBiases = Vector<double>.Build.Dense(currLayer.Length);
        for (int l = 0; l < currLayer.Length; l++)
        {
            Neuron curr = currLayer[l];
            double costBefore = 0;
            double costAfter = 0;

            for (int i = 0; i < curr.Inputs.Count; i++)
            {
                costBefore += Math.Pow(curr.Bias - curr.Inputs[i], 2);
                costAfter += Math.Pow(curr.Bias + MathFunctions.LIM0 - curr.Inputs[i], 2);
            }

            costBefore /= curr.Inputs.Count;
            costAfter /= curr.Inputs.Count;

            double biasDer = (costAfter - costBefore) / MathFunctions.LIM0;
            newBiases[l] = MathFunctions.applyDirectionToLearningStep(biasDer);
        }

        return newBiases;
    }


    public static double softMax(DepNeuron depNeuron)
    {
        return Math.Exp(depNeuron.Calculate()) / Math.Exp(depNeuron.Calculate());
    }

    public static double getCost(Vector<double> output, Vector<double> desireVector)
    {
        Vector<double> costVector = (output - desireVector).PointwisePower(2);
        return costVector.Sum();
    }

    public static double getCostAlt(Vector<double> weight, Vector<double> input)
    {
        double cost = 0;
        for (int i = 0; i < input.Count; i++)
        {
            cost += Math.Pow(weight[i] - input[i], 2);
        }

        return cost / input.Count;
    }

    public static Vector<double> getDesireVector(int label)
    {
        Vector<double> desireVector = Vector<double>.Build.Dense(10);
        desireVector[label] = 1;
        return desireVector;
    }


    public static Double applyDirectionToLearningStep(double direction)
    {
        return direction * LEARNING_RATE;
    }
}