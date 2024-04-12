using MathNet.Numerics.LinearAlgebra;
using Vector = MathNet.Numerics.LinearAlgebra.Vector<double>;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;

namespace numberRecogniser;

public class NetworkTrainer
{
    public static Random random = new Random();

    public static void applyWeights(Neuron neuron, Vector newWeights)
    {
        for (int i = 0; i < neuron.Weights.Length; i++)
        {
            neuron.Weights[i] -= newWeights[i];
        }
    }

    public static void applyBias(int[] neuronCords, Neuron[][] network, Vector newBiases)
    {
        Neuron neuron = network[neuronCords[0]][neuronCords[1]];
        Neuron[] lastLayer = network[neuronCords[0] - 1];
        for (int i = 0; i < lastLayer.Length; i++)
        {
            lastLayer[i].Bias -= newBiases[i];
        }
    }

    public static void run(Neuron[][] network)
    {
        int imageID = random.Next(0, 60000);
        int label = Model.feedImage(network, imageID);
        Double[] output = MathFunctions.Softmax(network[network.Length - 1]);

        //this does everything BUT feed good batches (the batches rn are average of 1)
        for (int i = network.Length - 1; i > 0; i--)
        {
            for (int j = 0; j < network[i].Length; j++)
            {
                Vector newWeights =
                    MathFunctions.getNewWeights(output, network[i][j],
                        Vector.Build.DenseOfArray(Model.labelToArr(label)));
                Vector newBiases =
                    MathFunctions.getNewBiases(output, network[i][j],
                        Vector.Build.DenseOfArray(Model.labelToArr(label)));
                applyWeights(network[i][j], newWeights);
                applyBias(new[] { i, j }, network, newBiases);
            }
        }

        Console.WriteLine("Cost: {0}",MathFunctions.getCost(output,Vector.Build.DenseOfArray(Model.labelToArr(label))));
    }
}