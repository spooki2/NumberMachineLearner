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

    public static void acumulateWeights(Neuron neuron, Vector newWeights)
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

    public static void accumulateBias(int[] neuronCords, Neuron[][] network, Vector newBiases)
    {
        Neuron neuron = network[neuronCords[0]][neuronCords[1]];
        Neuron[] lastLayer = network[neuronCords[0] - 1];
        for (int i = 0; i < lastLayer.Length; i++)
        {
            lastLayer[i].Bias -= newBiases[i];
        }
    }

    public static void run(Neuron[][] network, int batchSize)
    {
        int batch = batchSize;
        //create mockup network to save the values of each weight
        Neuron[] layer0M = new Neuron[784];
        Neuron[] layer1M = new Neuron[16];
        Neuron[] layer2M = new Neuron[16];
        Neuron[] layer3M = new Neuron[10];
        Neuron[][] networkM = new[] { layer0M, layer1M, layer2M, layer3M };


        while (true)
        {
            Vector weightAccumulators = Vector.Build.Dense(0);
            Vector biasAccumulators = Vector.Build.Dense(0);
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
                    if (batch == batchSize)
                    {
                        //Console.WriteLine("W: "+(weightAccumulators.Count-newWeights.Count));
                        //Console.WriteLine("B: "+(biasAccumulators.Count-newBiases.Count));
                    }

                    weightAccumulators = Vector.Build.Dense(newWeights.Count);
                    biasAccumulators = Vector.Build.Dense(newBiases.Count);
                    //add to accumulators
                    weightAccumulators += newWeights;
                    biasAccumulators += newBiases;

                    if (batch <= 0)
                    {
                        batch = batchSize;
                        double cost = MathFunctions.getCost(output, Vector.Build.DenseOfArray(Model.labelToArr(label)));
                        Console.WriteLine("Cost: {0}", cost);
                        //devide by batchSize
                        weightAccumulators.Map(x => x / batchSize);
                        biasAccumulators.Map(x => x / batchSize);
                        applyWeights(network[i][j], weightAccumulators.Map(MathFunctions.applyDirectionToLearningStep));
                        applyBias(new[] { i, j }, network,
                            biasAccumulators.Map(MathFunctions.applyDirectionToLearningStep));
                    }
                }
            }

            batch--;
        }
    }
}