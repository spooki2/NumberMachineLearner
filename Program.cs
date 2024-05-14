using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using numberRecogniser;
using DenseVector = MathNet.Numerics.LinearAlgebra.Double.DenseVector;
using Vector = MathNet.Numerics.LinearAlgebra.Complex32.Vector;
using System.Text;

Console.OutputEncoding = Encoding.UTF8;
//layer init
int smarts = 100;
Layer layer1 = new Layer(smarts, 28 * 28);
Layer layer2 = new Layer(smarts, smarts);
Layer layer3 = new Layer(10, smarts);
Layer[] _ = { layer1, layer2,layer3 };
Network network = new Network(_);

//random image
Random random = new Random();
int imageIndex = random.Next(0, 60000);
int label = 0;


Vector<double> inputs = Vector<double>.Build.Dense(0);
(label, inputs) = DataFeeder.getImage(imageIndex);
layer1.setInputs(inputs);


network.initRandom();
int count = 0;
double right = 0;

Layer layer1A = new Layer(network.layerArray[0].Length, network.layerArray[0][0].Inputs.Count);
Layer layer2A = new Layer(network.layerArray[1].Length, network.layerArray[1][0].Inputs.Count);
Layer layer3A = new Layer(network.layerArray[2].Length, network.layerArray[2][0].Inputs.Count);
Layer[] _A = { layer1A,layer2A, layer3A };
Network holderNetwork = new Network(_A);

/*
imageIndex = random.Next(0, 60000);
imageIndex = 3;
label = 0;
inputs = Vector<double>.Build.Dense(0);
*/

while (true)
{
    //imageIndex = random.Next(0, 5000);
    imageIndex = count;


    /*
    if (count % 2 == 0)
    {
        imageIndex = 1;
    }
    else
    {
        imageIndex = 0;
    }


*/



    (label, inputs) = DataFeeder.getImage(imageIndex);
    layer1.setInputs(inputs);
    network.Fprop();

    Vector<double> outputVector = network.layerArray[network.layerArray.Length - 1].calculateLayer();
    string indicator = "🟥";
    if (label == outputVector.MaximumIndex())
    {
        indicator = "🟩";
        right++;
    }


    Console.WriteLine("[{3}] Cost: {0}|label: {1}|guess {2} |{4}",
        network.getCost(MathFunctions.getDesireVector(label)).ToString("F55"), label,
        outputVector.MaximumIndex(), count, indicator);


    //Console.WriteLine(network.getCost(MathFunctions.getDesireVector(label)).ToString("F80"));


    holderNetwork.applyHolderNetwork(network.bprop(MathFunctions.getDesireVector(label)));
    if (count % 100 == 0)
    {
        network.applyHolderNetwork(holderNetwork);
        holderNetwork.clearNetwork();
    }

    if (count % 500 == 0)
    {
        Console.WriteLine("");
        Console.WriteLine("==> ACCURACY: {1}%",count,(right/5.0).ToString("F1"));
        Console.WriteLine("");
        right = 0;
    }


    if (count == 5000)
    {
        network.LEARNING_RATE /= 5;
    }
    /*

    for (int layerIndex = 0; layerIndex < network.layerArray.Length; layerIndex++)
    {
        Layer currLayer = network.layerArray[layerIndex];

        // Iterate through each neuron in the current layer
        for (int neuronIndex = 0; neuronIndex < currLayer.Length; neuronIndex++)
        {
            Neuron currNeuron = currLayer[neuronIndex];
            double[] weights = new double[currNeuron.Weights.Count];

            // Extract weights from the current neuron
            for (int i = 0; i < currNeuron.Weights.Count; i++)
            {
                weights[i] = currNeuron.Weights[i];
            }

            // Save weights as an image
            string imageName = $"output_layer{layerIndex}_neuron{neuronIndex}.png";
            Model.DrawImageFromDoubles(weights, imageName);
        }
    }

    */
    count++;
}


//neuron data:
//[inputs weight | <neuron> | output]
//output layer then serves as input for next neuron

//first layer = 784 long array of doubles, that represent image output (pixel data)
//second layer = 16 long array of neurons, each has 784 inputs & weights. 1 bias and 1 output
//third layer = 16 long array of neurons, each has 16 inputs & weights. 1 bias and 1 output
//fourth layer = 10 long array of neurons, each has 16 inputs & weights. 1 bias and 1 output