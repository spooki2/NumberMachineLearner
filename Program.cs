using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using numberRecogniser;
using NumberRecogniser;
using DenseVector = MathNet.Numerics.LinearAlgebra.Double.DenseVector;
using Vector = MathNet.Numerics.LinearAlgebra.Complex32.Vector;

Random random = new Random();
Vector<double> inputs = Vector<double>.Build.Dense(784);
Neuron[] hiddenLayer1 = new Neuron[16];
Neuron[] hiddenLayer2 = new Neuron[16];
Neuron[] outputLayer = new Neuron[10];
Neuron[][] network = { hiddenLayer1, hiddenLayer2, outputLayer };


Model.initRandom(hiddenLayer1, hiddenLayer2, outputLayer);

//fill inputs with image
int imageIndex = random.Next(0, 60000);
int label;
double[] image;
(label, image) = DataFeeder.getImage(imageIndex);
for (int i = 0; i < inputs.Count; i++)
{
    inputs[i] = image[i];
}


(label, image) = DataFeeder.getImage(imageIndex);


while (true)
{
    //imageIndex = random.Next(0, 60000);
    imageIndex = 1;
    (label, image) = DataFeeder.getImage(imageIndex);
    for (int i = 0; i < image.Length; i++)
    {
        inputs[i] = image[i];
    }

    //COPY PASTE THIS
    Propagate.forward(inputs, hiddenLayer1);
    networkTrainer.run(inputs, hiddenLayer1, label);

    double cost = 0;
    for (int i = 0; i < hiddenLayer1.Length; i++)
    {
        cost = MathFunctions.getCost(hiddenLayer1[i].Weights, hiddenLayer1[i].Inputs);
    }

    cost /= hiddenLayer1.Length;

    // Console.WriteLine("Cost - 1: {0}", cost.ToString("F12"));

    //COPY PASTE THIS

    Vector<double> hl1Calc = Vector<double>.Build.Dense(hiddenLayer1.Length);

    for (int i = 0; i < hl1Calc.Count; i++)
    {
        hl1Calc[i] = hiddenLayer1[i].Calculate();
    }

    networkTrainer.run(hl1Calc, hiddenLayer2, label);
    Console.WriteLine("TEST AREA");
    double cost2 = 0;
    for (int i = 0; i < hiddenLayer2.Length; i++)
    {
        cost2 = MathFunctions.getCost(hiddenLayer2[i].Weights, hiddenLayer2[i].Inputs);
    }

    cost2 /= hiddenLayer1.Length;

    // Console.WriteLine("Cost - 2: {0}", cost2.ToString("F12"));


    //COST 3

    Vector<double> hl2Calc = Vector<double>.Build.Dense(hiddenLayer2.Length);

    for (int i = 0; i < hl2Calc.Count; i++)
    {
        hl2Calc[i] = hiddenLayer2[i].Calculate();
    }

    networkTrainer.run(hl2Calc, outputLayer, label);

    double cost3 = 0;
    for (int i = 0; i < outputLayer.Length; i++)
    {
        cost3 = MathFunctions.getCost(outputLayer[i].Weights, outputLayer[i].Inputs);
    }

    cost3 /= hiddenLayer2.Length;

    //Console.WriteLine("Cost - 3: {0}", cost3.ToString("F12"));

    //
    Vector<double> outputCalcs = Vector<double>.Build.Dense(outputLayer.Length);

    for (int i = 0; i < outputLayer.Length; i++)
    {
        outputCalcs[i] = outputLayer[i].Calculate();
    }

    double costf = MathFunctions.getCost(outputCalcs, MathFunctions.getDesireVector(label));

    Console.WriteLine("=======");
    for (int i = 0; i < outputLayer.Length; i++)
    {
        Console.WriteLine("{0}: {1}",i,outputLayer[i].Calculate());
    }
    Console.WriteLine("=======");
    Console.WriteLine("Cost final: {0}", costf.ToString("F12"));

// Iterate through each layer
    for (int layerIndex = 0; layerIndex < network.Length; layerIndex++)
    {
        Neuron[] currLayer = network[layerIndex];

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
}


//neuron data:
//[inputs weight | <neuron> | output]
//output layer then serves as input for next neuron

//first layer = 784 long array of doubles, that represent image output (pixel data)
//second layer = 16 long array of neurons, each has 784 inputs & weights. 1 bias and 1 output
//third layer = 16 long array of neurons, each has 16 inputs & weights. 1 bias and 1 output
//fourth layer = 10 long array of neurons, each has 16 inputs & weights. 1 bias and 1 output