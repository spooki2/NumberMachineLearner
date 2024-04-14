using MathNet.Numerics.LinearAlgebra;
using numberRecogniser;
using NumberRecogniser;

Random random = new Random();

Neuron[] layer0 = new Neuron[784];
Neuron[] layer1 = new Neuron[16];
Neuron[] layer2 = new Neuron[16];
Neuron[] layer3 = new Neuron[10];
Neuron[][] network = new[] {layer0, layer1, layer2, layer3 };




int t = 0;
int label = random.Next(0, 60000);
double[] img = new double[784];
(t, img) = DataFeeder.getImage(label);

Neuron neuron = new Neuron(784);


neuron.Bias = random.NextDouble();
singularNeuronTrain.initRandom(neuron);
while (true)
{
    singularNeuronTrain.runSingle(network[1][0], img);
}

//neuron data:
//[inputs weight | <neuron> | output]
//output layer then serves as input for next neuron

//first layer = 784 long array of doubles, that represent image output (pixel data)
//second layer = 16 long array of neurons, each has 784 inputs & weights. 1 bias and 1 output
//third layer = 16 long array of neurons, each has 16 inputs & weights. 1 bias and 1 output
//fourth layer = 10 long array of neurons, each has 16 inputs & weights. 1 bias and 1 output
