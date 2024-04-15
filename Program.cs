using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using numberRecogniser;
using NumberRecogniser;
using DenseVector = MathNet.Numerics.LinearAlgebra.Double.DenseVector;
using Vector = MathNet.Numerics.LinearAlgebra.Complex32.Vector;

Random random = new Random();
Vector<double> inputs = Vector<double>.Build.Dense(784);
Neuron[] hiddenLayer1 = new Neuron[2];
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


imageIndex = random.Next(0, 60000);
(label, image) = DataFeeder.getImage(imageIndex);

for (int i = 0; i < image.Length; i++)
{
    inputs[i] = image[i];
}

Propagate.forward(inputs, hiddenLayer1);
//singular layer train
int stohastic = 25;
Matrix<double> massWeight = Matrix<double>.Build.Dense(0, 0);
Vector<double> massBias = Vector<double>.Build.Dense(0);

while (true)
{
    Neuron[] currLayer = hiddenLayer1;
    Matrix<double> newWeight = Matrix<double>.Build.Dense(0, 0);
    Vector<double> newBias = Vector<double>.Build.Dense(0);
    (newWeight, newBias) = Propagate.backward(currLayer, label); //BRINGS BACK ONLY THE FIRST NEURON
    massWeight = Matrix<double>.Build.Dense(newWeight.RowCount,newWeight.ColumnCount);
    massBias = Vector<double>.Build.Dense(newBias.Count);
    massWeight += newWeight;
    massBias += newBias;
    if (stohastic == 0)
    {
        Propagate.applyLayer(massWeight.Column(0), newBias, currLayer); //should be row but flipped, works either way.
        Propagate.applyLayer(massWeight.Column(1), newBias, currLayer);
        stohastic = 25;
        Console.WriteLine("STOHASTIC\n");
    }

    //Propagate.applyLayer(newWeight.Column(0), newBias, currLayer);


    double cost = 0;
    cost = MathFunctions.getCostAlt(currLayer[0].Weights, currLayer[0].Inputs);
    Console.WriteLine("cost: {0}", cost.ToString("F12"));

    double[] weights = new double[784];
    for (int i = 0; i < currLayer[0].Weights.Count; i++)
    {
        weights[i] = (currLayer[0].Weights[i]);
    }

    double[] weights2 = new double[784];
    for (int i = 0; i < currLayer[1].Weights.Count; i++)
    {
        weights2[i] = (currLayer[1].Weights[i]);
    }

    Model.DrawImageFromDoubles(weights, "output.png");
    Model.DrawImageFromDoubles(weights2, "output2.png");
    stohastic--;
}


while (true)
{
    //FRONT PROPAGATION
    Vector<double> output0 = Propagate.forward(inputs, network[0]);
    Vector<double> output1 = Propagate.forward(output0, network[1]);
    Vector<double> finalOutput = Propagate.forward(output1, network[2]);
    //FRONT PROPAGATION

    finalOutput = MathFunctions.softMax(finalOutput); // should be above or below?
    double cost = 0;
    cost = MathFunctions.getCost(finalOutput, MathFunctions.getDesireVector(label));
    Console.WriteLine("cost: {0}", cost.ToString("F12"));
}


//neuron data:
//[inputs weight | <neuron> | output]
//output layer then serves as input for next neuron

//first layer = 784 long array of doubles, that represent image output (pixel data)
//second layer = 16 long array of neurons, each has 784 inputs & weights. 1 bias and 1 output
//third layer = 16 long array of neurons, each has 16 inputs & weights. 1 bias and 1 output
//fourth layer = 10 long array of neurons, each has 16 inputs & weights. 1 bias and 1 output