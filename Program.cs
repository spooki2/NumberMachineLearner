using MathNet.Numerics.LinearAlgebra;
using numberRecogniser;
using NumberRecogniser;

Random random = new Random();

Neuron[] layer0 = new Neuron[784];
Neuron[] layer1 = new Neuron[16];
Neuron[] layer2 = new Neuron[16];
Neuron[] layer3 = new Neuron[10];
Neuron[][] network = new[] { layer0, layer1, layer2, layer3 };
Model.initRandom(network);
Model.NeuronConnector(network);


//img stuff
//DataFeeder.getImage(imageID);
//(int _, Double[] img) = DataFeeder.getImage(imageID);
//CSV2JPEG.Convert(img);
while (true)
{
    NetworkTrainer.run(network);
}
/*
Console.WriteLine("\n=== INPUT LAYER ===\n");
for (int i = 0; i < 5; i++)
{
    Console.WriteLine("{0}", layer0[i].Calculate());
}

Console.WriteLine("\n=== HIDDEN LAYER A ===\n");
for (int i = 0; i < 5; i++)
{
    Console.WriteLine("{0}", layer1[i].Calculate());
}

Console.WriteLine("\n=== HIDDEN LAYER B ===\n");
for (int i = 0; i < 5; i++)
{
    Console.WriteLine("{0}", layer2[i].Calculate());
}

Console.WriteLine("\n=== OUTPUT LAYER ===\n");
for (int i = 0; i < 5; i++)
{
    Console.WriteLine("{0}", output[i]);
}

//index of highest float in output
//int bestGuess = Array.IndexOf(output, output.Max());
//Console.WriteLine("\nLABEL: {0}\nRESULT: {1}", label, bestGuess);
*/