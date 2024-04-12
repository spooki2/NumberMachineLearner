namespace numberRecogniser;

public static class Model
{
    public static void initRandom(Neuron[][] network)
    {
        Random random = new Random();
        for (int i = 0; i < network.Length; i++) // For each layer
        {
            for (int j = 0; j < network[i].Length; j++) // For each neuron
            {
                int numberOfInputs =
                    (i == 0)
                        ? 0
                        : network[i - 1]
                            .Length; // If it's the first layer, 0 inputs, otherwise number of neurons in previous layer
                network[i][j] = new Neuron(random.NextDouble(), new double[numberOfInputs], new double[numberOfInputs]);
            }
        }
    }

    public static double[] labelToArr(double label)
    {
        double[] arr = new double[10];
        double index = label - 1;
        for (int i = 0; i < 10; i++)
        {
            if (i == index)
            {
                arr[i] = 1;
            }
            else
            {
                arr[i] = 0;
            }
        }

        return arr;
    }

    public static int feedImage(Neuron[][] network, int id)
    {
        int label = 0;
        double[] img = new double[784];
        (label, img) = DataFeeder.getImage(id);
        for (int i = 0; i < img.Length; i++)
        {
            network[0][i].Inputs = new double[] { img[i] };
        }

        return label;
    }

    public static void NeuronConnector(Neuron[][] Layers)
    {
        for (int i = 0; i < Layers.Length - 1; i++) // For each layer except the last one
        {
            for (int j = 0; j < Layers[i].Length; j++) // For each neuron in the current layer
            {
                Neuron n1 = Layers[i][j];
                for (int k = 0; k < Layers[i + 1].Length; k++) // For each neuron in the next layer
                {
                    Neuron n2 = Layers[i + 1][k];
                    double output = n1.Calculate(); // Calculate output once to avoid repeated calculations
                    int newLength = n2.Inputs.Length + 1;
                    double[] newInputs = new double[newLength];
                    Array.Copy(n2.Inputs, newInputs, n2.Inputs.Length);
                    newInputs[newLength - 1] = output;
                    n2.Inputs = newInputs; // Update inputs
                }
            }
        }
    }
}