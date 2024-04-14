namespace numberRecogniser;

public struct Neuron
{
    public double Bias { get; set; }
    public double[] Weights { get; set; }
    public double[] Inputs { get; set; }

    public double Calculate() //returns output
    {
        Func<double, double> activationFunc = MathFunctions.ReLU;
        return activationFunc(preActivCalc());
    }

    public double preActivCalc()
    {
        double sum = 0;
        for (int i = 0; i < Weights.Length; i++)
        {
            sum += Weights[i] * Inputs[i];
        }

        return (sum+Bias);
    }

    public Neuron(double Bias, double[] Weights, double[] Inputs)
    {
        this.Bias = Bias;
        this.Weights = Weights;
        this.Inputs = Inputs;
    }
    //init empty neuron
    public Neuron()
    {
        this.Bias = 0;
        this.Weights = new double[0];
        this.Inputs = new double[0];
    }

    public Neuron(int inputCount)
    {
        this.Bias = 0;
        this.Weights = new double[inputCount];
        this.Inputs = new double[inputCount];
    }

    public void Connect(Neuron n2)
    {
        double[] newInputs = new double[n2.Inputs.Length + 1];
        n2.Inputs.CopyTo(newInputs, 0);
        newInputs[n2.Inputs.Length] = this.Calculate();
        n2.Inputs = newInputs;
    }

    public bool Equals(Neuron other)
    {
        return Bias.Equals(other.Bias) && Weights.Equals(other.Weights) && Inputs.Equals(other.Inputs);
    }
    public static bool operator == (Neuron n1, Neuron n2)
    {
        return n1.Equals(n2);
    }
    public static bool operator != (Neuron n1, Neuron n2)
    {
        return !n1.Equals(n2);
    }
}