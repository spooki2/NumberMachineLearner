namespace numberRecogniser;

using MathNet.Numerics.LinearAlgebra;

public class Neuron
{
    public double Bias { get; set; }
    public Vector<double> Weights { get; set; }
    public Vector<double> Inputs { get; set; }

    public double calculate() //returns output
    {
        Func<double, double> activationFunc = MathFunctions.ReLU;
        return activationFunc(calculateNOACT());
    }

    public double calculateNOACT()
    {
        double sum = 0;
        for (int i = 0; i < Weights.Count; i++)
        {
            sum += Weights[i] * Inputs[i];
        }
        sum += Bias;
        return sum;
    }

    public Neuron(int size)
    {
        Bias = 0;
        Weights = Vector<double>.Build.Dense(size);
        Inputs = Vector<double>.Build.Dense(size);
    }

}