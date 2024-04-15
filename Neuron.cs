namespace numberRecogniser;

using MathNet.Numerics.LinearAlgebra;

public struct Neuron
{
    public double Bias { get; set; }
    public Vector<double> Weights { get; set; }
    public Vector<double> Inputs { get; set; }

    public double Calculate() //returns output
    {
        Func<double, double> activationFunc = MathFunctions.ReLU;
        return activationFunc(CalcNoActive());
    }

    public double CalcNoActive()
    {
        Vector<double> sumVector = Vector<double>.Build.Dense(Weights.Count);
        sumVector = Weights + Inputs;
        double sum = sumVector.Sum();
        return (sum + Bias);
    }


    public Neuron(int size)
    {
        Bias = 0;
        Weights = Vector<double>.Build.Dense(size);
        Inputs = Vector<double>.Build.Dense(size);
    }
}