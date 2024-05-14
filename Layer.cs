using MathNet.Numerics.LinearAlgebra;

namespace numberRecogniser;

public class Layer
{
    public Neuron[] neuronArray { get; set; }

    public Neuron this[int index]
    {
        get { return neuronArray[index]; }
        set { neuronArray[index] = value; }
    }

    public int Length
    {
        get { return neuronArray.Length; }
    }

    public Layer(Neuron[] neurons)
    {
        neuronArray = new Neuron[neurons.Length];
        setLayer(neurons);
    }

    //create layer
    public Layer(int neuronCount, int inputSize)
    {
        neuronArray = new Neuron[neuronCount];
        for (int i = 0; i < neuronCount; i++)
        {
            neuronArray[i] = new Neuron(inputSize);
        }
    }

    public void setLayer(Neuron[] neurons)
    {
        for (int i = 0; i < neurons.Length; i++)
        {
            this.neuronArray[i] = neurons[i];
        }
    }


    public Neuron[] getLayer()
    {
        return neuronArray;
    }

    public void setInputs(Vector<double> Inputs)
    {
        for (int i = 0; i < neuronArray.Length; i++)
        {
            Vector<double> inputsCopy = Vector<double>.Build.DenseOfVector(Inputs);
            neuronArray[i].Inputs = inputsCopy;
        }
    }

    public Vector<double> calculateLayer()
    {
        double[] output = new double[neuronArray.Length];

        for (int i = 0; i < neuronArray.Length; i++)
        {
            output[i] = neuronArray[i].calculate();
        }

        Vector<double> outputVector = Vector<double>.Build.DenseOfArray(output);
        return outputVector;
    }

    public void initLayer(int size)
    {
        for (int i = 0; i < neuronArray.Length; i++)
        {
            neuronArray[i] = new Neuron(size);
        }
    }
}