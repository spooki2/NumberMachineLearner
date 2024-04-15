using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex32;

namespace numberRecogniser;

public class Propagate
{
    public static Vector<double> forward(Vector<double> inputs, Neuron[] layer)
    {
        //inputs vector

        for (int i = 0; i < layer.Length; i++)
        {
            layer[i].Inputs = inputs;
        }

        //(calculations)
        Vector<double> output = Vector<double>.Build.Dense(layer.Length);
        for (int i = 0; i < layer.Length; i++)
        {
            output[i] = layer[i].Calculate();
        }

        return output;
    }


    public static (Matrix<double> newWeight, Vector<double> newBias) backward(Neuron[] layer,int label)
    {

        Matrix<double> newWeight = MathFunctions.newWeight(layer, MathFunctions.getDesireVector(label));
        Vector<double> newBias = Vector<double>.Build.Dense(layer.Length);
        return (newWeight, newBias);
    }

    public static void applyLayer(Vector<double> newWeights, Vector<double> newBias, Neuron[] layer)
    {
        for (int i = 0; i < layer.Length; i++)
        {
            layer[i].Bias -= newBias[i];
            for (int j = 0; j < layer[i].Weights.Count; j++)
            {
                layer[i].Weights[j] -= newWeights[j];
            }
        }
    }
}