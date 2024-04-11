using System.Numerics;
//switch over to cuda
namespace numberRecogniser;

public class MathFunctions
{
    const double LEARNING_RATE = 0.01;

    public static Func<double, double> ReLU = x => Math.Max(0, x);

    public static Func<double, double> ReLUtag = x => x > 0 ? 1 : 0;
    const double LIM0 = 0.000001;

    public static double[] Softmax(Neuron[] neurons)
    {
        var scores = neurons.Select(neuron => neuron.Calculate()).ToArray();
        double maxScore = scores.Max(); // For numerical stability
        var expScores = scores.Select(score => Math.Exp(score - maxScore)).ToArray();
        double sumOfExp = expScores.Sum();
        return expScores.Select(score => score / sumOfExp).ToArray();
    }


    //white = Calculate()
    //green = Calculate before Activator
    //red = cost
    //blue = weight
    //purple = bias

    //CHAIN RULE:

    public static (double[], double[]) howDoPeopleComeUpWithThisShit(Neuron neuron, double desire)
    {
        //DESIRE SHOULDNT BE A DOUBLE IT HOULD BE A VECTOR!
        Console.WriteLine("size: {0}",neuron.Inputs.Length);
        double[] costWeightLiM0tagVec = new double[neuron.Inputs.Length]; //adjusment for previous neuron weight
        double[] costBiasLiM0tagVec = new double[neuron.Inputs.Length]; //adjusment for previous neuron bias

        //bias = ReLUTAG(preActCalc)*2*(calculate-desire)
        for (int i = 0; i < neuron.Inputs.Length; i++)
        {
            double BNC = neuron.Inputs[i];
            double PAC = neuron.preActivCalc(); //pre activator calculate
            double d, c;
            (d, c) = (desire, neuron.Calculate());
            costWeightLiM0tagVec[i] = ((preActivatorCalcWeightLIM0tag(BNC) * calcPreActivatorCalcLIM0tag(PAC) *
                                        costCalcLIM0tag(d, c)));
            costBiasLiM0tagVec[i] = (ReLUtag(PAC) * 2 * (c - d));
        }

        return (costWeightLiM0tagVec, costBiasLiM0tagVec);
    }

    //=
    public static double preActivatorCalcWeightLIM0tag(double backNeuronCalculate)
    {
        return backNeuronCalculate;
    }

    //*
    public static double calcPreActivatorCalcLIM0tag(double preActivatorCalc)
    {
        return MathFunctions.ReLUtag(preActivatorCalc);
    }

    //*
    public static double costCalcLIM0tag(double desire, double calculate)
    {
        return 2 * (calculate - desire);
    }

    public static void backPropagation(ref Neuron[][] network, double label)
    {
        double[] desire = Model.labelToArr(label);
        //currently the function above works on 1 neuron and all its sons.
        //do this to the entire last layer and get the average vector map
        //update the weights using learning curve/gradient decrease (?)
        //now with the fixed weights do this to the previous layer, continue till start

        for (int i = network.Length - 1; i >= 0; i--)
        {
            //these get applied each layer
            int devider = 0;
            double[] avgWeightDerivative = new double[network[i].Length];
            double[] avgBiasDerivative = new double[network[i].Length];
            //output -> hidden -> input
            Neuron[] layer = network[i];
            for (int q = 0; q < layer.Length; q++)
            {
                double[] tempW = new double[layer[q].Inputs.Length];
                double[] tempB = new double[layer[q].Inputs.Length];
                (tempW, tempB) = howDoPeopleComeUpWithThisShit(layer[q], desire);
                for (int j = 0; j < layer[q].Inputs.Length; j++)
                {
                    avgWeightDerivative[j] += tempW[j];
                    avgBiasDerivative[j] += tempB[j];
                }

                devider++;
            }

            //avg out the layer
            for (int j = 0; j < avgWeightDerivative.Length; j++)
            {
                avgWeightDerivative[j] /= devider;
                avgBiasDerivative[j] /= devider;
            }

            //apply the changes
            for (int q = 0; q < layer.Length; q++)
            {
                for (int j = 0; j < layer[q].Weights.Length; j++)
                {
                    //take step towards using learning rate
                    layer[q].Weights[j] -= LEARNING_RATE * avgWeightDerivative[j];
                    layer[q].Bias -= LEARNING_RATE * avgBiasDerivative[j];
                }
            }
        }
    }
}