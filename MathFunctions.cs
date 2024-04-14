using System.Numerics;
using MathNet.Numerics;
using Vector = MathNet.Numerics.LinearAlgebra.Vector<double>;
using Matrix = MathNet.Numerics.LinearAlgebra.Matrix<double>;

namespace numberRecogniser;

public class MathFunctions
{
    const double LEARNING_RATE = 0.01;
    //public const double LIM0 = Double.MinValue;
    public const double LIM0 = 0.000001;

    public static Func<double, double> ReLU = x => Math.Max(0, x);

    public static Func<double, double> ReLUtag = x => x > 0 ? 1 : 0;


    public static double[] softMax(Neuron[] neurons)
    {
        var scores = neurons.Select(neuron => neuron.Calculate()).ToArray();
        double maxScore = scores.Max(); // For numerical stability
        var expScores = scores.Select(score => Math.Exp(score - maxScore)).ToArray();
        double sumOfExp = expScores.Sum();
        return expScores.Select(score => score / sumOfExp).ToArray();
    }

    public static double softMax(Neuron neuron)
    {
        return Math.Exp(neuron.Calculate()) / Math.Exp(neuron.Calculate());
    }

    public static double getCost(Double[] output, Vector desire)
    {
        double cost = 0;
        for (int i = 0; i < output.Length; i++)
        {
            cost += Math.Pow(output[i] - desire[i], 2);
        }

        return cost;
    }

    //white = Calculate()
    //green = Calculate before Activator
    //red = cost
    //blue = weight
    //purple = bias


    public static Vector getNewBiases(Double[] output,Neuron neuron, Vector desireVector)
    {
        Vector biasSteps = Vector.Build.Dense(neuron.Inputs.Length);
        for (int i = 0; i < neuron.Inputs.Length; i++)
        {
            biasSteps[i] =
                biasDerivative(neuron.preActivCalc(),(getCost(output, desireVector)));
        }
        //turn to step
        biasSteps = biasSteps.Map(applyDirectionToLearningStep);
        return biasSteps;
    }

    public static Vector getNewWeights(Double[] output,Neuron neuron, Vector desireVector)
    {
        Vector weightSteps = Vector.Build.Dense(neuron.Weights.Length);
        for (int i = 0; i < neuron.Weights.Length; i++)
        {
            weightSteps[i] =
                weightDerivative(neuron.Inputs[i], neuron.preActivCalc(), getCost(output, desireVector));
        }
        //turn to step
        weightSteps = weightSteps.Map(applyDirectionToLearningStep);
        return weightSteps;
    }


    public static (Vector, Vector) neuronSuggestedDirection(Double[] output,Neuron neuron, double desire) //neuron and its desire
    {
        //put in neuron - get back vector of desired weight and bias changes

        Vector desireVector = Vector.Build.DenseOfArray(Model.labelToArr(desire));
        Vector weightStepVector = Vector.Build.Dense(neuron.Inputs.Length);
        Vector biasStepVector = Vector.Build.Dense(neuron.Inputs.Length);

        for (int i = 0; i < neuron.Inputs.Length; i++)
        {
            double biasDer = biasDerivative(neuron.preActivCalc(), getCost(output, desireVector));
            double weigtDer = biasDer * neuron.Inputs[i];
            weightStepVector[i] += biasDer;
            biasStepVector[i] += weigtDer;
        }

        return (weightStepVector, biasStepVector);
    }

    public static double biasDerivative(double preActCalc, double calcSubDes)
    {
        //biasDerivative = ReLUTAG(preActCalc)*(calculate-desire)
        return ReLUtag(preActCalc) * (calcSubDes);
    }

    public static double weightDerivative(double input, double preActCalc, double calcSubDes)
    {
        //weightDerivative = Input * ReLUtag(preActCalc)*(calc-desire)
        return input * biasDerivative(preActCalc, calcSubDes);
    }

    public static double Sigmoid(double value) {
        double k = Math.Exp(value);
        return k / (0.1f + k);
    }


    public static Double applyDirectionToLearningStep(double direction)
    {
        return direction * LEARNING_RATE;
    }
}
/*
   public static (double[], double[]) howDoPeopleComeUpWithThisShit(Neuron neuron, double desire)
   {
       //DESIRE SHOULDNT BE A DOUBLE IT HOULD BE A VECTOR!
       Console.WriteLine("size: {0}", neuron.Inputs.Length);
       double[] costWeightLiM0tagVec = new double[neuron.Inputs.Length]; //adjusment for previous neuron weight
       double[] costBiasLiM0tagVec = new double[neuron.Inputs.Length]; //adjusment for previous neuron bias

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
*/