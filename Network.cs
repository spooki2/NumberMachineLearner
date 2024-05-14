using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using Vector = MathNet.Numerics.LinearAlgebra.Complex.Vector;

namespace numberRecogniser;

public class Network
{
    public double LEARNING_RATE = 0.01;
    //public double LEARNING_RATE = 0.001;

    public Layer[] layerArray = new Layer[3]; //DOESNT STACK

    public Network(Layer[] layers)
    {
        setNetwork(layers);
    }

    public double LearningRate
    {
        get => LEARNING_RATE;
        set => LEARNING_RATE = value;
    }

    // Property for getting the length of the array

    public void clearNetwork()
    {
        for (int i = 0; i < layerArray.Length; i++)
        {
            for (int j = 0; j < layerArray[i].Length; j++)
            {
                for (int k = 0; k < layerArray[i][j].Weights.Count; k++)
                {
                    layerArray[i][j].Weights[k] = 0;
                }
            }
        }
    }

    public void initRandom()
    {
        int inputSize = layerArray[0][0].Inputs.Count;
        Random random = new Random();
        for (int i = 0; i < layerArray.Length; i++)
        {
            for (int j = 0; j < layerArray[i].Length; j++)
            {
                layerArray[i][j].Bias = 0.0;
                for (int k = 0; k < layerArray[i][j].Weights.Count; k++)
                {
                    {
                        layerArray[i][j].Weights[k] = (random.NextDouble() - 0.5);
                    }
                }
            }
        }
    }


    public void setNetwork(Layer[] layers)
    {
        for (int i = 0; i < layers.Length; i++)
        {
            this.layerArray[i] = layers[i];
        }
    }


    public void Fprop()
    {
        for (int i = 1; i < layerArray.Length; i++)
        {
            //each layer
            for (int j = 0; j < layerArray[i].Length; j++)
            {
                //each neuron in layer
                for (int k = 0; k < layerArray[i][j].Inputs.Count; k++)
                {
                    //for each input in each neuron
                    layerArray[i][j].Inputs[k] = layerArray[i - 1][k].calculate();
                }
            }
        }
    }

    public Network bprop(Vector<double> desireVector)
    {
        //desireVector = MathFunctions.getDesireVector(0);
        //desireVector += MathFunctions.getDesireVector(5);

        //DOESNT SCALE
        Layer layer1 = new Layer(layerArray[0].Length, layerArray[0][0].Inputs.Count);
        Layer layer2 = new Layer(layerArray[1].Length, layerArray[1][0].Inputs.Count);
        Layer layer3 = new Layer(layerArray[2].Length, layerArray[2][0].Inputs.Count);
        Layer[] _ = { layer1,layer2, layer3 };
        Network holderNetwork = new Network(_);
        //DOESNT SCALE
        /*
          (C0/a(L)) = 2(A(L)-y)
          (a(L)/z(L)) = s'(z(L))
          (z(L)/w(L)) = a(L-1)
       */
        Layer outputLayer = layerArray[layerArray.Length - 1];
        Layer outputLayerHolder = holderNetwork.layerArray[layerArray.Length - 1];
        //for each weight you work on its corrosponding input (a-1)


        for (int i = 0; i < outputLayer.Length; i++)
        {
            for (int j = 0; j < outputLayer[i].Inputs.Count; j++)
            {
                double val = 1;
                val *= 2 * (outputLayer[i].calculate() - desireVector[i]);
                val *= MathFunctions.ReLUtag(outputLayer[i].calculateNOACT());
                val *= outputLayer[i].Inputs[j];
                //outputLayer[i].Weights[j] -= val * LEARNING_RATE;
                outputLayerHolder[i].Weights[j] -= val * LEARNING_RATE;
            }
        }

        //#---------------------------- SECOND LAYER ----------------------------#
        /*
          (C0/a(L)) = 2(A(L)-y)
          (a(L)/z(L)) = s'(z(L))
          (z(L)/a(L-1)) = w(L)
          (a(L-1)/Z(L-1)) = s'(z(L-1))
          (Z(L-1)/W(L-1)) = A(L-2)
       */
        Layer outputLayerBefore = layerArray[layerArray.Length - 2];
        Layer outputLayerBeforeHolder = holderNetwork.layerArray[layerArray.Length - 2];

        for (int i = 0; i < outputLayer.Length; i++)
        {
            for (int j = 0; j < outputLayerBefore.Length; j++)
            {
                for (int k = 0; k < outputLayerBefore[j].Inputs.Count(); k++)
                {
                    double val = 1;
                    val *= 2 * (outputLayer[i].calculate() - desireVector[i]);
                    val *= MathFunctions.ReLUtag(outputLayer[i].calculateNOACT());
                    val *= outputLayer[i].Weights[j];
                    val *= MathFunctions.ReLUtag(outputLayerBefore[j].calculateNOACT());
                    val *= outputLayerBefore[j].Inputs[k];
                    //outputLayerBefore[j].Weights[k] -= val * LEARNING_RATE/10;
                    outputLayerBeforeHolder[j].Weights[k] -= val * LEARNING_RATE;
                }
            }
        }
        return holderNetwork;
    }

    public double getCost(Vector<double> desireVector)
    {
        Vector<double> outputVector = layerArray[layerArray.Length - 1].calculateLayer();
        //outputVector = MathFunctions.softMax(outputVector);
        Vector<double> costVector = (outputVector - desireVector).PointwisePower(2);
        return costVector.Sum();
    }

    public double tiltWeight(int i, int j, int k, int label)
    {
        return 0;
        /*
         i = layer
         j = neuron
         k = weight




         */
        /*
        w_eq: a<l-1>*ReLUtag(noActCalc)*2*(a<l>-y)
        get layer of w_eq, for the final.
        then go back and slowly fill in a<l-1> and switch it out to a<l-2>
        when ur at the last layer ???
        */
    }

    public double tiltBias(int i, int j, int label)
    {
        return MathFunctions.ReLUtag(layerArray[i][j].calculateNOACT()) * 2 *
               getCost(MathFunctions.getDesireVector(label));
    }

    public Network Bprop(int label, int stoha)
    {
        //DOESNT SCALE
        Layer layer1 = new Layer(layerArray[0].Length, layerArray[0][0].Inputs.Count);
        Layer layer2 = new Layer(layerArray[1].Length, layerArray[1][0].Inputs.Count);
        Layer layer3 = new Layer(layerArray[2].Length, layerArray[2][0].Inputs.Count);
        Layer[] _ = { layer1, layer2, layer3 };
        Network holderNetwork = new Network(_);
        //DOESNT SCALE

        for (int i = layerArray.Length - 1; i >= 0; i--)
        {
            //for each layer

            for (int j = layerArray[i].Length - 1; j >= 0; j--)
            {
                //for each neuron
                holderNetwork.layerArray[i][j].Bias -= (tiltBias(i, j, label) * LEARNING_RATE) / stoha;

                for (int k = layerArray[i][j].Weights.Count - 1; k >= 0; k--)
                {
                    //for each Weight
                    holderNetwork.layerArray[i][j].Weights[k] -= (tiltWeight(i, j, k, label) * LEARNING_RATE) / stoha;
                }
            }
        }

        return holderNetwork;
    }

    public void applyHolderNetwork(Network holderNetwork, bool reverse = false)
    {
        for (int i = 0; i < layerArray.Length; i++)
        {
            //for each layer

            for (int j = 0; j < layerArray[i].Length; j++)
            {
                //for each neuron
                for (int k = 0; k < layerArray[i][j].Weights.Count; k++)
                {
                    //for each Weight
                    if (reverse)
                    {
                        layerArray[i][j].Weights[k] += holderNetwork.layerArray[i][j].Weights[k];
                    }
                    else
                    {
                        layerArray[i][j].Weights[k] -= holderNetwork.layerArray[i][j].Weights[k];
                    }
                }
            }
        }
    }
}