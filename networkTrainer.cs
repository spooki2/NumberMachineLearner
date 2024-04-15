namespace numberRecogniser;

using MathNet.Numerics.LinearAlgebra;

public class networkTrainer
{
    public static void run(Vector<double> input, Neuron[] layer, int label)
    {
        //Propagate.forward(input, layer);
        Matrix<double> massWeight = Matrix<double>.Build.Dense(0, 0);
        Vector<double> massBias = Vector<double>.Build.Dense(0);

        //where loop could go
        Matrix<double> newWeight = Matrix<double>.Build.Dense(0, 0);
        Vector<double> newBias = Vector<double>.Build.Dense(0);
        (newWeight, newBias) = Propagate.backward(layer, label); //BRINGS BACK ONLY THE FIRST NEURON
        massWeight = Matrix<double>.Build.Dense(newWeight.RowCount, newWeight.ColumnCount);
        massBias = Vector<double>.Build.Dense(newBias.Count);
        massWeight += newWeight;
        massBias += newBias;

        for (int i = 0; i < layer.Length; i++)
        {
            Propagate.applyLayer(massWeight.Column(i), newBias,
                layer); //should be row but flipped, works either way.
        }


        /*
        double[] weights = new double[784];
        for (int i = 0; i < currLayer[0].Weights.Count; i++)
        {
            weights[i] = (currLayer[0].Weights[i]);
        }

        Model.DrawImageFromDoubles(weights, "output.png");
        */
    }
}