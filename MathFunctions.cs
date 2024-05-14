using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Complex32;

namespace numberRecogniser;

public class MathFunctions
{

    //public static Func<double, double> ReLU = x => Math.Max(0, x);
    public static double ReLU(double x)
    {
        double e2x = Math.Exp(2 * x);
        return (e2x - 1) / (e2x + 1);
    }

    //public static Func<double, double> ReLUtag = x => x > 0 ? 1 : 0;
    public static double ReLUtag(double x)
    {
        double tanhX = Math.Tanh(x);
        return 1 - tanhX * tanhX;
    }



    public static Vector<double> softMax(Vector<double> output)
    {
        double maxInput = output.Max();
        Vector<double> normalizedInput = output.Subtract(maxInput);
        Vector<double> expValues = normalizedInput.PointwiseExp();
        double expSum = expValues.Sum();
        Vector<double> softmaxOutput = expValues.Divide(expSum);

        return softmaxOutput;
    }




    public static Vector<double> getDesireVector(int label)
    {
        Vector<double> desireVector = Vector<double>.Build.Dense(10);
        desireVector[label] = 1;
        return desireVector;
    }


}