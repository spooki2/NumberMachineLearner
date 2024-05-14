using System.Drawing;
using MathNet.Numerics.LinearAlgebra;

namespace numberRecogniser;

public static class Model
{
    public static void drawVector(Vector<double> values,String name)
    {
        Model.DrawImageFromDoubles(values.ToArray(),name);

    }
    public static void DrawImageFromDoubles(double[] pixels, String name)
    {
        Bitmap image = new Bitmap(28, 28);

        for (int i = 0; i < pixels.Length; i++)
        {
            int x = i % 28;
            int y = i / 28;

            double pixelValue = pixels[i];
            int colorValue = (int)(pixelValue * 255); // Map pixel value from range [0, 1] to [0, 255]
            //clamp color
            if (colorValue > 255)
            {
                colorValue = 255;
            }
            else if (colorValue < 0)
            {
                colorValue = 0;
            }

            Color color = Color.FromArgb(colorValue, colorValue, colorValue);
            image.SetPixel(x, y, color);
        }

        image.Save(name, System.Drawing.Imaging.ImageFormat.Png);
        //throw excep
    }
}