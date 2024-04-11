using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace NumberRecogniser
{
    public static class CSV2JPEG
    {
        public static void Convert(double[] value)
        {
            // Assuming a 28x28 grayscale image, each value represents one pixel
            int width = 28;
            int height = 28;

            using (Bitmap bitmap = new Bitmap(width, height))
            {
                for (int i = 0; i < height; i++)
                {
                    for (int j = 0; j < width; j++)
                    {
                        // Convert the double value to a byte (0-255 scale)
                        byte val = (byte)(value[i * width + j] * 255);
                        Color color = Color.FromArgb(val, val, val); // Grayscale
                        bitmap.SetPixel(j, i, color);
                    }
                }

                // Save the image
                bitmap.Save("img.jpg", ImageFormat.Jpeg);
            }
        }
    }
}