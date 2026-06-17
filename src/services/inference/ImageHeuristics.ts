import * as tf from '@tensorflow/tfjs';

export interface ImageStats {
  meanIntensity: number;
  maxIntensity: number;
  variance: number;
  brightSpotSize: number;
  brightSpotCenter: [number, number]; // x, y (0 to 1)
  isAbnormal: boolean;
}

export async function analyzeImageHeuristics(tensor: tf.Tensor): Promise<ImageStats> {
  // Ensure we are working with a 2D/3D tensor of shape [H, W, Channels] or [1, H, W, Channels]
  let img = tensor;
  if (img.shape.length === 4) {
    img = img.squeeze([0]);
  }
  
  // Convert to grayscale
  const grayscale = tf.mean(img, 2);
  const [h, w] = grayscale.shape as [number, number];
  
  // Compute basic stats
  const meanTensor = tf.mean(grayscale);
  const maxTensor = tf.max(grayscale);
  
  const meanIntensity = (await meanTensor.data())[0];
  const maxIntensity = (await maxTensor.data())[0];
  
  // Compute variance
  const varianceTensor = tf.mean(tf.square(grayscale.sub(meanIntensity)));
  const variance = (await varianceTensor.data())[0];
  
  // Find bright spots (potential anomalies)
  // Threshold: anything brighter than mean + 2 * stddev, or an absolute threshold
  const stddev = Math.sqrt(variance);
  const threshold = Math.max(meanIntensity + 2 * stddev, 0.7); // At least 70% brightness
  
  const brightMask = grayscale.greaterEqual(tf.scalar(threshold));
  
  // Calculate size of bright spot
  const brightSpotSizeTensor = tf.sum(tf.cast(brightMask, 'float32'));
  const brightSpotSize = (await brightSpotSizeTensor.data())[0] / (h * w);
  
  // Find approximate center of mass of the bright spot
  // We'll create coordinate grids
  const xCoords = tf.tile(tf.expandDims(tf.range(0, w), 0), [h, 1]);
  const yCoords = tf.tile(tf.expandDims(tf.range(0, h), 1), [1, w]);
  
  const maskedX = xCoords.mul(tf.cast(brightMask, 'float32'));
  const maskedY = yCoords.mul(tf.cast(brightMask, 'float32'));
  
  const sumX = (await tf.sum(maskedX).data())[0];
  const sumY = (await tf.sum(maskedY).data())[0];
  
  let centerX = 0.5;
  let centerY = 0.5;
  
  if (brightSpotSize > 0) {
    const numBrightPixels = brightSpotSize * h * w;
    centerX = (sumX / numBrightPixels) / w;
    centerY = (sumY / numBrightPixels) / h;
  }
  
  // Determine if abnormal
  // If there's a distinct bright spot that isn't the whole image, it's likely an anomaly (tumor, fracture, etc)
  const isAbnormal = brightSpotSize > 0.005 && brightSpotSize < 0.25 && maxIntensity > 0.75;
  
  // Cleanup
  grayscale.dispose();
  meanTensor.dispose();
  maxTensor.dispose();
  varianceTensor.dispose();
  brightMask.dispose();
  brightSpotSizeTensor.dispose();
  xCoords.dispose();
  yCoords.dispose();
  maskedX.dispose();
  maskedY.dispose();
  
  return {
    meanIntensity,
    maxIntensity,
    variance,
    brightSpotSize,
    brightSpotCenter: [centerX, centerY],
    isAbnormal
  };
}
