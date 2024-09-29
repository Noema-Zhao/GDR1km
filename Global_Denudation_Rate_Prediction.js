/*
 * ----------------------------------------------------------------------------------
 * Title: A high-resolution map of global denudation rates based on machine learning
 * Description: This JavaScript script is designed for use in Google Earth Engine (GEE)
 *              to analyze and predict global denudation rates using a Random Forest 
 *              machine learning model. The model is trained on various environmental 
 *              factors such as slope, precipitation, vegetation indices, and temperature.
 *
 * Author: Jiaxi Zhao
 * Department of Atmospheric and Oceanic Sciences,
 * School of Physics, Peking University, Beijing 100871, China
 * Email: noemazhao@stu.pku.edu.cn
 * 
 * Version: 1.0
 * Date: September 30, 2024
 *
 * Usage: 
 *  - This script processes geospatial environmental data to train a Random Forest model
 *    within Google Earth Engine (GEE).
 *  - Input: 10Be-based denudation rate data and environmental predictors in GEE assets.
 *  - Outputs: Predicted global denudation map, feature importance, and model evaluation.
 *
 * Requirements:
 *  - Google Earth Engine account
 *  - Access to environmental data assets within GEE
 *
 * Notes:
 *  - Ensure all required assets are loaded correctly before running the script.
 *  - Contact the author if there are any issues or questions regarding this code.
 *
 * Please cite relevant research if you use or modify this code.
 * ----------------------------------------------------------------------------------
 */

 var dataset = ee.FeatureCollection("projects/ee-maumassant/assets/denudation_all_features");
var sample = dataset;
sample = dataset.randomColumn();
var split = 0.8;
var training = sample.filter(ee.Filter.lt('random', split));
var testing = sample.filter(ee.Filter.gte('random', split));

var regression = ee.Classifier.smileRandomForest({
  numberOfTrees: 500,
  minLeafPopulation: 1,
  bagFraction: 0.9,
  seed: 0
}).setOutputMode('REGRESSION')
  .train({
    features: training,
    classProperty: 'log_denud',
    inputProperties: [
      'Elevation', 'Slope',
      'Runoff',
      'MAT', 'Tmin', 'Tseason', 'MDR',
      'MAP', 'Pseason', 'Pwet',
      'NDVI', 'EVI',
      'PGA',
      'Lithology'
    ]
  });
print('Regression RF', regression.explain());

// calculate relative importance of predictors
var importance = ee.Dictionary(regression.explain().get('importance'))
var sum = importance.values().reduce(ee.Reducer.sum())
var relativeImportance = importance.map(function(key, val) {
   return (ee.Number(val).multiply(100)).divide(sum)
  })
  

// make predictions of basin-wide denudation rates with trained model
var prediction = testing.classify(regression, 'log_denud');
print('prediction:', prediction);

// model evaluation
var actuals = testing.aggregate_array('log_denud');
var predicted = prediction.aggregate_array('log_denud');
var actualsList = actuals.getInfo();
var predictedList = predicted.getInfo();

// Classify the test data for accuracy test
var testData = testing.classify(regression, 'log_denud_Prediction').map(function(data) {
  return data.set('line', data.get('log_denud'));
});

// 14 environmental variables
var Elevation = ee.Image("projects/ee-maumassant/assets/Elevation").rename('Elevation');
var Slope = ee.Image("projects/ee-maumassant/assets/Slope").rename('Slope');
var Runoff = ee.Image("projects/ee-maumassant/assets/Runoff").rename('Runoff');
var MAT = ee.Image("projects/ee-maumassant/assets/MAT").rename('MAT');
var Tmin = ee.Image("projects/ee-maumassant/assets/Tmin").rename('Tmin');
var Tseason = ee.Image("projects/ee-maumassant/assets/Tseason").rename('Tseason');
var MDR = ee.Image("projects/ee-maumassant/assets/MDR").rename('MDR');
var MAP = ee.Image("projects/ee-maumassant/assets/MAP").rename('MAP');
var Pseason = ee.Image("projects/ee-maumassant/assets/Pseason").rename('Pseason');
var Pwet = ee.Image("projects/ee-maumassant/assets/Pwet").rename('Pwet');
var NDVI = ee.Image("projects/ee-maumassant/assets/NDVI").rename('NDVI');
var EVI = ee.Image("projects/ee-maumassant/assets/EVI").rename('EVI');
var PGA = ee.Image("projects/ee-maumassant/assets/PGA").rename('PGA');
var Lithology = ee.Image("projects/ee-maumassant/assets/Lithology").rename('Lithology');

var resample = function(image) {
  return image.resample('bilinear').reproject({
    crs: Elevation.projection().crs(),
    scale: 1000
  });
};

var Slope = resample(Slope);
var Runoff = resample(Runoff);
var MAT = resample(MAT);
var Tmin = resample(Tmin);
var Tseason = resample(Tseason);
var MDR = resample(MDR);
var MAP = resample(MAP);
var Pseason = resample(Pseason);
var Pwet = resample(Pwet);
var NDVI = resample(NDVI);
var EVI = resample(EVI);
var PGA = resample(PGA);
var Lithology = resample(Lithology);

var combined = ee.Image([Elevation, Slope, Runoff, MAT, Tmin, Tseason, MDR, MAP, Pseason, Pwet, NDVI, EVI, PGA, Lithology]).updateMask(Elevation);

// make predictions of global denudation rates with trained model
var predictionLogDenud = combined.classify(regression, 'log_denud');
print('predictionLogDenud', predictionLogDenud);

var predictionDenud = predictionLogDenud.exp();
var predictionDenud = predictionDenud.rename('predictedDenudation');
Map.addLayer(predictionDenud, { min: 0, max: 500, palette: ['purple', 'blue', 'cyan', 'green', 'yellow', 'red'] }, 'Log Denudation Prediction');

Export.image.toDrive({
  image: predictionDenud,
  description:  'Global_Denudation_Rate_1km',
  fileFormat: 'GeoTIFF',
  maxPixels: 1e13
});

