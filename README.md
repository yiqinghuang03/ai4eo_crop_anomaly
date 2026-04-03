# ai4eo final project

## project overview
This project explores an unsupervised anomaly detection pipeline for Earth Observation data.  
The main goal is to temporally identify potentially abnormal regions within crop fields from Sentinel-2 imagery.

## pipeline description

1. **data preparation**
   - Load Sentinel-2 image patches from selected agricultural areas
   - Select relevant multispectral bands
   - Apply basic preprocessing such as normalization and augmentation 

2. **feature learning**
   - Use a self-supervised learning approach to learn features from multi bands.
   - SimCLR style method is used

3. **embedding extraction**
   - After representation learning, image patches are fed to the encoder
   - Each patch is mapped to a feature embedding in embedding space

4. **anomaly scoring**
   - Distances in embedding space are used as anomaly scores
   - Patches that are farther from normal clusters or neighborhood structure are defined as more likely to be anomalous

5. **interpretation/ visualization**
   - The expected final output is a spatial anomaly map showing which regions of a field may require further inspection

## data source
The project uses Sentinel-2 satellite imagery.

### bands
- B2 (Blue)
- B3 (Green)
- B4 (Red)
- B8 (NIR)
- B11 (SWIR)
- B12 (SWIR)

### why this data
- sufficient spatial information 
- openly accessible EO data for researchers 

## current Status

Currently the project has reached the stage of a defined pipeline and partial implementation. However:
- The code is currently under debugging and is not yet fully executable end-to-end.
- The repository is not fully up to date, as some recent local changes (still in experimenting) have not been synchronized.

## planned Next Steps

1. Generate anomaly scores for all patches
2. Visualization from patch-level into spatial maps with relative relations between neighbor fields. 
3. Improve interpretability and output more understandable graphics.
 
