- Try working on entire dataset !!

- Preprocessing
    - Still can add new features such as PTT
    - Adding P_Onset and T_Offset amplitudes to the features
    - Using something more robust than the medians/stds for each type of peak
        - median difference from the mean
        - 
    - Remove the completely NaN observations (29 of them) from output data
    - add a feature including how many NaNs are in the peak detection for each peak
    - merge redundant features (e.g. XX_intervals) into one (or more)
        - using a statistic or
        - training a model, e.g. RandomForest using only redundant features and yielding probability of a patient to be
          within a class
          
    - deal with nans after substractions
    
    
- Feature selection stuff from previous week
  

- Models
    - Push stuff to GitHub
    - SVMs, Gaussian Processes, Tensorflow, 
