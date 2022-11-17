# Pipeline

- preprocessing
  - get rid of noise
    - wavelets, kernels, B-splines (?), [low-pass filter, high-pass filter, butter worth filter]
  - extract single heartbeat from the data
    - neurokit vs biosppy libraries
    - or not (Guglielmo)
- extract features
    - ones to watch:
        - Amplitude of S
        - Amplitude of R
        - Length of QT interval
        - length of PR interval & PR segment
    - clustering for feature selection (with all features, to see which are relevant)
    - find things in literature (e.g. PTT)
    - automatic feature selection
    
