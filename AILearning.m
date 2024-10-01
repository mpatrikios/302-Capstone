imds = imageDatastore('dataset', 'IncludeSubfolders', true, 'LabelSource', 'foldernames'); % Loads Dataset

[trainImgs, valImgs, testImgs] = splitEachLabel(imds, 0.7, 0.15, 0.15, 'randomize'); %Splits Dataset

layers = [imageInputLayer([224 224 3])        % Input layer (assumes RGB images)
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2,'Stride',2)

    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(numel(categories(imds.Labels)))  % Number of classes
    softmaxLayer
    classificationLayer];

augmenter = imageDataAugmenter('RandRotation', [-10, 10], 'RandXReflection', true);
augTrainImgs = augmentedImageDatastore([224 224], trainImgs, 'DataAugmentation', augmenter);


options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'ValidationData', valImgs, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(trainImgs, layers, options);

predictedLabels = classify(net, testImgs);
actualLabels = testImgs.Labels;
accuracy = mean(predictedLabels == actualLabels);

confusionchart(actualLabels, predictedLabels);

save('plantFungiModel.mat', 'net');




