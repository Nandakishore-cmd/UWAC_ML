
opts = detectImportOptions('C:\Users\DBA\Downloads\Documents\underwater_modulation_dataset.csv');
dataset = readtable('C:\Users\DBA\Downloads\Documents\underwater_modulation_dataset.csv', opts);

disp('Column Names in Dataset:');
disp(dataset.Properties.VariableNames);

modulationCol = 'Modulation_Technique';

if ~ismember(modulationCol, dataset.Properties.VariableNames)
    error('Column "%s" not found in dataset. Check column names.', modulationCol);
end

features = dataset(:, 1:end-1);
labels = categorical(dataset.(modulationCol));

%(70% Train, 30% Test)
cv = cvpartition(height(dataset), 'HoldOut', 0.3);
trainData = dataset(training(cv), :);
testData = dataset(test(cv), :);

X_train = trainData(:, 1:end-1);
Y_train = categorical(trainData.(modulationCol));

X_test = testData(:, 1:end-1);
Y_test = categorical(testData.(modulationCol));

% Decision Tree Model
DT_model = fitctree(X_train, Y_train);

% Random Forest Model
RF_model = fitcensemble(X_train, Y_train, 'Method', 'Bag');

% Predicting using both models
pred_DT = predict(DT_model, X_test);
pred_RF = predict(RF_model, X_test);

% Confusion Matrices
figure;
subplot(1,2,1);
confusionchart(Y_test, pred_DT);
title('Random Forest Confusion Matrix');

subplot(1,2,2);
confusionchart(Y_test, pred_RF);
title('Decision Tree Confusion Matrix');

SNR_dB = linspace(0, 30, length(Y_test));
SNR_linear = 10.^(SNR_dB/10);

BER_DT = zeros(size(SNR_dB));
BER_RF = zeros(size(SNR_dB));

numSamples = min(length(pred_DT), length(SNR_dB));
uniqueLabels = categories(Y_test);
numClasses = length(uniqueLabels);
trueLabels = grp2idx(Y_test);
predLabels_DT = grp2idx(pred_DT);
predLabels_RF = grp2idx(pred_RF);

confMat_DT = confusionmat(trueLabels, predLabels_DT);
confMat_RF = confusionmat(trueLabels, predLabels_RF);

accuracy_DT = sum(diag(confMat_DT)) / sum(confMat_DT(:));
precision_DT = diag(confMat_DT) ./ sum(confMat_DT, 1)';
recall_DT = diag(confMat_DT) ./ sum(confMat_DT, 2);
f1_DT = 2 * (precision_DT .* recall_DT) ./ (precision_DT + recall_DT); 

accuracy_RF = sum(diag(confMat_RF)) / sum(confMat_RF(:));
precision_RF = diag(confMat_RF) ./ sum(confMat_RF, 1)'; 
recall_RF = diag(confMat_RF) ./ sum(confMat_RF, 2);
f1_RF = 2 * (precision_RF .* recall_RF) ./ (precision_RF + recall_RF); 

fprintf('\nRandom Forest Model Performance:\n');
fprintf('Accuracy: %.2f%%\n', accuracy_DT * 100);
for i = 1:numClasses
    fprintf('Class %s -> Precision: %.2f, Recall: %.2f, F1-Score: %.2f\n', ...
            uniqueLabels{i}, precision_DT(i), recall_DT(i), f1_DT(i));
end

fprintf('\nDecision Tree Model Performance:\n');
fprintf('Accuracy: %.2f%%\n', accuracy_RF * 100);
for i = 1:numClasses
    fprintf('Class %s -> Precision: %.2f, Recall: %.2f, F1-Score: %.2f\n', ...
            uniqueLabels{i}, precision_RF(i), recall_RF(i), f1_RF(i));
end


