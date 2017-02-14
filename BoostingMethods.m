%% CS155 mini Project 1 
clear X y cut_out

%% Load data 
% double click the data and save as matrix. 
% train 2008/train
%% 1. sort data

X = train2008(:,2:382);
y = train2008(:,383); 

%% 2. delete the original data 
clear train2008 % to save memory
%% 2.1 delete redundant data (column consisting of same value)
cut_out = [1, 2, 12, 14, 16, 47, 58, 129, 130, 131, 135, 136, 137, 254, 258]
X(:,cut_out) = []; 
%%

%% 2.2 Normalization 
for jj =1: 366; 
    X_nor(:,jj) = (X(:,jj) - mean(X(:,jj)))./std(X(:,jj));
    
end
%% 3.1. Split data into training data and validation data

tree_array = 10:10:400;
learn = 0.2:0.2:1;
accuracy_train = zeros(11,5);
accuracy_val = zeros(11,5);
met = {'LogitBoost','GentleBoost', 'RobustBoost', 'LPBoost', 'TotalBoost'}


for j =1:5;
    for i =10:20; 
    
tic;
n_tree = tree_array(i);
trainRatio = 0.7;
valRatio = 0.3;
testRatio = 0 ;
%%
[trainInd,valInd,testInd] = dividerand(64667,trainRatio,valRatio,testRatio);
size_train = size(trainInd);
size_val = size(valInd);
%% 3. Train model with ensemble algorithms
ClassTreeEns = fitensemble(X(trainInd,:),y(trainInd),met{j},n_tree,'Tree'); % train model with 70% of data.
label_train = predict(ClassTreeEns,X(trainInd,:)); % Predicted y_train
label_val = predict(ClassTreeEns,X(valInd,:)); % Predicted y_val

accuracy_train(i-9,j) = 1 - sum(abs(y(trainInd)-label_train))/size_train(2) % trainingaccuracy 
accuracy_val(i-9,j) = 1 - sum(abs(y(valInd)-label_val))/size_val(2) % validation accuracy

clear trainInd valInd 
toc;
    end
end
%% Plot
figure;
for i=1:3;
    plot(100:10:200,accuracy_train(:,i),'--')
    
    hold on;
end

for i=1:3;
    plot(100:10:200,accuracy_val(:,i))
    hold on;
end
legend('LogitBoost train', 'GentleBoost train', 'RobustBoost train', 'LogitBoost val', 'GentleBoost val', 'RobustBoost val')
title('Training & Validation Accuracies vs number of estimators')