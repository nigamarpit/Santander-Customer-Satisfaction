clear all
clc
% format long g

train = dlmread('train.csv',',',1,1);
test = dlmread('test.csv',',',1,0);
% id = test(:,1);
test = test(:,2:size(test,2));

train = preProcessing(train);
test = preProcessing(test);
disp('pre-processing');
% train = select0and1(train);
% disp('select0and1');

% train = normalization(train);
% test = normalization(test);
% disp('normalization');


% %pca
% labels = train(:, size(train, 2));
% train = train(:, 1:(size(train, 2)-1));
% 
% [pc, scores ,latent, ~, explained] = pca(train);
% 
% % explain = explained(find(explained >= 0.0001), :);
% % [row, ~] = size(explain);
% 
% row = size(train, 2)-10;
% 
% train = train * pc(:, 1:row);
% train = [train, labels];
% test = test * pc(:, 1:row);

csvwrite('trainPca10.csv', train);
csvwrite('testPca10.csv', test);

            
            