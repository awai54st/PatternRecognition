%%%%%%%% Machine Learning for Computer Vision   %%%%%%%%
%%%%%%%% Matlab Tutorial                        %%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% SUPPORT VECTOR MACHINES %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% clear all, close all, clc
clear all
close all
clc

%import libraries
config

%% Read data
[heart_scale_label, heart_scale_inst] = libsvmread('heart_scale');

%% Access data and understand the "nature"
vector_1 = heart_scale_inst(1,:);
vector_2 = heart_scale_inst(2,:);

vector_1 = full (vector_1)';
vector_2 = full (vector_2)';
train_set = [vector_1 vector_2];

%% Split Data Cross validation - Spliting data to training - validation - testing sets...
train_data = heart_scale_inst(1:150,:);
train_label = heart_scale_label(1:150,:);
test_data = heart_scale_inst(151:270,:);
test_label = heart_scale_label(151:270,:);

%% Train model
model_linear = svmtrain(train_label, train_data, '-t 0');
model_precomputed = svmtrain(train_label, [(1:150)', train_data*train_data'], '-t 4');

%% Predict
[predict_label_L, accuracy_L, dec_values_L] = svmpredict(test_label, test_data, model_linear);
[predict_label_P, accuracy_P, dec_values_P] = svmpredict(test_label, [(1:120)', test_data*train_data'], model_precomputed);

accuracy_L
accuracy_P

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Random Forests %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% clear all, close all, clc
clear all
close all
clc

%% create toy data
N= 50;
t = linspace(0.5, 2*pi, N);
x = t.*cos(t);
y = t.*sin(t);

t = linspace(0.5, 2*pi, N);
x2 = t.*cos(t+2);
y2 = t.*sin(t+2);

t = linspace(0.5, 2*pi, N);
x3 = t.*cos(t+4);
y3 = t.*sin(t+4);

X= [[x' y']; [x2' y2']; [x3' y3']];
X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];

scatter(X(:,1), X(:,2), 20, Y)

%% set parameters
rand('state', 0);
randn('state', 0);

opts= struct;
opts.depth= 9;
opts.numTrees= 100;
opts.numSplits= 5;
opts.verbose= true;
opts.classifierID= 2;

%% train
tic;
m= forestTrain(X, Y, opts);
timetrain= toc;

%% test
tic;
yhatTrain = forestTest(m, X);
timetest= toc;

%% Random Classifiers... 
% Look at classifier distribution for fun, to see what classifiers were
% chosen at split nodes and how often
fprintf('Classifier distributions:\n');
classifierDist= zeros(1, 4);
unused= 0;
for i=1:length(m.treeModels)
    for j=1:length(m.treeModels{i}.weakModels)
        cc= m.treeModels{i}.weakModels{j}.classifierID;
        if cc>1 %otherwise no classifier was used at that node
            classifierDist(cc)= classifierDist(cc) + 1;
        else
            unused= unused+1;
        end
    end
end
fprintf('%d nodes were empty and had no classifier.\n', unused);
for i=1:4
    fprintf('Classifier with id=%d was used at %d nodes.\n', i, classifierDist(i));
end

%% plot results
xrange = [-1.5 1.5];
yrange = [-1.5 1.5];
inc = 0.02;
[x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
image_size = size(x);
xy = [x(:) y(:)];

[yhat, ysoft] = forestTest(m, xy);
decmap= reshape(ysoft, [image_size 3]);
decmaphard= reshape(yhat, image_size);

figure;
subplot(121);
imagesc(xrange,yrange,decmaphard);
hold on;
set(gca,'ydir','normal');
cmap = [1 0.8 0.8; 0.95 1 0.95; 0.9 0.9 1];
colormap(cmap);
plot(X(Y==1,1), X(Y==1,2), 'o', 'MarkerFaceColor', [.9 .3 .3], 'MarkerEdgeColor','k');
plot(X(Y==2,1), X(Y==2,2), 'o', 'MarkerFaceColor', [.3 .9 .3], 'MarkerEdgeColor','k');
plot(X(Y==3,1), X(Y==3,2), 'o', 'MarkerFaceColor', [.3 .3 .9], 'MarkerEdgeColor','k');
hold off;
title(sprintf('%d trees, Train time: %.2fs, Test time: %.2fs\n', opts.numTrees, timetrain, timetest));

subplot(122);
imagesc(xrange,yrange,decmap);
hold on;
set(gca,'ydir','normal');
plot(X(Y==1,1), X(Y==1,2), 'o', 'MarkerFaceColor', [.9 .3 .3], 'MarkerEdgeColor','k');
plot(X(Y==2,1), X(Y==2,2), 'o', 'MarkerFaceColor', [.3 .9 .3], 'MarkerEdgeColor','k');
plot(X(Y==3,1), X(Y==3,2), 'o', 'MarkerFaceColor', [.3 .3 .9], 'MarkerEdgeColor','k');
hold off;

title(sprintf('Train accuracy: %f\n', mean(yhatTrain==Y)));