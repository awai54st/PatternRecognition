clc
clear all
close all

%load face data
load face.mat

%% Question 0

% Display all images

img_width = 56;
img_height = 46;
class_number = 52;
class_size = 10;
image_size = 520;

large_image = zeros(img_height*class_size,img_width*class_number);
for j = 1:class_number
    for i=1:class_size
        eachImage = X(:,class_size*(j-1)+i);
        eachImageMat = vec2mat(eachImage,img_width);
        %Insert each image into one large image
        large_image((img_height*(i-1)+1):(img_height*i), (img_width*(j-1)+1):(img_width*j)) = eachImageMat;
    end
end
imshow(mat2gray(large_image));

%% Question 1

% Perform k-fold cross-validationn using class cvpartition

k = 10;
c = cvpartition(l,'Kfold',k); % separate the index list, l, into k separations

training_data = X(:, training(c, 1));
test_data = X(:, test(c, 1));
training_size = size(training_data, 2);
test_size = size(test_data, 2);

imageMean = mean(training_data, 2); % mean image from training set
imageMean_show = vec2mat(imageMean, img_width);
imshow(mat2gray(imageMean_show));
%imwrite(mat2gray(imageMean_show),'mean.png')

A = (training_data-repmat(imageMean, [1, training_size]));
testImageA = (test_data-repmat(imageMean, [1, test_size]));

S = A * A' / training_size; % data covariance matrix using 1/N*A*At
S_alternative = A' * A / training_size; % data covariance matrix using 1/N*At*A

[V, D] = eig(S); % calculate V as the eigenvectors and D as eigenvalues (in D's diagonal)
%V = normc(V);
[V_alternative, D_alternative] = eig(S_alternative); % calculate V as the eigenvectors and D as eigenvalues (in D's diagonal)
temp = V_alternative;
V_alternative = A * V_alternative;  % using 1/N*At*A gives same eigenvalues, and V = A*eigenvectors when using 1/N*At*A

% Plot the magnitude of eigenvalues
% figure;
% subplot(1,2,1);
% plot(flipud(abs(diag(D))));
% xlabel('No. of Eigenvalues');
% ylabel('Magnetude of Eigenvalues');
% title('Eigenvalues');
% subplot(1,2,2);
% plot(flipud(abs(diag(D))));
% axis([200 600 0 1000]);
% xlabel('No. of Eigenvalues');
% ylabel('Magnetude of Eigenvalues');
% title('Eigenvalues (Zoomed in at the last non-zero eigenvalues)');

figure;
subplot(1,2,1);
plot(flipud(abs(diag(D))));
axis([200 600 0 1000]);
xlabel('No. of Eigenvalues');
ylabel('Magnetude of Eigenvalues');
title('Eigenvalues from Direct PCA');
subplot(1,2,2);
plot(flipud(abs(diag(D_alternative))));
axis([200 600 0 1000]);
xlabel('No. of Eigenvalues');
ylabel('Magnetude of Eigenvalues');
title('Eigenvalues from Alternative PCA');

% Print first 64 eigenfaces - direct pca

large_eigenfaces = zeros(8*img_height, 8*img_width);


for j = 1:8
    for i = 1:8
        input_tmp = V(:,end-(i-1)-8*(j-1));
        mat = vec2mat(input_tmp,img_width);
        mat = normc(mat);
        large_eigenfaces(img_height*(j-1)+1:img_height*j, (img_width*(i-1)+1):(img_width*i)) = mat;
    end
end

figure;
imshow(mat2gray(large_eigenfaces));


% Print first 64 eigenfaces - alternative pca

large_eigenfaces = zeros(8*img_height, 8*img_width);

for j = 1:8
    for i = 1:8
        input_tmp = V_alternative(:,end-(i-1)-8*(j-1));
        mat = vec2mat(input_tmp,img_width);
        mat = normc(mat);
        large_eigenfaces(img_height*(j-1)+1:img_height*j, (img_width*(i-1)+1):(img_width*i)) = mat;
    end
end

figure;
imshow(mat2gray(large_eigenfaces));



%% Question 2)a)

%-Reconstruct training data-
%Normalize eigenvectors, 
%vectors are fliped because it was ordered ascendingly
%{
VNormalized = normc(V_alternative);
VNormalizedFlip = fliplr(VNormalized);


[reconImages468, reconTestImages468] = faceRecog(A, testImageA, 468, VNormalizedFlip, imageMean, training_size, test_size, test_data);
[reconImages200, reconTestImages200] = faceRecog(A, testImageA, 200, VNormalizedFlip, imageMean, training_size, test_size, test_data);
[reconImages20, reconTestImages20] = faceRecog(A, testImageA, 20, VNormalizedFlip, imageMean, training_size, test_size, test_data);

%Get reconstruction error according to error = sum of unused eigenvalues

numOfEigenvector = 468;
eigenValuesAscending = abs(diag(D_alternative));
for i = 1:numOfEigenvector
    reconError(i) = sum(eigenValuesAscending(1 : training_size-i));
end
plot(reconError);
xlabel('Number of Eigenvectors');
ylabel('Reconstruction Error');



%drawing
figure;
subplot(3,4,1);
imshow(mat2gray(vec2mat(X(:,1), img_width)));
title('Training data');
subplot(3,4,2);
imshow(mat2gray(vec2mat(reconImages468(:,1), img_width)));
title('468 Eigenvectors');
subplot(3,4,3);
imshow(mat2gray(vec2mat(reconImages200(:,1), img_width)));
title('200 Eigenvectors');
subplot(3,4,4);
imshow(mat2gray(vec2mat(reconImages20(:,1), img_width)));
title('20 Eigenvectors');

subplot(3,4,5);
imshow(mat2gray(vec2mat(X(:,2), img_width)));
title('Training data');
subplot(3,4,6);
imshow(mat2gray(vec2mat(reconImages468(:,2), img_width)));
title('468 Eigenvectors');
subplot(3,4,7);
imshow(mat2gray(vec2mat(reconImages200(:,2), img_width)));
title('200 Eigenvectors');
subplot(3,4,8);
imshow(mat2gray(vec2mat(reconImages20(:,2), img_width)));
title('20 Eigenvectors');

subplot(3,4,9);
imshow(mat2gray(vec2mat(X(:,3), img_width)));
title('Training data');
subplot(3,4,10);
imshow(mat2gray(vec2mat(reconImages468(:,3), img_width)));
title('468 Eigenvectors');
subplot(3,4,11);
imshow(mat2gray(vec2mat(reconImages200(:,3), img_width)));
title('200 Eigenvectors');
subplot(3,4,12);
imshow(mat2gray(vec2mat(reconImages20(:,3), img_width)));
title('20 Eigenvectors');

%drawing for test images
figure;
subplot(3,4,1);
imshow(mat2gray(vec2mat(test_data(:,5), img_width)));
title('Test data');
subplot(3,4,2);
imshow(mat2gray(vec2mat(reconTestImages468(:,5), img_width)));
title('468 Eigenvectors');
subplot(3,4,3);
imshow(mat2gray(vec2mat(reconTestImages200(:,5), img_width)));
title('200 Eigenvectors');
subplot(3,4,4);
imshow(mat2gray(vec2mat(reconTestImages20(:,5), img_width)));
title('20 Eigenvectors');

subplot(3,4,5);
imshow(mat2gray(vec2mat(test_data(:,6), img_width)));
title('Test data');
subplot(3,4,6);
imshow(mat2gray(vec2mat(reconTestImages468(:,6), img_width)));
title('468 Eigenvectors');
subplot(3,4,7);
imshow(mat2gray(vec2mat(reconTestImages200(:,6), img_width)));
title('200 Eigenvectors');
subplot(3,4,8);
imshow(mat2gray(vec2mat(reconTestImages20(:,6), img_width)));
title('20 Eigenvectors');

subplot(3,4,9);
imshow(mat2gray(vec2mat(test_data(:,7), img_width)));
title('Test data');
subplot(3,4,10);
imshow(mat2gray(vec2mat(reconTestImages468(:,7), img_width)));
title('468 Eigenvectors');
subplot(3,4,11);
imshow(mat2gray(vec2mat(reconTestImages200(:,7), img_width)));
title('200 Eigenvectors');
subplot(3,4,12);
imshow(mat2gray(vec2mat(reconTestImages20(:,7), img_width)));
title('20 Eigenvectors');
%}

%% Question 2)b)
%NNNumOfEigenvector = 255;
NNTimeArray = 1:468;
for NNNumOfEigenvector = 1:468
    tic;
    for eachFold = 1:k
        NNTrainingData = X(:, training(c, eachFold));
        NNTestData = X(:, test(c, eachFold));
        NNTrainingSize = size(NNTrainingData, 2);
        NNTestSize = size(NNTestData, 2);    
        NNImageMean = mean(NNTrainingData, 2); % mean image from training set

        NNTrainingImageA = (NNTrainingData-repmat(NNImageMean, [1, NNTrainingSize]));
        NNTestImageA = (NNTestData-repmat(NNImageMean, [1, NNTestSize]));

        NNS = NNTrainingImageA' * NNTrainingImageA / NNTrainingSize; % data covariance matrix using 1/N*At*A
        [NNV_temp, NND] = eig(NNS); % calculate V as the eigenvectors and D as eigenvalues (in D's diagonal)

        %Normalize eigenvectors, 
        %vectors are fliped because it was ordered ascendingly
        NNVfliped = fliplr(NNV_temp);
        NNVChosen = NNVfliped(:, 1:NNNumOfEigenvector);

        %using 1/N*At*A gives same eigenvalues, and V = A*eigenvectors when using 1/N*At*A
        NNEigenvectorChosen = normc(NNTrainingImageA * NNVChosen);

        %High-dimentional data projects to low-dimention 
        NNeigenProjection = NNEigenvectorChosen' * NNTrainingImageA;
        NNTestEigenProjection = NNEigenvectorChosen' * NNTestImageA;

        %error is stored in reconError[foldNum, imageIndex]
        %Index of min error is stored in minIndex[foldNum, imageIndex]
        for i = 1 : NNTestSize
            [reconError(eachFold, i), minIndex(eachFold, i)] = min(sqrt(sum((repmat(NNTestEigenProjection(:,i),1,NNTrainingSize)-NNeigenProjection).^2)));
        end
    end
    NNTimeArray(NNNumOfEigenvector) = toc;
    
    %For each picture, there should be k pictures that belongs to it, where
    %K is the number of fold
    numOfCorrectRecog = 0;
    for i = 1 : NNTestSize
        recogIndex(:,i) = ceil(minIndex(:,i)/9);
        for j = 1 : k
            if recogIndex(j,i)==i
                numOfCorrectRecog = numOfCorrectRecog + 1;
            end
        end
    end
    recongAccuracy(NNNumOfEigenvector) = numOfCorrectRecog / image_size;
end
plot(recongAccuracy);
xlabel('Number of Eigenvectors');
ylabel('Reconstruction Accurracy');

yyaxis right;
plot(NNTimeArray);
ylabel('Time of Reconstruction/s');

%{
%% Question 3) One-vs-all

clear all;
close all;
clc;

load face.mat

%generate cross-validation sets
k = 10;
c = cvpartition(l,'Kfold',k); % separate the index list, l, into k separations
accuracy = zeros(k,1);
label_number = max(l);
conf_matrix = zeros(label_number, label_number);
img_width = 56;
img_height = 46;
%number of eigenvector chosen
numOfEigenvector = 467;
%PCA or RAW DATA
PCA_FLAG = 1;

%initialise training and testing data
if PCA_FLAG == 0
    training_data = zeros(468, img_width*img_height, k);
    training_label = zeros(468, 1, k);
    test_data = zeros(52, img_width*img_height, k);
    test_label = zeros(52, 1, k);
else 
    training_data = zeros(468, numOfEigenvector, k);
    training_label = zeros(468, 1, k);
    test_data = zeros(52, numOfEigenvector, k);
    test_label = zeros(52, 1, k);
end

%loop through a mesh of C * scale
crange = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 500, 1000, 2000];
srange = [0.1, 0.2, 0.3, 0.4, 0.5];
accuracy_overall = zeros(size(crange, 2)*size(srange, 2));
index = 1;

for scale = srange
for boxConstraints = crange

% loop through all k-folds
for j=1:k
    
    training_data_tmp = X(:, training(c, j))';
    training_label(:, :, j) = l(:, training(c, j))';
    test_data_tmp = X(:, test(c, j))';
    test_label(:, :, j) = l(:, test(c, j))';
    training_size = size(training_data_tmp, 1);
    test_size = size(test_data_tmp, 1);
    
    if PCA_FLAG == 1
        A = (training_data_tmp'-repmat(mean(training_data_tmp', 2), [1, training_size])); % PCA coefficients for training data
        B = (test_data_tmp'-repmat(mean(test_data_tmp', 2), [1, test_size])); % PCA coefficients for testing data
        S_alternative = A' * A / training_size; % data covariance matrix using 1/N*At*A
        [V_alternative, D_alternative] = eig(S_alternative); % calculate V as the eigenvectors and D as eigenvalues (in D's diagonal)
        V_alternative = A * V_alternative;  % using 1/N*At*A gives same eigenvalues, and V = A*eigenvectors when using 1/N*At*A
        
        %-Reconstruct training data-
        %Normalize eigenvectors, 
        %vectors are fliped because it was ordered ascendingly
        VNormalizedFlip = fliplr(normc(V_alternative));
        eigenvectorChosen = VNormalizedFlip(:, 1:numOfEigenvector);
        %High-dimentional data projects to low-dimention 
        training_data_tmp = A' * eigenvectorChosen;
        test_data_tmp = B' * eigenvectorChosen;
    end
    training_data(:,:,j) = training_data_tmp;
    test_data(:,:,j) = test_data_tmp;
    
end

% loop through all k-folds
%j=1;

t_begin = cputime; % timer
for j=1:k
    
    prob = [];
    
    % train
    for i=1:label_number
        training_label_normalised = double(training_label(:,:,j)==i); % generate label set
        model = fitcsvm(training_data(:,:,j), training_label_normalised,...
            'Standardize',true,'KernelFunction','polynomial','PolynomialOrder', 3, 'KernelScale',...
            'auto', 'BoxConstraint', boxConstraints);
        [~, p] = predict(model, test_data(:,:,j));
        prob = [prob, p(:, 2)];
    end
    
    % predict the class with the highest probability
    [~,pred] = max(prob,[],2);
    accuracy(j) = sum(pred == test_label(:,:,j)) ./ numel(test_label(:,:,j));    %# accuracy
    conf_matrix = conf_matrix + confusionmat(test_label(:,:,j), pred);     %# confusion matrix
    
end

t_OAA = cputime - t_begin; % timer

accuracy_overall(index) = sum (accuracy)/k;
% figure;
% imagesc(conf_matrix) % confusion matrix as the sum of all 10 folds

index = index + 1;
end
end


%% Question 3) One-vs-one, original X

clear all;
close all;
clc;

load face.mat

%generate cross-validation sets
k = 10;
c = cvpartition(l,'Kfold',k); % separate the index list, l, into k separations
accuracy = zeros(k,1);
label_number = max(l);
conf_matrix = zeros(label_number, label_number);
img_width = 56;
img_height = 46;
%number of eigenvector chosen
numOfEigenvector = 47;
%PCA or RAW DATA
PCA_FLAG = 1;

%intialise traininig and testing data
if PCA_FLAG == 0
    training_data = zeros(468, img_width*img_height, k);
    training_label = zeros(468, 1, k);
    test_data = zeros(52, img_width*img_height, k);
    test_label = zeros(52, 1, k);
else 
    training_data = zeros(468, numOfEigenvector, k);
    training_label = zeros(468, 1, k);
    test_data = zeros(52, numOfEigenvector, k);
    test_label = zeros(52, 1, k);
end

% loop through all k-folds
for fold=1:k
    
    training_data_tmp = X(:, training(c, fold))';
    training_label(:, :, fold) = l(:, training(c, fold))';
    test_data_tmp = X(:, test(c, fold))';
    test_label(:, :, fold) = l(:, test(c, fold))';
    training_size = size(training_data_tmp, 1);
    test_size = size(test_data_tmp, 1);
    
    if PCA_FLAG == 1
        A = (training_data_tmp'-repmat(mean(training_data_tmp', 2), [1, training_size])); % PCA coefficients for training data
        B = (test_data_tmp'-repmat(mean(test_data_tmp', 2), [1, test_size])); % PCA coefficients for testing data
        S_alternative = A' * A / training_size; % data covariance matrix using 1/N*At*A
        [V_alternative, D_alternative] = eig(S_alternative); % calculate V as the eigenvectors and D as eigenvalues (in D's diagonal)
        V_alternative = A * V_alternative;  % using 1/N*At*A gives same eigenvalues, and V = A*eigenvectors when using 1/N*At*A
        
        %-Reconstruct training data-
        %Normalize eigenvectors, 
        %vectors are fliped because it was ordered ascendingly
        VNormalizedFlip = fliplr(normc(V_alternative));
        eigenvectorChosen = VNormalizedFlip(:, 1:numOfEigenvector);
        %High-dimentional data projects to low-dimention 
        training_data_tmp = A' * eigenvectorChosen;
        test_data_tmp = B' * eigenvectorChosen;
    end
    training_data(:,:,fold) = training_data_tmp;
    test_data(:,:,fold) = test_data_tmp;
    
end

t_begin = cputime; % timer

for fold=1:k
    
    vote = [];
    
    for i=1:label_number-1
        for j=i+1:label_number

            training_label_a = training_label(:,:,fold)==i; % label set for class 1
            training_label_b = training_label(:,:,fold)==j; % label set for class 2
            training_label_tmp = training_label(:,:,fold); % total label set
            training_data_tmp = training_data(:,:,fold); % total data set
            training_label_cropped = training_label_tmp(training_label_a | training_label_b)==i; % crop out class 1&2
            training_data_cropped = training_data_tmp(training_label_a | training_label_b, :);
            model = fitcsvm(training_data_cropped, training_label_cropped,...
                'Standardize',true,'KernelFunction','rbf','RBF_Sigma', 0.2, 'KernelScale',...
                'auto');
            [p, ~] = predict(model, test_data(:,:,fold));
            vote = [vote, p*i + not(p)*j];
        end
    end

    pred = mode(vote, 2);
    %# predict the class with the highest probability
    accuracy(fold) = sum(pred == test_label(:,:,fold)) ./ numel(test_label(:,:,fold));    %# accuracy
    conf_matrix = conf_matrix + confusionmat(test_label(:,:,fold), pred);     %# confusion matrix

end

t_OAO = cputime - t_begin;


accuracy_overall = sum (accuracy)/k;
imagesc(conf_matrix) % confusion matrix

%% Plot log linear for OAA and OAO (C parameter)
figure;
semilogx(crange, accuracy_overall);
xlabel('C');
ylabel('accuracy');
title('Plot of Accuracy Against C Parameter for Linear Kernel');

%% Plot log linear mesh for REF case (C and sigma)
XLABEL=[ 1 2 4 8 16 32 64]
surf(log(X),Y,Z);
set(gca,'XTickLabel',XLABEL);
set(gca,'XTick',log(XLABEL));

%}
