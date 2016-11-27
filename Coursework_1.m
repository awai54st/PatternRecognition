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
VNormalized = normc(V_alternative);
VNormalizedFlip = fliplr(VNormalized);


[reconImages468, reconTestImages468] = faceRecog(A, testImageA, 468, VNormalizedFlip, imageMean, training_size, test_size, test_data);
[reconImages200, reconTestImages200] = faceRecog(A, testImageA, 200, VNormalizedFlip, imageMean, training_size, test_size, test_data);
[reconImages20, reconTestImages20] = faceRecog(A, testImageA, 20, VNormalizedFlip, imageMean, training_size, test_size, test_data);

%Get reconstruction error according to error = sum of unused eigenvalues
%{
numOfEigenvector = 468;
eigenValuesAscending = abs(diag(D_alternative));
for i = 1:numOfEigenvector
    reconError(i) = sum(eigenValuesAscending(1 : training_size-i));
end
plot(reconError);
xlabel('Number of Eigenvectors');
ylabel('Reconstruction Error');
%}


%Again get reconstruction error according to sqrt((x - xRecon)^2) =
%sqrt((x - (xMean + reconImages468))^2) = sqrt((A - reconImages468)^2)
%{
trainingReconError468 = sqrt((A - reconImages468).^2);
meanTrainingError468 = mean(trainingReconError468);
trainingReconError200 = sqrt((A - reconImages200).^2);
meanTrainingError200 = mean(trainingReconError200);
trainingReconError20 = sqrt((A - reconImages20).^2);
meanTrainingError20 = mean(trainingReconError20);
x = 1:468;
plot(x, meanTrainingError468, x, meanTrainingError200, x, meanTrainingError20);

testingReconError468 = sqrt((testImageA - reconTestImages468).^2);
meanTestingError468 = mean(testingReconError468);
...
%}

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


%% Question 2)b)
NNNumOfEigenvector = 129;
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
recongAccuracy = numOfCorrectRecog / image_size;



