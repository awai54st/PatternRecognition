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

large_image = zeros(img_height*class_size,img_width*class_number);
for j = 1:class_number
    for i=1:class_size
        input_tmp = X(:,class_size*(j-1)+i);
        mat = vec2mat(input_tmp,img_width);
        large_image((img_height*(i-1)+1):(img_height*i), (img_width*(j-1)+1):(img_width*j)) = mat;
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

mean_image = mean(training_data, 2); % mean image from training set
A = (training_data-repmat(mean_image, [1, training_size]));

S = A * A' / training_size; % data covariance matrix

[V, D] = eig(S); % calculate V as the eigenvectors and D as eigenvalues (in D's diagonal)



