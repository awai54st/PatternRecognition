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

mean_image = mean(training_data, 2); % mean image from training set
A = (training_data-repmat(mean_image, [1, training_size]));

%S = A * A' / training_size; % data covariance matrix using 1/N*A*At
S_alternative = A' * A / training_size; % data covariance matrix using 1/N*At*A

%[V, D] = eig(S); % calculate V as the eigenvectors and D as eigenvalues (in D's diagonal)
[V_alternative, D] = eig(S_alternative); % calculate V as the eigenvectors and D as eigenvalues (in D's diagonal)
V = A * V_alternative;  % using 1/N*At*A gives same eigenvalues, and V = A*eigenvectors when using 1/N*At*A

% Plot the magnitude of eigenvalues
figure;
subplot(1,2,1);
plot(flipud(abs(diag(D))));
%plot(sort(abs(diag(D)), 'descend')); Is it necessary to use sort?
xlabel('No. of Eigenvalues');
ylabel('Magnetude of Eigenvalues');
title('Eigenvalues');
subplot(1,2,2);
plot(flipud(abs(diag(D))));
axis([200 600 0 1000]);
xlabel('No. of Eigenvalues');
ylabel('Magnetude of Eigenvalues');
title('Eigenvalues (Zoomed in at the last non-zero eigenvalues)');

% Print first 64 eigenfaces

large_eigenfaces = zeros(8*img_height, 8*img_width);

for j = 1:8
    for i = 1:8
        input_tmp = V(:,end-(i-1)-8*(j-1));
        mat = vec2mat(input_tmp,img_width);
        large_eigenfaces(img_height*(j-1)+1:img_height*j, (img_width*(i-1)+1):(img_width*i)) = mat;
    end
end

figure;
imshow(mat2gray(large_eigenfaces));


% Print 437-500 eigenfaces

large_eigenfaces = zeros(8*img_height, 8*img_width);

for j = 1:8
    for i = 1:8
        input_tmp = V(:,end-(i-1)-8*(j-1)-64);
        mat = vec2mat(input_tmp,img_width);
        large_eigenfaces(img_height*(j-1)+1:img_height*j, (img_width*(i-1)+1):(img_width*i)) = mat;
    end
end

figure;
imshow(mat2gray(large_eigenfaces));










