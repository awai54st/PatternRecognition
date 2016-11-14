clear all;
close all;
clc;


%load face data
load face.mat

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
eigenvalues = flipud(diag(D));
%eigenvalues = eigenvalues(1:468);

% Broken-stick stopping rule

p = 2576;

for k = 1:p
    sum = 0;
    for i = k:p
        sum = sum + 1/i;
    end
    sum = sum/p;
    if eigenvalues(k) > sum
        disp(k);
    end
end