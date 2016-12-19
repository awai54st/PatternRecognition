%% Question 3) One-vs-all, original X

clear all;
close all;
clc;

load face.mat

k = 10;
c = cvpartition(l,'Kfold',k); % separate the index list, l, into k separations
accuracy = zeros(k,1);
label_number = max(l);
conf_matrix = zeros(label_number, label_number);
img_width = 56;
img_height = 46;
%number of eigenvector chosen
%numOfEigenvector = [1:2:467];
numOfEigenvector = 255;
%PCA or RAW DATA
PCA_FLAG = 0;
LDA_FLAG = 1;


t_OAA = zeros(size(numOfEigenvector, 2));
accuracy_overall = zeros(size(numOfEigenvector, 2));
index = 1;

for pca = numOfEigenvector

if PCA_FLAG == 0
    training_data = zeros(468, img_width*img_height, k);
    training_label = zeros(468, 1, k);
    test_data = zeros(52, img_width*img_height, k);
    test_label = zeros(52, 1, k);
else 
    training_data = zeros(468, pca, k);
    training_label = zeros(468, 1, k);
    test_data = zeros(52, pca, k);
    test_label = zeros(52, 1, k);
end

%crange = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 500, 1000, 2000];
%srange = [0.1, 0.2, 0.3, 0.4, 0.5];
crange = [1];
srange = [1000];
coef0 = [0];

%for boxConstraints = crange
for offset = coef0;

% loop through all k-folds
%j=1;
for j=1:k
%for j = 1
    
    training_data_tmp = X(:, training(c, j))';
    training_label(:, :, j) = l(:, training(c, j))';
    test_data_tmp = X(:, test(c, j))';
    test_label(:, :, j) = l(:, test(c, j))';
    training_size = size(training_data_tmp, 1);
    test_size = size(test_data_tmp, 1);
    
    if PCA_FLAG == 1
        A = (training_data_tmp'-repmat(mean(training_data_tmp', 2), [1, training_size]));
        B = (test_data_tmp'-repmat(mean(test_data_tmp', 2), [1, test_size]));
        S_alternative = A' * A / training_size; % data covariance matrix using 1/N*At*A
        [V_alternative, D_alternative] = eig(S_alternative); % calculate V as the eigenvectors and D as eigenvalues (in D's diagonal)
        V_alternative = A * V_alternative;  % using 1/N*At*A gives same eigenvalues, and V = A*eigenvectors when using 1/N*At*A
        
        %-Reconstruct training data-
        %Normalize eigenvectors, 
        %vectors are fliped because it was ordered ascendingly
        VNormalizedFlip = fliplr(normc(V_alternative));
        eigenvectorChosen = VNormalizedFlip(:, 1:pca);
        %High-dimentional data projects to low-dimention 
        training_data_tmp = A' * eigenvectorChosen;
        test_data_tmp = B' * eigenvectorChosen;
    end
    if LDA_FLAG == 1
        A = (training_data_tmp'-repmat(mean(training_data_tmp', 2), [1, training_size]));
        B = (test_data_tmp'-repmat(mean(test_data_tmp', 2), [1, test_size]));
        LDAModel = fitcdiscr(A', training_label(:,:,j));
        
        
%         cMean = zeros(size(training_data_tmp, 2),size(training_data_tmp, 2));
%         Sb = zeros(size(training_data_tmp, 2),size(training_data_tmp, 2));
%         Sw = zeros(size(training_data_tmp, 2),size(training_data_tmp, 2));
%         
%         pcaMean = mean(training_data_tmp,1)';
%         
%         for i = 1:label_number
%             cMean = mean(training_data_tmp(9*i-8:9*i, :),1)';
%             Sb = Sb + (cMean-pcaMean)*(cMean-pcaMean)';
%         end
%         
%         Sb = 9*Sb;
%         
%         for i = 1:label_number
%             cMean = mean(training_data_tmp(9*i-8:9*i, :),1)';
%             for j = 9*i-(9-1):9*i
%                 Sw = Sw + (training_data_tmp(j, :)'-cMean)*(training_data_tmp(j, :)'-cMean)';
%             end
%         end
%         
        % Obtaining Fisher eigenvectors and eigenvalues
        [Vf, Df] = eig(LDAModel.BetweenSigma,LDAModel.Sigma);
        
        % Calculating weights
        Df = fliplr(diag(Df));
        Vf = fliplr(Vf);
        
        % Calculating fisher weights
        W1f = A'*Vf;
        W2f = B'*Vf;
        
        training_data_tmp = W1f;
        test_data_tmp = W2f;

    end
    training_data(:,:,j) = training_data_tmp;
    test_data(:,:,j) = test_data_tmp;
    
end

% loop through all k-folds
%j=1;

t_begin = cputime;
for j=1:k
%for j=1
    
    prob = [];
    
    for i=1:label_number
        training_label_normalised = double(training_label(:,:,j)==i);
        model = fitcsvm(training_data(:,:,j), training_label_normalised,...
            'Standardize',true, 'KernelScale', 300, 'KernelFunction',...
            'rbf', 'BoxConstraint', 5);
%         model = fitcsvm(training_data(:,:,j), training_label_normalised,...
%             'Standardize',true,'KernelFunction','polynomial','PolynomialOrder', 3, 'KernelScale',...
%             scale, 'BoxConstraint', boxConstraints);
        [~, p] = predict(model, test_data(:,:,j));
        prob = [prob, p(:, 2)];
    end
    
    % predict the class with the highest probability
    [~,pred] = max(prob,[],2);
    accuracy(j) = sum(pred == test_label(:,:,j)) ./ numel(test_label(:,:,j));    %# accuracy
    conf_matrix = conf_matrix + confusionmat(test_label(:,:,j), pred);     %# confusion matrix
    
end

t_OAA(index) = cputime - t_begin;

accuracy_overall(index) = sum (accuracy)/k;
% figure;
% imagesc(conf_matrix)

index = index + 1;
end
end