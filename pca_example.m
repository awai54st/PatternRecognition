function [ mean_face, eigen_faces, w ] = pca( training_set, M )
%PCA Summary of this function goes here
%   Detailed explanation goes here

N = size(training_set,2);

%%%%%%%%%%%% Compute Average Face %%%%%%%%%%%%
mean_face = mean(training_set,2);

%%%%%%%%%%%% Subtract Mean Face from Images %%%%%%%%%%%%

mean_face_mat = repmat(mean_face, [1,N]);
A = training_set-mean_face_mat;

%%%%%%%%%%%% Compute Covariance Matrix, S %%%%%%%%%%%%
S = A*A'/N;

%%%%%%%%%%%% Compute Eigenvectors of S %%%%%%%%%%%%
[u,lambda] = eig(S);
eigen_values = diag(lambda);
[temp, eigen_index] = sort(eigen_values, 'descend');

%%%%%%%%%%%% Best M Eigenvectors %%%%%%%%%%%%
eigen_faces = u(:,eigen_index(1:M));

%%%%%%%%%%%% Representing Faces onto Eigenfaces %%%%%%%%%%%%
w = (A'*eigen_faces)';

end

