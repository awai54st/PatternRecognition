function [reconImages, reconTestImages] = faceRecog(A, testImageA, numOfEigenvector, VNormalizedFlip, imageMean, training_size, test_size, test_data)
eigenvectorChosen = VNormalizedFlip(:, 1:numOfEigenvector);

%High-dimentional data projects to low-dimention 
eigenProjection = eigenvectorChosen' * A;

%each face could be resconstructed as a linear combination of  
%the best numOfEigenvector eigenvectors
imageMeanMat = repmat(imageMean, 1, training_size);
reconImages = imageMeanMat + eigenvectorChosen * eigenProjection;

%-Reconstruct Test data

testImageMeanMat = repmat(imageMean, 1, test_size);

testImageEigenProjection = eigenvectorChosen' * testImageA;
reconTestImages = testImageMeanMat + eigenvectorChosen * testImageEigenProjection;


end