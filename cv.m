%%%%%%%% Machine Learning for Computer Vision   %%%%%%%%
%%%%%%%% Matlab Tutorial                        %%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% COMPUTER VISION %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% clear all, close all, clc
clear all
close all
clc

%% Read image
I=imread('imperial.jpg');

%% Show image
imshow(I)

%% Size?
[height, width, dim] = size(I);

%% RGB - 3 channels - Show Red Channel only
I_red(:,:,1)=I(:,:,1);
I_red(:,:,2)=zeros(height,width);
I_red(:,:,3)=zeros(height,width);
figure(1); imshow(I_red);

%% Show green channel only
I_green(:,:,1)=uint8(zeros(height,width));
I_green(:,:,2)=uint8(I(:,:,2));
I_green(:,:,3)=uint8(zeros(height,width));
figure(2); imshow(I_green);

%% Show blue channel only
I_blue(:,:,1)=uint8(zeros(height,width));
I_blue(:,:,2)=uint8(zeros(height,width));
I_blue(:,:,3)=uint8(I(:,:,3));
figure(3); imshow(I_blue);

%% RGB to gray...
gray = sum(I,3)/3;
figure(4); imshow(gray);

%% or maybe...
I_gray = uint8(sum(I,3)/3);
figure(5); imshow(I_gray);

%% rgb2gray - I_gray_matlab=0.299Red + 0.587Green + 0.114Blue
I_gray_matlab = rgb2gray(I);
figure(6); imshow(I_gray_matlab);

%% Patches.... 100x100 pixels
patches = 10;
[height, width, dim] = size(I_gray);
patch_height_idx = randperm(height, patches);
patch_width_idx = randperm(width, patches);
%patch_size_idx = randperm(100, patches);
patch_size = 20;

result_matrix = zeros(patch_size*patch_size,patches,'uint8');

for i=1:patches,
    patch_height = patch_height_idx(i);
    patch_width = patch_width_idx(i);
    
    
    % Check if it exceeds the image limits
    if patch_height + patch_size > height
        patch_height = height - patch_size;
    end
    if patch_width + patch_size > width
        patch_width = width - patch_size;
    end
    
    % Crop random patch, save it, resize to 100x100 and save it to results
    patch = imcrop(I_gray, [patch_width, patch_height, patch_size-1, patch_size-1]);
    figure; imshow(patch)
    
    result_matrix(:,i) = reshape(patch, patch_size*patch_size, 1);
end

%% Mean?
my_mean=[(sum(uint32(result_matrix),2))]./patches;
img_my_mean = reshape(uint8(my_mean), patch_size, patch_size);
figure; imshow(img_my_mean);

%% Matlab mean
matlab_mean=mean(uint32(result_matrix),2);
img_matlab_mean = reshape(uint8(matlab_mean), patch_size, patch_size);
figure; imshow(img_matlab_mean);

%% Test accuracy
test=abs(matlab_mean - my_mean);
max(test)

%% Covariance Math type    S = 1/(N-1)?(x-?)(x-?)' for all x from 1 to N
% which is equivalent
%(Linear Algebra) S = 1/(N-1)(result_matrix-my_mean_over_patches)(result_matrix-my_mean_over_patches)'
my_mean_over_patches = my_mean*ones(1,patches);
mean_centred = double(result_matrix) - my_mean_over_patches;
my_cov = (mean_centred*mean_centred')./(patches-1);
img_my_cov = uint8(my_cov / max(my_cov(:)) * 255);
figure; imshow(img_my_cov);

%% Covariance Matlab
matlab_cov = cov(double(result_matrix).');
img_matlab_cov = uint8(matlab_cov / max(matlab_cov(:)) * 255);
figure; imshow(img_matlab_cov);

%% Test
test=abs(matlab_cov - my_cov);
max(max(test))

%% Lecture 1 - PCA
[eig_vec, eig_val] = eig(my_cov);

%% After classification - confusion matrix
g_t = [1 0 0 0 1 0 0 0 0 0; % ground truth
    0 1 0 0 0 1 1 0 1 0;
    0 0 1 0 0 0 0 1 0 1;
    0 0 0 1 0 0 0 0 0 0];

e_t = [1 1 0 0 1 0 0 0 0 0; % your predictions
    0 0 1 0 0 1 0 0 0 1;
    0 0 0 0 0 0 1 1 0 0;
    0 0 0 1 0 0 0 0 1 0];

plotconfusion(g_t,e_t)

%% Structs...
student(1).Name = 'Sarah';
student(1).Year = 3;
student(1).Course = 'Machine Learning for Computer Vision';
student(1).Final_Marks = [85 82 78;  % courseworks
                          80 81 80]; % interviews

student(2).Name = 'Mark';
student(2).Year = 3;
student(2).Course = 'Machine Learning for Computer Vision';
student(2).Final_Marks = [45 52 58;  % courseworks
                          39 41 40]; % interviews


student(3).Name = 'Rigas';
student(3).Year = 3;
student(3).Course = 'Machine Learning for Computer Vision';
student(3).Final_Marks = [99 99 45;  % courseworks
                          20 80 99]; % interviews


figure; bar(student(1).Final_Marks); title(['Coursework and Interview marks for ', student(1).Name]);
figure; bar(student(2).Final_Marks); title(['Coursework and Interview marks for ', student(2).Name]);
figure; bar(student(3).Final_Marks); title(['Coursework and Interview marks for ', student(3).Name]);


figure; subplot(1,3,1); bar(student(1).Final_Marks); title(['Coursework and Interview marks for ', student(1).Name]);
subplot(1,3,2); bar(student(2).Final_Marks); title(['Coursework and Interview marks for ', student(2).Name]);
subplot(1,3,3); bar(student(3).Final_Marks); title(['Coursework and Interview marks for ', student(3).Name]);

average_courseworks_marks_interviews = student(1).Final_Marks;
average_courseworks_marks_interviews = [average_courseworks_marks_interviews student(2:end).Final_Marks];

mean(average_courseworks_marks_interviews,2)