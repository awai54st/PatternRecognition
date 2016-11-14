clc
close all

%load face data
load face.mat

%% Q1

TrainNum=52*9;
TestNum=52;

%10-fold crossvalidation
%10 items in each class and 9 data into training set, 1 into test set, same
%as leave-one-out in this case
k=10;                               %Define ratio of partition, k is the proportion sorted into test set
c = cvpartition(l,'Kfold',k);       %Create partition object

%Demonstrate with 1st set
TestIdx=test(c,1);                    %Create index list for test set
TrainingIdx=training(c,1);            %Index list for training set
TestDataTemp=X(:,TestIdx);              
TrainDataTemp=X(:,TrainingIdx);

%find mean face image for training data
mean_image = mean(TrainDataTemp,2);

TestData=TestDataTemp-repmat(mean_image,1,TestNum);  %Obtain test data by subtracting from mean face
TrainData=TrainDataTemp-repmat(mean_image,1,TrainNum);%Obtain train data

%Compute the covariance matrix s
PhiTrain=TrainData;
s = (PhiTrain*PhiTrain')/TrainNum;

%Find the eigenvectors and eigenvalues of covariance matrix s
tic;
[V,D] = eig(s);
AATtime=toc;

[Y,I] = sort(diag(D),'descend');
Vnorm=normc(V);
%Show the magnitude of eigenvalues
figure;
subplot(1,2,1)
plot(Y)
xlabel('No. of eigenvalues');
ylabel('Magnetude of eigenvalues');
title('Eigenvalues in Decreasing Order');
subplot(1,2,2)
plot(Y)
axis([0 700 0 1300])
xlabel('No. of eigenvalues');
ylabel('Magnetude of eigenvalues');
title('Eigenvalues in Decreasing Order');

figure;
plot(Y)
axis([0 700 0 1300])
xlabel('No. of eigenvalues');
ylabel('Magnetude of eigenvalues');
title('Eigenvalues in Decreasing Order');

figure;
imagesc(reshape(mean_image,56,46));
title('mean face')
colormap gray

%Plot the three most significant eigenfaces
figure(1);
subplot(4,3,2)
imagesc(reshape(mean_image,56,46));
title('mean face')
subplot(4,3,4)
imagesc(reshape(Vnorm(:,I(1)),56,46));
title('Eigenface 1');
subplot(4,3,5)
imagesc(reshape(Vnorm(:,I(2)),56,46));
title('Eigenface 2');
subplot(4,3,6)
imagesc(reshape(Vnorm(:,I(3)),56,46));
title('Eigenface 3');
subplot(4,3,7)
imagesc(reshape(Vnorm(:,I(4)),56,46));
title('Eigenface 4');
subplot(4,3,8)
imagesc(reshape(Vnorm(:,I(5)),56,46));
title('Eigenface 5');
subplot(4,3,9)
imagesc(reshape(Vnorm(:,I(6)),56,46));
title('Eigenface 6');
subplot(4,3,10)
imagesc(reshape(Vnorm(:,I(465)),56,46));
title('Eigenface 465');
subplot(4,3,11)
imagesc(reshape(Vnorm(:,I(466)),56,46));
title('Eigenface 466');
subplot(4,3,12)
imagesc(reshape(Vnorm(:,I(467)),56,46));
title('Eigenface 467');
colormap gray
set(figure(1),'Position', [100,100,600,600]);
%maximum of 260 eigenvectors to be used according to figure 1.

figure;
subplot(1,3,1)
imagesc(reshape(Vnorm(:,I(5)),56,46));
subplot(1,3,2)
imagesc(reshape(Vnorm(:,I(5)),56,46));
subplot(1,3,3)
imagesc(reshape(Vnorm(:,I(5)),56,46));

%Find the suitable number of eigenvectors
cv=0;
i=0;
while cv<0.95*sum(Y)
    i=i+1;
    cv=cv+Y(i);
end


%% Q2

s2 = (PhiTrain'*PhiTrain)/TrainNum;

tic;
[V2,D2] = eig(s2);
ATAtime=toc;

[Y2,I2] = sort(diag(D2),'descend');
figure
plot(Y2)

%Dimension change
U2=PhiTrain*V2;
U2norm=normc(U2);
figure
subplot(2,3,1);
imagesc(reshape(U2(:,I2(1)),56,46));
subplot(2,3,2);
imagesc(reshape(U2(:,I2(2)),56,46));
subplot(2,3,3);
imagesc(reshape(U2(:,I2(3)),56,46));
subplot(2,3,4);
imagesc(reshape(U2(:,I2(4)),56,46));
subplot(2,3,5);
imagesc(reshape(U2(:,I2(5)),56,46));
subplot(2,3,6);
imagesc(reshape(U2(:,I2(6)),56,46));

colormap gray
%Eigenvalues are same and eigenvectors are closely related. u=Av, ppt p42
%The eigenvalues are not normalised, therefore after performing u=Av, the
%intensities of eigenvector varies, some becomes negative. See report. ppt
%p43

%% Q3

%Speficy the number of eigenvectors to select
EigenNum=468;                   
EigenVctSel=U2norm(:,I2(1:EigenNum)); 
%Projection of training data onto eigenface space
OmegaTrain = EigenVctSel'*PhiTrain; %[w1,w2...]
ReconImages = EigenVctSel*OmegaTrain;%[v1,v2...]*[w1,w2...]

%Plot the first image and its reconstruction from the training set
figure;
subplot(1,2,1)
imagesc(reshape(TrainDataTemp(:,1),56,46));
subplot(1,2,2)
recon = mean_image + ReconImages(:,1);
imagesc(reshape(recon,56,46));
colormap gray
title('testtest');

%Plot the second image and its reconstruction from the training set
figure;
subplot(1,2,1)
imagesc(reshape(TrainDataTemp(:,2),56,46));
subplot(1,2,2)
recon = mean_image + ReconImages(:,2);
imagesc(reshape(recon,56,46));
colormap gray

%Perform reconstruction on test set
PhiTest=TestData;
%Projection of test data onto eigenface space
OmegaTest = EigenVctSel'*PhiTest;
ReconImages = EigenVctSel*OmegaTest;

%Plot the first image in test set and its reconstruction
figure;
subplot(1,2,1)
imagesc(reshape(TestDataTemp(:,1),56,46));
subplot(1,2,2)
recon = mean_image + ReconImages(:,1);
imagesc(reshape(recon,56,46));
colormap gray

%% Q4

%Perform recognition on first test set (For demonstration)
%for i=1:TestNum
%    [ErrMatrix(1,i),ResultMatrix(1,i)]=min(sqrt(sum((repmat(OmegaTest(:,i),1,TrainNum)-OmegaTrain).^2)));
%end


%%The following is the alternative method. It is time-consuming using AA^T

%A complete training and testing process for k-fold crossvalidation

NNTimeArray=[];
NNAccuracyArray=[];
NNErrorArray=[];
for eigNum=129:129
    tic
    for fold = 1:10
        TestIdx=test(c,fold);                    %Create index list for test set
        TrainingIdx=training(c,fold);            %Index list for training set
        TestDataTemp=X(:,TestIdx);              
        TrainDataTemp=X(:,TrainingIdx);

        %find mean face image for training data
        mean_image = mean(TrainDataTemp,2);

        TestData=TestDataTemp-repmat(mean_image,1,TestNum);  %Obtain test data by subtracting from mean face
        TrainData=TrainDataTemp-repmat(mean_image,1,TrainNum);%Obtain train data

        %Compute the covariance matrix s
        PhiTrain=TrainData;
        s = (PhiTrain'*PhiTrain)/TrainNum;

        %Find the eigenvectors and eigenvalues of covariance matrix s
        [V,D] = eig(s);
        [Y,I] = sort(diag(D),'descend');

        %Speficy the number of eigenvectors to select
        EigenNum=eigNum;                   
        EigenVctSelV=V(:,I(1:EigenNum)); 

        %Normalise u
        EigenVctSelU=PhiTrain*EigenVctSelV;
        EigenVctSelUnorm=normc(EigenVctSelU);

        %Perform reconstruction on test set
        PhiTest=TestData;

        %Projection of Training data onto eigenface space
        OmegaTrain = EigenVctSelUnorm'*PhiTrain;
        OmegaTest = EigenVctSelUnorm'*PhiTest;

        %Modify result matrix for the remaining 9 tests
        for i=1:TestNum
            [NNErrMatrix(fold,i),NNResultMatrix(fold,i)]=min(sqrt(sum((repmat(OmegaTest(:,i),1,TrainNum)-OmegaTrain).^2)));
        end

        %The alternative method of using minimum reconstructed image error
    end
    NNtime=toc/10;
    
    %find mean error for all folds for a fixed number of eigenvectors
    NNErrorArray=[NNErrorArray,mean(mean(NNErrMatrix,1))];
    
    %Find recognition accuracy
    for i = 1 : TestNum
        NNLabelMatrix(:,i)=ceil(NNResultMatrix(:,i)/9);
    end
    NN=reshape(NNLabelMatrix,[520,1]);
    LBtemp=repmat([1:52],10,1);
    LB=reshape(LBtemp,[520,1]);
    NNAccuracy=size(NN(NN==LB),1)/520;
    NNTimeArray=[NNTimeArray,NNtime];
    NNAccuracyArray=[NNAccuracyArray,NNAccuracy];
end

figure;
plot(NNErrorArray);
title('Mean error v.s No.')

figure;
plot(NNTimeArray);
title('Accuracy v.s No.')

figure;
plot(NNAccuracyArray);


ErrMatrixClass=[];
ErrMatrixTrainClass=[];
NIPArray=[];
NIPTrainArray=[];
AMTimeArray=[];
AMAccuracyArray=[];
for eigNum=1:9
    tic
    for fold = 1 : 10

        for i=1:52
                EigenNumClass=eigNum;

                TestIdx=test(c,fold);                    %Create index list for test set
                TrainingIdx=training(c,fold);            %Index list for training set
                TestDataTemp=X(:,TestIdx);              
                TrainDataTemp=X(:,TrainingIdx);

                MeanImageClass = mean(TrainDataTemp(:,9*(i-1)+1:9*i),2);

                TestDataClass=TestDataTemp-repmat(MeanImageClass,1,52);  %Obtain test data by subtracting from mean face
                TrainDataClass=TrainDataTemp(:,9*(i-1)+1:9*i)-repmat(MeanImageClass,1,9);%Obtain train data
                PhiTrainClass=TrainDataClass;
                PhiTestClass=TestDataClass;

                sClass=PhiTrainClass'*PhiTrainClass;
                [VClass,IClass]=eig(sClass);
                [YClass,IClass] = sort(diag(IClass),'descend');
                EigenVctSelClassV=VClass(:,IClass(1:EigenNumClass)); 

                EigenVctSelClassU=PhiTrainClass*EigenVctSelClassV;

                EigenVctSelClassUnorm=normc(EigenVctSelClassU);

                OmegaTestClass = EigenVctSelClassUnorm'*PhiTestClass;
                OmegaTrainClass = EigenVctSelClassUnorm'*PhiTrainClass;
                ReconImagesClass = EigenVctSelClassUnorm*OmegaTestClass;
                ReconImagesTrainClass = EigenVctSelClassUnorm*OmegaTrainClass;

                ErrMatrixClass=[ErrMatrixClass;sqrt(sum((ReconImagesClass-PhiTestClass).^2))];
                ErrMatrixTrainClass=[ErrMatrixTrainClass;sqrt(sum((ReconImagesTrainClass-PhiTrainClass).^2))];
        end
        
        [NIP(fold,:),AMLabelMatrix(fold,:)]=min(ErrMatrixClass);
        [NIPTrain(fold,:),AMLabelTrainMatrix(fold,:)]=min(ErrMatrixTrainClass);
        ErrMatrixClass=[];
        ErrMatrixTrainClass=[]
    end
    NIPArray=[NIPArray,mean(mean(NIP,1),2)];
    NIPTrainArray=[NIPTrainArray,mean(mean(NIPTrain,1),2)];
    AMtime=toc/10;
    AM=reshape(AMLabelMatrix,[520,1]);
    LBtemp=repmat([1:52],10,1);
    LB=reshape(LBtemp,[520,1]);
    AMAccuracy=size(AM(AM==LB),1)/520;
    AMTimeArray=[AMTimeArray,AMtime];
    AMAccuracyArray=[AMAccuracyArray,AMAccuracy];
end

figure;
subplot(121)
plot(NIPArray)
title('Recon Error on Testing Data.');
xlabel('No. of Eigenvector')
ylabel('Reconstruction Error')
grid on;

subplot(122)
NIPTrainArray(end)=0;
plot(NIPTrainArray)
title('Recon Error on Training Data');
xlabel('No. of Eigenvector')
ylabel('Reconstruction Error')
grid on;

figure;
plot(AMTimeArray);
figure;
plot(AMAccuracyArray);

AM=reshape(AMLabelMatrix,[520,1]);
NN=reshape(NNLabelMatrix,[520,1]);
%LBtemp=repmat([1:52],10,1);
%LB=reshape(LBtemp,[520,1]);

AMAccuracy=size(AM(AM==LB),1)/520
NNAccuracy=size(NN(NN==LB),1)/520

cfmAM=confusionmat(LB,AM);
hmo=HeatMap(cfmAM,'Colormap','redbluecmap','RowLabels',[1:52],'ColumnLabels',[1:52]);
addTitle(hmo,'Confusion Matrix for the Alternative Method');
plot(hmo)

ctmNN=confusionmat(NN,LB);
hmo=HeatMap(ctmNN,'Colormap','redbluecmap','RowLabels',[1:52],'ColumnLabels',[1:52]);
addTitle(hmo,'Confusion Matrix for the Nearest Neighbor Method');
plot(hmo)
