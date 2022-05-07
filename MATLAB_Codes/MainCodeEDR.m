%%
% This code is developed for estimating the Equivalent Damping Ratio (EDR) of rectangular 
% RC shear walls at displacements corresponding to 1% lateral drift ratio using a 
% machine learning algorithm based on Gaussian Process Regression (GPR)
% method. The equivalent damping ratio value of each specimen is estimated 
% according to Jacobsen (1930) and (1960) method.
%
% f0 : lateral force @ 1% lateral drift ratio
% ED: cyclic dissipated energy @ 1% lateral drift ratio
%
% This code utilizes an algorithm composed of 3 stages:
% Stage 1: A number of models are constructed, trained and validated for
%          evaluating f0 & ED values
% Stage 2: The two most successful models for identifying f0 & ED values   
%          regarding mean R2 or mean RELRMSE are selected.
% Stage 3: EDR values are evaluated using the estimated f0 & ED values by 
%          the two selected models and according to Jacobsen (1930) method. 
%           
% Number of specimens: 161 (rectangular RC shear walls subjected to quasi-static reversed cyclic loading)   
% Number of features to f0 @ 1% lateral drift: 10
% Number of features to identify  ED @ 1% lateral drift: 11
% Split rate is adjustable (85% by default)
% Log transformation of the input & output variables is used.
%
% List of functions that the code needs to run:
% 1- assessment.m; 2- covNoise.m; 3- covSEardj.m; 4- covSum.m
% 5- gpr.m; 6- minimize.m; 7- modelSelector.m; 8- resultReporter.m
% 9- sq_dist.m; 10- testGPR.m; 11- trainGPR.m
%
% List of data that the code needs to run:
% 1- EDR.mat; 2- logED_output.mat; 3- u0.mat; 4- logf0_output.mat
% 5- logED_input.mat; 6- logf0_input.mat
% Record of revisions:
%     Date             Programmer               Description of change
%  ===========    =====================         =====================
%  07.05.2022     Siamak TAHAEI YAGHOUBI            Original code

%% Setup
clear;clc;close all;
cd 'YourDirectory\' ;
% Set the number of times regression analysis will be repeated.
repeat_no = 100 ; 
%% Load data:
u0 = load('u0.mat').disp; % Maximum lateral displacement of the hysteretic loop in the vicinity of 1.0% drift ratio from the database
EDRs = load('EDR.mat').EDRs ; %Equivalent damping ratio values from the database

%% Constructing, training and validating models for the intermidiate outputs: 
% jj = 1 ; The lateral force (f0)
% jj = 2 ; The cyclic dissipated energy (ED)
for jj = 1:1:2 

    % Loading data for the intermidiate outputs (f0 & ED)
    if jj == 1
        X0 = load('logf0_input.mat') ; Y0 = load('logf0_output.mat') ; % Natural logarithm of lateral force (f0) from the database
        X = X0.f0_input; Y = Y0.f0_output ;
        obs_type = 'lateral force (f0)' ;
        ek = 'f0' ;
    elseif jj == 2
        X0 = load('logED_input.mat') ; Y0 = load('logED_output.mat') ; % Natural logarithm of the cyclic dissipated energy (ED) from the database
        X = X0.ED_input; Y = Y0.ED_output ;
        obs_type = 'cyclic dissipated energy (ED)' ;
        ek = 'ED' ;
    end

    %% Split training-testing data
    rate = 0.85; 
    for ii = 1:1:repeat_no
        [n d] = size(X);                 % Samples x bands
        if jj == 1
            if ii == 1
                r_matrix = zeros(repeat_no, n) ; % Each row of the r_matrix includes specimen numbers shuffled randomly. Its column no. equals specimen no. & its row no. equals repeat_no. 
            end
            r = randperm(n);                 % Random index
            r_matrix(ii,1:1:end) = r(1, 1:1:end) ;
            ntrain = round(rate*n);          % No. of training samples
            
            all_u0Test(:,:,ii) = u0(r_matrix(ii,(ntrain+1):1:end),:) ; % u0 values of this 3 dimensional array will be used for calculating equivalent damping ratio according to Jacobsen (1930) equation. 
            all_EDRsTest(:,:,ii) = EDRs(r_matrix(ii,(ntrain+1):1:end),:) ; % Equivalent damping ratio values of this 3 dimensional array will be used for varification of the predicted damping ratio values.
         end
        Xtrain = X(r_matrix(ii,1:1:ntrain),:);       % Training set (jj=1 => f0; jj=2 => ED)
        Ytrain = Y(r_matrix(ii,1:1:ntrain),:);       % Observed training variable (jj=1 => f0; jj=2 => ED)
        Xtest  = X(r_matrix(ii,(ntrain+1):1:end),:);   % Test set (jj=1 => f0; jj=2 => ED)
        Ytest  = Y(r_matrix(ii,(ntrain+1):1:end),:);   % Observed test variable (jj=1 => f0; jj=2 => ED)
            
        [ntest do] = size(Ytest);

        if ii == 1
            eval(['all_Xtest' num2str(jj) '=' 'zeros(size(Xtest,1), size(Xtest,2), repeat_no) ;']) ; % all_Xtest1: Test f0 data for all analysis repeats; all_Xtest2: Test ED data for all analysis repeats
            eval(['all_Ytest' num2str(jj) '=' 'zeros(size(Ytest,1), size(Ytest,2), repeat_no) ;']) ; % all_Ytest1: Observed test f0 data for all analysis repeats; all_Ytest2: Observed test ED data for all analysis repeats
            eval(['all_Yp' num2str(jj) '=' 'zeros(size(Ytest,1), size(Ytest,2), repeat_no) ;']) ; % all_Yp1: Predicted f0 values for all analysis repeats; all_Yp2: Predicted ED values for all analysis repeats
            eval(['all_my' num2str(jj) '=' 'zeros(1, 1, repeat_no) ;']) ; % all_my1: Mean value of observed f0 training data for all analysis repeats; all_my2: Mean value of observed ED training data for all analysis repeats
        end
        eval(['all_Xtest' num2str(jj) '(1:1:end, 1:1:end, ii)' '=' 'Xtest ;']) ;
        eval(['all_Ytest' num2str(jj) '(1:1:end, 1:1:end, ii)' '=' 'Ytest ;']) ;
        
        %% Remove the mean of Y for training only
         my      = mean(Ytrain) ;
         Ytrain  = Ytrain - repmat(my,ntrain,1);
         eval(['all_my' num2str(jj) '(1, 1, ii)' '= my ;' ])

        %% Train, test and save aLL models for f0 & ED
        fprintf(['Training ' 'GPR' ' for repeat No.' num2str(ii) ' from ', num2str(repeat_no),' repeats \n'])
        eval(['model = train' 'GPR' '(Xtrain,Ytrain);']) ; % Train the model
        eval(['Yp = test' 'GPR' '(model,Xtest);']);       % Test the model  
        eval(['allModels' num2str(jj) '{ii,1}' ' = ' 'model ;']) ; % Save the constructed model to allModels1 (for f0) or allModels2 for (ED) 

        Yp = Yp + repmat(my,ntest,1);
        Yp = exp(Yp) ;
        Ytest = exp(Ytest) ;

        disp(['Analyses for ', obs_type])
        RESULTS = assessment(Ytest, Yp, 'regress')  % Display analysis results

        if ii == 1
            eval(['results_matrix' num2str(jj) '=' 'zeros(7,repeat_no) ;']) ; % results_matrix1: for saving regression analysis results of f0; results_matrix2: for saving regression analysis results of ED
        end

        eval(['results_matrix' num2str(jj) '(1,ii) = ' 'RESULTS.ME ;']) ;
        eval(['results_matrix' num2str(jj) '(2,ii) = ' 'RESULTS.RMSE ;']) ;
        eval(['results_matrix' num2str(jj) '(3,ii) = ' 'RESULTS.RELRMSE ;']) ;
        eval(['results_matrix' num2str(jj) '(4,ii) = ' 'RESULTS.MAE ;']) ;
        eval(['results_matrix' num2str(jj) '(5,ii) = ' 'RESULTS.RE ;']) ;
        eval(['results_matrix' num2str(jj) '(6,ii) = ' 'RESULTS.R ;']) ;
        eval(['results_matrix' num2str(jj) '(7,ii) = ' 'RESULTS.R2 ;']) ;
        disp(repmat('-',1,30))
        eval(['all_Ytest' num2str(jj) '(1:1:end, 1:1:end, ii)' '=' 'Ytest ;']) ;
        eval(['all_Yp' num2str(jj) '(1:1:end, 1:1:end, ii)' '=' 'Yp ;']) ;
        
    end    
    %% Selecting the most successful models for predicting f0 & ED 
    if jj == 1
        f0_model = modelSelector(jj,repeat_no,results_matrix1,allModels1, all_Xtest1, all_my1, all_Ytest1, ek) 
    elseif jj == 2
        ED_model = modelSelector(jj,repeat_no,results_matrix2,allModels2, all_Xtest2, all_my2, all_Ytest2, ek) 
    end
    disp(repmat('-',1,30)) 
end
%% Predicting equivalent damping ratio values
results_matrixEDR = zeros(7,repeat_no) ; % A temporary matrix for saving analysis results
disp ('Analyses for Equivalent Damping Ratio')
for ii = 1:1:repeat_no 
    
    % Predicting f0 values for test data using the selected f0 model
    Yp_f0M = testGPR(f0_model, all_Xtest1(1:1:end,1:1:end, ii)); 
    Yp_f0M = Yp_f0M + (all_my1(1,1,ii)) ;
    
    % Predicting ED values for test data using the selected ED model
    Yp_EDM = testGPR(ED_model, all_Xtest2(1:1:end,1:1:end, ii));
    Yp_EDM = Yp_EDM + (all_my2(1,1,ii)) ;
    
    u0Test = log(all_u0Test(1:1:end,1,ii)) ; % logarithm of u0 for using in Jacobsen (1930) equation
    
    % Calculating equivalent damping ratio by Jacobsen (1930) equation and using the predicted f0 & ED values 
    Yp_EDR = exp(log(1/(2*pi)) + Yp_EDM - u0Test - Yp_f0M) ;
    all_Yp_EDR(:, :, ii) = Yp_EDR ;
    
    disp(['Test output No.' num2str(ii)])
    
    RESULTS = assessment(all_EDRsTest(:, :, ii), Yp_EDR, 'regress')  
    results_matrixEDR(1,ii) = RESULTS.ME ;
    results_matrixEDR(2,ii) = RESULTS.RMSE ;
    results_matrixEDR(3,ii) = RESULTS.RELRMSE ;
    results_matrixEDR(4,ii) = RESULTS.MAE ;
    results_matrixEDR(5,ii) = RESULTS.RE ;
    results_matrixEDR(6,ii) = RESULTS.R ;
    results_matrixEDR(7,ii) = RESULTS.R2 ;
   
end
%% Reporting analysis results
disp(repmat('+',1,30)) 
ek = 'f0' ;  
resultReporter(results_matrix1, all_Yp1, all_Ytest1, ek)
ek = 'ED' ; 
resultReporter(results_matrix2, all_Yp2, all_Ytest2, ek)
ek = 'EDR' ;
resultReporter(results_matrixEDR, all_Yp_EDR, all_EDRsTest, ek)















