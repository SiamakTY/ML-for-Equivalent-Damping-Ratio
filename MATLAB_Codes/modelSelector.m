
function [selected_model] = modelSelector(jj,repeat_no,results_matrix,all_models, Xtest, means, Ytest, ek)

disp(['Selecting the most successful model for ', ek, ' ...'])
algModels_no = floor(repeat_no/4);
R2Models_no = floor(algModels_no/2) ;
RMSEModels_no = algModels_no - R2Models_no ;
[R2_select, indexR2] = maxk(results_matrix(7,1:1:end), R2Models_no) ;
[RMSE_select, indexRMSE]  = mink(results_matrix(2,1:1:end), RMSEModels_no) ;
for ii = 1:1:R2Models_no
    algModels{ii,1} = all_models{indexR2(1,ii),1} ;          
end
for ii = (R2Models_no + 1):1:algModels_no
    algModels{ii,1} = all_models{indexRMSE(1,(ii - R2Models_no)),1} ;          
end
R2s = zeros(repeat_no, algModels_no) ;
RMSEs = zeros(repeat_no, algModels_no) ; 
for ii = 1:1:algModels_no
    for kk = 1:1:repeat_no
        model = algModels{ii,1} ;
        features = Xtest(1:1:end,1:1:end,kk) ;
        Yp = testGPR(model, Xtest(1:1:end,1:1:end,kk)) ;
        Yp = exp(Yp + means(1,1,kk)) ;
        RESULTSGPR = assessment(Ytest(:,1,kk), Yp, 'regress') ;
        R2s(kk,ii) = RESULTSGPR.R2 ; RMSEs(kk,ii) = RESULTSGPR.RMSE ;
    end
end
R2s_mean = zeros(1,algModels_no) ;
RMSEs_mean = zeros(1,algModels_no) ;
for ii = 1:1:algModels_no
    R2s_mean(1,ii) = mean(R2s(1:1:end,ii)) ;
    RMSEs_mean(1,ii) = mean(RMSEs(1:1:end,ii)) ; 
end
if jj == 1
    [max_R2, I] = max(R2s_mean) ;
    selected_model = algModels{I(1),1} ;
elseif jj == 2
    [min_RMSE, I] = min(RMSEs_mean) ;
     selected_model = algModels{I(1),1} ;
end
disp(['The most successful model for ', ek, ' is selected:'])
