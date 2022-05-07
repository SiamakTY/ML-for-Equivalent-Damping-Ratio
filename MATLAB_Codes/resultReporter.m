function [] = resultReporter(results_matrix, Yp, Ytest, ek)

n = size(Yp, 3) ;
m = size(Yp,1) ;
MAPEs = zeros(1,n) ; 
for ii = 1:1:n
    sum = 0 ;
    for jj = 1:1:size(Yp,1)
       sum = sum + abs(Ytest(jj,1,ii) - Yp(jj,1,ii))/abs(Ytest(jj,1,ii)) ;
    end
    MAPEs(1,ii) = (100/m)*sum ;
end

str = ['Mean ME value for ', ek,' is: ',num2str(mean(results_matrix(1,1:1:end)))] ; disp(str) 
str = ['Mean RMSE value for ', ek,' is: ',num2str(mean(results_matrix(2,1:1:end)))] ; disp(str) 
str = ['Mean RELRMSE value for ', ek,' is: ',num2str(mean(results_matrix(3,1:1:end)))] ; disp(str) 
str = ['Mean MAE value for ', ek,' is: ',num2str(mean(results_matrix(4,1:1:end)))] ; disp(str) 
str = ['Mean RE value for ', ek,' is: ',num2str(mean(results_matrix(5,1:1:end)))] ; disp(str) 
str = ['Mean R value for ', ek,' is: ',num2str(mean(results_matrix(6,1:1:end)))] ; disp(str) 
str = ['Mean R2 value for ', ek,' is: ',num2str(mean(results_matrix(7,1:1:end)))] ; disp(str) 
str = ['Mean MAPE value for ', ek,' is: ',num2str(mean(MAPEs(1,1:1:end))), '%'] ; disp(str) 

if strcmp(ek,'EDR') ~= 1
disp(repmat('+',1,35)) 
end