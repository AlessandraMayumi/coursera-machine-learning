function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30
% try all possible pairs of values for C and sigma (e.g., C = 0.3 and sigma = 0.1)
% 8 2 = 64 different models

##size(X) = 211,2
##size(y) = 211,1
##size(Xval) = 200,2
##size(yval) = 200,1

e = 1000;

Cvalues = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigmavalues = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
for i = 1:8
  for j = 1:8

    Ctemp = Cvalues(i);
    sigmatemp = sigmavalues(j);
    
    x1 = [1 2 1]; x2 = [0 4 -1];
    model = svmTrain(Xval, yval, Ctemp, @(x1, x2) gaussianKernel(x1, x2, sigmatemp));

##    x1 = Xval(:,1);
##    x2 = Xval(:,2);
##    model = svmTrain(Xval, yval, Ctemp, @(x1, x2) gaussianKernel(x1, x2, sigmatemp));
    
    predictions = svmPredict(model, Xval);

    errtemp = mean(double(predictions ~= yval));
    
    if e > errtemp 
      e = errtemp;
      C = Ctemp;
      sigma = sigmatemp;
    endif
    
  endfor
endfor

% =========================================================================
C 
sigma
end
