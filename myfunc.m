function [var2,var3] = myfunc(vec_x)
% You can add a description
% if you want
var1 = length(vec_x);
var2 = mysubfunc(vec_x, var1);
if var2 > var1
var3 = var2 * var1;
else
var3 = var2 / var1;

end