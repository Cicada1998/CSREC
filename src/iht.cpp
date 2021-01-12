#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
// [[Rcpp::export]]
arma::vec iht(arma::vec y,arma::mat A){
  //measurement matrix,x-m by n;m<<n,y-m dim,
  int m= A.n_rows; 
  int n= A.n_cols;
  double u=0.5;
  int k = floor(m/4); 
  arma::vec x(n, fill::zeros),x1,x2;
  arma::uvec index,index0;
  arma::uvec index1 = regspace<uvec>(k,n-1);
  for(int i = 0; i < k; i++){
    x1 = A.t() * (y - A * x);
    x2 = x + u *x1;
    index = sort_index(abs(x2),"descend");
    index0 = index(index1);
    x2(index0).zeros();
    x = x2;
  }
  return x;
  
}
