#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
// [[Rcpp::export]]
Rcpp::List ista(arma::vec y,arma::mat A,int lambda,int itr){    
  int n = A.n_cols;
  arma::vec x(n, fill::ones),c_k;
  //power iteration to get A.t()*A max eigenvalue,stepsize = 1/L
  arma::mat B = A.t()*A;
  for(int j = 0; j < 30; j++)
  {
    x = normalise(B.t()*B*x);
  }
  arma::vec v1 = x/norm(x,2);
  double sigma1 = norm(B*v1, 2);
  
  double t_max = 1/sigma1;
  
  double tk = 0.95 * t_max;
 
  arma::vec x_k = A.t()*y;//initial
  double alp = lambda * tk;
  arma::vec S(n,fill::ones);S = S*alp;
  arma::vec err(itr+1,fill::zeros);
  
  for(int i = 0; i < itr; i++){
    c_k = x_k - 2 * tk * A.t() * (A * x_k - y);
    arma::vec shir = abs(c_k)-S;
    shir.elem(find(shir<0)).zeros();//soft-thresholding
    x_k = (shir) % sign(c_k);
    err(i+1) = norm(A * x_k - y,2);
    
    if(abs(err(i+1) - err(i)) < 0.000001  )
      break;
  }
  for(int i = 0; i < itr; i++){
    err(i) = err(i+1);
  }
  //x_k.elem(find(abs(x_k)<0.1)).zeros();
  return Rcpp::List::create(Rcpp::Named("x_k") = x_k,
                            Rcpp::Named("err") = err
  );
}
