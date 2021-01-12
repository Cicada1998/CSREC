#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;
// [[Rcpp::export]]
Rcpp::List fista(arma::vec y,arma::mat A,int lambda,int itr){ 
  int n = A.n_cols;
  arma::vec x(n, fill::ones),c_k;
  ////power iteration
  arma::mat B = A.t()*A;
  for(int j = 0; j < 30; j++)
  {
    x = normalise(B.t()*B*x);
  }
  arma::vec v1 = x/norm(x,2);
  double sigma1 = norm(B*v1, 2);
  double Li = 1/sigma1;
  
  double tk = 1;
  
  arma::vec xhk(n, fill::zeros);//initial
  arma::vec yk = xhk;
  double alp = lambda * tk;
  arma::vec S(n,fill::ones);S = S*alp;
  arma::vec err(itr+1,fill::zeros);
  
  for(int i = 0; i < itr; i++){
    c_k = yk - 2 * Li * A.t() * (A * yk - y);
    arma::vec shir = abs(c_k)-S;
    shir.elem(find(shir<0)).zeros();//soft-thresholding
    arma::vec xhk1 = (shir) % sign(c_k);
    double tk1 = 0.5 + 0.5*sqrt(1 + 4*tk*tk);
    double tt = (tk - 1)/tk1;
    yk = xhk1 - tt*(xhk1 - xhk);
    tk = tk1;
    xhk = xhk1;
    err(i+1) = norm(A * xhk - y,2);
    if(abs(err(i+1) - err(i)) < 0.00000001)
      err = err(1,itr);
    break;
  }
  for(int i = 0; i < itr; i++){
    err(i) = err(i+1);
  }
  //xhk.elem(find(abs(xhk)<0.1)).zeros();
  return Rcpp::List::create(Rcpp::Named("xhk") = xhk,
                            Rcpp::Named("err") = err);
}
