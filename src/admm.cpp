#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace arma;

// [[Rcpp::export]]
Rcpp::List admm_lasso(arma::mat A,arma::vec b,double lambda,
                      int maxiter,double rho){
  
  int n = A.n_cols;
  
  arma::vec x(n,fill::zeros);
  arma::vec z(n,fill::zeros);
  arma::vec u(n,fill::zeros);
  arma::mat I;I.eye(n,n);
  
  arma::vec r_norm(maxiter,fill::zeros);
  arma::vec eps_pri(maxiter,fill::zeros);
  double sqrtn = std::sqrt(static_cast<float>(n));  
  arma::vec S(n,fill::ones);S = S*(lambda / rho);
  double abstol = 1e-4;
  double reltol = 1e-2;		
  for (int k=0;k<maxiter;k++){
    //update 'x'
    x = solve(A.t()*A+rho*I,A.t()*b+rho*(z-u));
    //update 'z' 
    arma::vec shrink = abs(x + u)-S;
    shrink.elem(find(shrink<0)).zeros();//soft-thresholding
    z = (shrink) % sign(x + u);
    //update 'u'
    u = u + (x - z);
    
    r_norm(k) = norm(x-z);
    
    if (norm(x)>norm(-z)){
      eps_pri(k) = sqrtn*abstol + reltol*norm(x);
    } else {
      eps_pri(k) = sqrtn*abstol + reltol*norm(-z);
    }
    
    if (r_norm(k) < eps_pri(k)){
      break;
    }
  }

Rcpp::List output;
  output["x"] = x;            
  output["r_norm"] = r_norm;
  output["eps_pri"] = eps_pri;
  return(output);
}
