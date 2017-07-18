data {
  int<lower=1> N;
  int<lower=1> P; # number of replicates
  int<lower=1> K; # number of latent functions
  int<lower=1> L; # number of priors
  int<lower=1, upper=L> prior[K]; # prior assignment for each function
  real alpha_prior[L,2];
  real length_scale_prior[L,2];
  real marginal_alpha_prior[2];
  real marginal_lengthscale_prior[2];
  real sigma_prior[2];

  matrix[P,K] design;
  row_vector[N] y[P];
  real x[N];
}
parameters {
  real<lower=0> length_scale[L];
  real<lower=0> alpha[L];
  real<lower=0> marginal_alpha;
  real<lower=0> marginal_lengthscale;
  real<lower=0> sigma;
  vector[N] f_eta[K];
}
transformed parameters {
  matrix[K,N] f;

  for (l in 1:L)
  {
    matrix[N, N] L_cov;
    matrix[N, N] cov;
    cov = cov_exp_quad(x, alpha[l], length_scale[l]);
    for (n in 1:N)
      cov[n, n] = cov[n, n] + 1e-12;
    L_cov = cholesky_decompose(cov);

    for (k in 1:K)
      {
        if (prior[k] == l)
          f[k] = (L_cov * f_eta[k])';
      }
  }
}
model {

  matrix[N, N] L_cov;

  for (l in 1:L)
  {
    length_scale[l] ~ gamma(length_scale_prior[l,1], length_scale_prior[l,2]);
    alpha[l] ~ gamma(alpha_prior[l,1], alpha_prior[l,2]);
  }

  sigma ~ gamma(sigma_prior[1], sigma_prior[2]);

  for (i in 1:K)
    f_eta[i] ~ normal(0, 1);



  marginal_lengthscale ~ gamma(marginal_lengthscale_prior[1], marginal_lengthscale_prior[2]);
  marginal_alpha ~ gamma(marginal_alpha_prior[1], marginal_alpha_prior[2]);


  {
    matrix[N, N] cov;
    cov = cov_exp_quad(x, marginal_alpha, marginal_lengthscale);
    for (n in 1:N)
      cov[n, n] = cov[n, n] + square(sigma);
    L_cov = cholesky_decompose(cov);
  }

  // cov = cov_exp_quad(x, marginal_alpha, marginal_lengthscale);
  //   for (n in 1:N)
  //     cov[n, n] = pow(sigma, 2) + cov[n, n];

  for (i in 1:P)
    y[i] ~ multi_normal_cholesky(design[i]*f, L_cov);
}
