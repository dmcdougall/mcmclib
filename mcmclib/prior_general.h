#ifndef MCMC_PRIOR_GENERAL_H
#define MCMC_PRIOR_GENERAL_H

struct _prior_data {
  double *evals;
  double *evecs;
  double regularity;
};

typedef struct _prior_data prior_data;

#endif
