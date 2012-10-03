#ifndef MCMC_PRIOR_GENERAL_H
#define MCMC_PRIOR_GENERAL_H

struct _prior_data {
  double *evals;
  double *evecs;
  double regularity;

  // Private members
  unsigned int _type;
  void (* _generate_draw)(struct _prior_data *p, gsl_rng *r, double *draw,
      int n);
};

typedef struct _prior_data prior_data;

void _initialise_prior_data(prior_data *p, unsigned int type, int n);

#endif
