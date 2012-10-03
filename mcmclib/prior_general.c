#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "infmcmc.h"
#include "prior_general.h"

void _generate_prior_draw_general(prior_data *p, gsl_rng *r, double *draw,
    int n);

void _initialise_prior_data(prior_data *p, unsigned int type, int n) {
  p->evals = (double *)malloc(sizeof(double) * n);
  p->evecs = (double *)malloc(sizeof(double) * n * n);

  if (type == MCMC_INFCHAIN_GENERAL) {
    p->_generate_draw = _generate_prior_draw_general;
  }
}

int mcmc_infchain_set_prior_data(mcmc_infchain *chain, double *evals,
    double *evecs, double regularity) {
  int i;

  for (i = 0; i < chain->ndofs; i++) {
    if ((chain->_prior->evals[i] = evals[i]) <= 0) {
      // Precision operator is not positive-definite, so fail
      return -1;
    }
  }

  if ((chain->_prior->regularity = regularity) < 0) {
    return -1;
  }

  memcpy(chain->_prior->evecs, evecs,
      sizeof(double) * chain->ndofs * chain->ndofs);
  return 0;
}

void _free_prior_data(prior_data *p) {
  free(p->evals);
  free(p->evecs);
}

void _generate_prior_draw_general(prior_data *p, gsl_rng *r, double *draw,
    int n) {
  int i, j;
  double xi, *rand_coeffs;

  rand_coeffs = (double *)malloc(sizeof(double) * n);

  for (i = 0; i < n; i++) {
    rand_coeffs[i] = gsl_ran_gaussian_ziggurat(r, 1);
  }

  for (i = 0; i < n; i++) {
    draw[i] = 0.0;
    for (j = 0; j < n; j++) {
      // This can probably be optimised
      draw[i] += pow(p->evals[j], -p->regularity / 2.0) * rand_coeffs[j] * p->evecs[j*n+i];
    }
  }
  free(rand_coeffs);
}
