#include <stdlib.h>
#include <string.h>

#include "infmcmc.h"
#include "prior_general.h"

void _initialise_prior_data(prior_data *p, int n) {
  p->evals = (double *)malloc(sizeof(double) * n);
  p->evecs = (double *)malloc(sizeof(double) * n * n);
}

int mcmc_infchain_set_prior_data(mcmc_infchain *chain, double *evals,
    double *evecs) {
  int i;

  for (i = 0; i < chain->ndofs; i++) {
    if ((chain->_prior->evals[i] = evals[i]) <= 0) {
      // Precision operator is not positive-definite, so fail
      return -1;
    }
  }

  memcpy(chain->_prior->evecs, evecs,
      sizeof(double) * chain->ndofs * chain->ndofs);
  return 0;
}

void _free_prior_data(prior_data *p) {
  free(p->evals);
  free(p->evecs);
}
