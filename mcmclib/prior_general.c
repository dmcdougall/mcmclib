#include <stdlib.h>
#include <string.h>

#include "prior_general.h"

void initialise_prior_data(prior_data *p, int n) {
  p->evals = (double *)malloc(sizeof(double) * n);
  p->evecs = (double *)malloc(sizeof(double) * n * n);
}

int set_prior_data(prior_data *p, int n, double *evals, double *evecs) {
  int i;

  for (i = 0; i < n; i++) {
    if ((p->evals[i] = evals[i]) <= 0) {
      // Precision operator is not positive-definite, so fail
      return -1;
    }
  }

  memcpy(p->evecs, evecs, sizeof(double) * n * n);
  return 0;
}

void free_prior_data(prior_data *p) {
  free(p->evals);
  free(p->evecs);
}
