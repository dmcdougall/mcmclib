#include <math.h>
#include <infmcmc.h>
#include <prior_general.h>

int main(int argc, char **argv) {
  int i, j, n;
  double dx, *evals, *evecs, norm;

  n = 100;
  dx = (double) 1.0 / (n + 1);

  evals = (double *)malloc(sizeof(double) * n);
  evecs = (double *)malloc(sizeof(double) * n * n);

  // Eigenvalues and eigenvectors of the Laplacian with Dirichlet boundary vals
  for (i = 0; i < n; i++) {
    norm = 0.5 - (sin(2.0 * M_PI * (i + 1)) / (4.0 * M_PI * (i + 1)));
    norm = sqrt(norm);
    for (j = 0; j < n; j++) {
      evecs[i*n+j] = sin(M_PI * (i + 1) * (j + 1) * dx) / norm;
    }
    evals[i] = (i + 1) * (i + 1) * M_PI * M_PI;
  }

  // Now set up the Markov chain and prior
  mcmc_infchain *chain = (mcmc_infchain *)malloc(sizeof(mcmc_infchain));
  mcmc_init_infchain(chain, MCMC_INFCHAIN_GENERAL, 10, 10);
  mcmc_infchain_set_prior_data(chain, evals, evecs, 2);
  mcmc_infchain_prior_draw(chain);

  FILE *fp = fopen("rand_prior_draw.txt", "w");
  fprintf(fp, "%lf %lf\n", 0.0, 0.0);
  for (i = 0; i < n; i++) {
    fprintf(fp, "%lf %lf\n", (i + 1) * dx, chain->_prior_draw[i]);
  }
  fprintf(fp, "%lf %lf\n", 1.0, 0.0);
  fclose(fp);

  free(evals);
  free(evecs);
}
