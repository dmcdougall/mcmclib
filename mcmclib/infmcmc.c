#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <Accelerate/Accelerate.h>

#include "infmcmc.h"

#define nj1 32
#define nk1 32

void mcmc_init_infchain(mcmc_infchain *chain, const int nj, const int nk) {
  const int maxk      = (nk >> 1) + 1;
  const int sspectral = sizeof(fftw_complex) * nj * maxk;
  const int sphysical = sizeof(double      ) * nj * nk;
  //const int obsVecMem = sizeof(double      ) * sizeObsVector;
  FILE *fp;
  unsigned long int seed;
  
  // Set up variables
  chain->nj = nj;
  chain->nk = nk;
  //chain->sizeObsVector = sizeObsVector;
  chain->current_iter = 0;
  chain->accepted = 0;
  chain->_short_time_acc_prob_avg = 0.0;
  chain->_bLow = 0.0;
  chain->_bHigh = 1.0;
  
  // Allocate a ton of memory
  chain->current_physical_state      = (double *)malloc(sphysical);
  chain->avg_physical_state          = (double *)malloc(sphysical);
  chain->var_physical_state          = (double *)malloc(sphysical);
  chain->proposed_physical_state     = (double *)malloc(sphysical);
  chain->_M2                       = (double *)malloc(sphysical);
  //chain->current_state_observations  = (double *)malloc(obsVecMem);
  //chain->proposed_state_observations = (double *)malloc(obsVecMem);
  //chain->data                      = (double *)malloc(obsVecMem);
  
  chain->current_spectral_state  = (fftw_complex *)fftw_malloc(sspectral);
  chain->avg_spectral_state      = (fftw_complex *)fftw_malloc(sspectral);
  chain->prior_draw             = (fftw_complex *)fftw_malloc(sspectral);
  chain->proposed_spectral_state = (fftw_complex *)fftw_malloc(sspectral);
  
  memset(chain->current_physical_state,  0, sphysical);
  memset(chain->avg_physical_state,      0, sphysical);
  memset(chain->var_physical_state,      0, sphysical);
  memset(chain->proposed_physical_state, 0, sphysical);
  memset(chain->_M2,                   0, sphysical);
  memset(chain->current_spectral_state,  0, sspectral);
  memset(chain->avg_spectral_state,      0, sspectral);
  memset(chain->proposed_spectral_state, 0, sspectral);
  
  chain->acc_prob = 0.0;
  chain->avg_acc_prob = 0.0;
  chain->log_likelihood_current_state = 0.0;
  
  /*
   * Set some default values
   */
  chain->alpha_prior = 3.0;
  chain->rwmh_stepsize = 1e-4;
  chain->prior_var = 1.0;
  chain->prior_std = 1.0;
  
  chain->r = gsl_rng_alloc(gsl_rng_taus2);
  
  fp = fopen("/dev/urandom", "rb");
  
  if (fp != NULL) {
    fread(&seed, sizeof(unsigned long int), 1, fp);
    gsl_rng_set(chain->r, seed);
    fclose(fp);
    printf("Using random seed\n");
  }
  else {
    gsl_rng_set(chain->r, 0);
    printf("Using zero seed\n");
  }
  
  chain->_c2r = fftw_plan_dft_c2r_2d(nj, nk, chain->proposed_spectral_state, chain->proposed_physical_state, FFTW_MEASURE);
  chain->_r2c = fftw_plan_dft_r2c_2d(nj, nk, chain->current_physical_state, chain->current_spectral_state, FFTW_MEASURE);
}

void mcmc_free_infchain(mcmc_infchain *chain) {
  // Free all allocated memory used by the chain
  free(chain->current_physical_state);
  free(chain->avg_physical_state);
  free(chain->var_physical_state);
  free(chain->proposed_physical_state);
  free(chain->_M2);
  //free(chain->current_state_observations);
  //free(chain->proposed_state_observations);
  
  fftw_free(chain->current_spectral_state);
  fftw_free(chain->avg_spectral_state);
  fftw_free(chain->prior_draw);
  fftw_free(chain->proposed_spectral_state);
  
  gsl_rng_free(chain->r);
}

void mcmc_reset_infchain(mcmc_infchain *chain) {
  mcmc_free_infchain(chain);
  mcmc_init_infchain(chain, chain->nj, chain->nk);
}

void mcmc_write_infchain(const mcmc_infchain *chain, FILE *fp) {
  const int s = chain->nj * chain->nk;
  
  fwrite(&(chain->nj),                 sizeof(int),    1,         fp);
  fwrite(&(chain->nk),                 sizeof(int),    1,         fp);
  fwrite(&(chain->current_iter),        sizeof(int),    1,         fp);
  fwrite(chain->current_physical_state,  sizeof(double), s,         fp);
  fwrite(chain->avg_physical_state,      sizeof(double), s,         fp);
  fwrite(chain->var_physical_state,      sizeof(double), s,         fp);
  fwrite(&(chain->log_likelihood_current_state), sizeof(double), 1,         fp);
  fwrite(&(chain->acc_prob),            sizeof(double), 1,         fp);
  fwrite(&(chain->avg_acc_prob),         sizeof(double), 1,         fp);
}

void mcmc_write_infchain_info(const mcmc_infchain *chain, FILE *fp) {
  fwrite(&(chain->nj),             sizeof(int), 1, fp);
  fwrite(&(chain->nk),             sizeof(int), 1, fp);
}

void mcmc_write_vectorfield_infchain(const mcmc_infchain *U, const mcmc_infchain *V, FILE *fp) {
  const int s = U->nj * U->nk;
  
  fwrite(U->current_physical_state,  sizeof(double), s,         fp);
  fwrite(U->avg_physical_state,      sizeof(double), s,         fp);
  fwrite(U->var_physical_state,      sizeof(double), s,         fp);
  fwrite(V->current_physical_state,  sizeof(double), s,         fp);
  fwrite(V->avg_physical_state,      sizeof(double), s,         fp);
  fwrite(V->var_physical_state,      sizeof(double), s,         fp);
  fwrite(&(U->log_likelihood_current_state), sizeof(double), 1,         fp);
  fwrite(&(U->acc_prob),            sizeof(double), 1,         fp);
  fwrite(&(U->avg_acc_prob),         sizeof(double), 1,         fp);
}

void mcmc_print_infchain(mcmc_infchain *chain) {
  printf("Iteration %d\n", chain->current_iter);
  printf("-- Length is         %d x %d\n", chain->nj, chain->nk);
  printf("-- llhd val is       %lf\n", chain->log_likelihood_current_state);
  printf("-- Acc. prob is      %.10lf\n", chain->acc_prob);
  printf("-- Avg. acc. prob is %.10lf\n", chain->avg_acc_prob);
  printf("-- Beta is           %.10lf\n\n", chain->rwmh_stepsize);
  //finmcmc_printCurrentState(C);
  //finmcmc_printAvgState(C);
  //finmcmc_printVarState(C);
}

void random_prior_draw(mcmc_infchain *chain) {
  int j, k;
  const int maxk = (chain->nk >> 1) + 1;
  const int nko2 = chain->nk >> 1;
  const int njo2 = chain->nj >> 1;
  double xrand, yrand, c;
  //const double one = 1.0;
  
  c = 4.0 * M_PI * M_PI;
  
  for(j = 0; j < chain->nj; j++) {
    for(k = 0; k < maxk; k++) {
      xrand = gsl_ran_gaussian_ziggurat(chain->r, chain->prior_std);
      if((j == 0) && (k == 0)) {
        chain->prior_draw[0] = 0.0;
      }
      else if((j == njo2) && (k == nko2)) {
        chain->prior_draw[maxk*njo2+nko2] = /*chain->nj */ xrand / pow(c * ((j * j) + (k * k)), (double)chain->alpha_prior/2.0);
      }
      else if((j == 0) && (k == nko2)) {
        chain->prior_draw[nko2] = /*chain->nj */ xrand / pow(c * ((j * j) + (k * k)), (double)chain->alpha_prior/2.0);
      }
      else if((j == njo2) && (k == 0)) {
        chain->prior_draw[maxk*njo2] = /*chain->nj */ xrand / pow(c * ((j * j) + (k * k)), (double)chain->alpha_prior/2.0);
      }
      else {
        xrand /= sqrt(2.0);
        yrand = gsl_ran_gaussian_ziggurat(chain->r, chain->prior_std) / sqrt(2.0);
        chain->prior_draw[maxk*j+k] = /*chain->nj */ (xrand + I * yrand) / pow(c * ((j * j) + (k * k)), (double)chain->alpha_prior/2.0);
        if(j > njo2) {
          chain->prior_draw[maxk*j+k] = conj(chain->prior_draw[maxk*(chain->nj-j)+k]);
        }
      }
    }
  }
}

void random_divfree_prior_draw(mcmc_infchain *chain1, mcmc_infchain *chain2) {
  int j, k;
  const int maxk = (chain1->nk >> 1) + 1;
  const int njo2 = chain2->nj >> 1;
  const int nko2 = chain1->nk >> 1;
  double modk;
  
  random_prior_draw(chain1);
  
  for (j = 0; j < chain1->nj; j++) {
    for (k = 0; k < maxk; k++) {
      if (j == 0 && k == 0) {
        chain1->prior_draw[0] = 0.0;
        chain2->prior_draw[0] = 0.0;
        continue;
      }
      
      modk = sqrt(j * j + k * k);
      if (j < njo2) {
        chain2->prior_draw[maxk*j+k] = -j * chain1->prior_draw[maxk*j+k] / modk;
      }
      else if (j > njo2) {
        chain2->prior_draw[maxk*j+k] = -(j - chain1->nj) * chain1->prior_draw[maxk*j+k] / modk;
      }
      else {
        chain2->prior_draw[maxk*j+k] = 0.0;
      }
      
      if (k < nko2) {
        chain1->prior_draw[maxk*j+k] *= k / modk;
      }
      else {
        chain1->prior_draw[maxk*j+k] = 0.0;
      }
    }
  }
}

void mcmc_seed_with_prior(mcmc_infchain *chain) {
  const int size = sizeof(fftw_complex) * chain->nj * ((chain->nk >> 1) + 1);
  fftw_complex *uk = (fftw_complex *)fftw_malloc(size);
  const fftw_plan p = fftw_plan_dft_c2r_2d(chain->nj, chain->nk, uk, chain->current_physical_state, FFTW_ESTIMATE);
  
  random_prior_draw(chain);
  memcpy(uk, chain->prior_draw, size);
  
  //fixme: put into current spectral state
  
  fftw_execute(p);
  fftw_destroy_plan(p);
  fftw_free(uk);
}

void mcmc_seed_with_divfree_prior(mcmc_infchain *chain1, mcmc_infchain *chain2) {
  const int size = sizeof(fftw_complex) * chain1->nj * ((chain1->nk >> 1) + 1);
  fftw_complex *uk = (fftw_complex *)fftw_malloc(size);
  const fftw_plan p = fftw_plan_dft_c2r_2d(chain1->nj, chain1->nk, uk, chain1->current_physical_state, FFTW_ESTIMATE);
  
  random_divfree_prior_draw(chain1, chain2);
  
  memcpy(uk, chain1->prior_draw, size);
  fftw_execute_dft_c2r(p, uk, chain1->current_physical_state);
  
  memcpy(uk, chain2->prior_draw, size);
  fftw_execute_dft_c2r(p, uk, chain2->current_physical_state);
  
  memcpy(chain1->current_spectral_state, chain1->prior_draw, size);
  memcpy(chain2->current_spectral_state, chain2->prior_draw, size);
  
  fftw_destroy_plan(p);
  fftw_free(uk);
}

void mcmc_propose_RWMH(mcmc_infchain *chain) {
  int j, k;
  const int maxk = (chain->nk >> 1) + 1;
  const int N = chain->nj * chain->nk;
  const double sqrtOneMinusBeta2 = sqrt(1.0 - chain->rwmh_stepsize * chain->rwmh_stepsize);
  double *u = (double *)malloc(sizeof(double) * chain->nj * chain->nk);
  fftw_complex *uk = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * chain->nj * maxk);
  
  memcpy(u, chain->current_physical_state, sizeof(double) * chain->nj * chain->nk);
  fftw_execute_dft_r2c(chain->_r2c, u, chain->current_spectral_state);
  
  // Draw from prior distribution
  random_prior_draw(chain);
  
  for(j = 0; j < chain->nj; j++) {
    for(k = 0; k < maxk; k++) {
      chain->proposed_spectral_state[maxk*j+k] = sqrtOneMinusBeta2 * chain->current_spectral_state[maxk*j+k] + chain->rwmh_stepsize * chain->prior_draw[maxk*j+k];
      chain->proposed_spectral_state[maxk*j+k] /= N;
    }
  }
  
  memcpy(uk, chain->proposed_spectral_state, sizeof(fftw_complex) * chain->nj * maxk);
  fftw_execute_dft_c2r(chain->_c2r, uk, chain->proposed_physical_state);
  fftw_free(uk);
  free(u);
}

void mcmc_adapt_RWMH_stepsize(mcmc_infchain *chain, double inc) {
  // Adapt to stay in 20-30% range.
  int adaptFreq = 100;
  double rate;
  
  if (chain->current_iter > 0 && chain->current_iter % adaptFreq == 0) {
    rate = (double) chain->_short_time_acc_prob_avg / adaptFreq;
    
    if (rate < 0.2) {
      //chain->_bHigh = chain->rwmh_stepsize;
      //chain->rwmh_stepsize = (chain->_bLow + chain->_bHigh) / 2.0;
      chain->rwmh_stepsize -= inc;
    }
    else if (rate > 0.3) {
      //chain->_bLow = chain->rwmh_stepsize;
      //chain->rwmh_stepsize = (chain->_bLow + chain->_bHigh) / 2.0;
      chain->rwmh_stepsize += inc;
    }
    
    chain->_short_time_acc_prob_avg = 0.0;
  }
  else {
    chain->_short_time_acc_prob_avg += chain->acc_prob;
  }
}

void mcmc_propose_divfree_RWMH(mcmc_infchain *chain1, mcmc_infchain *chain2) {
  int j, k;
  const int maxk = (chain1->nk >> 1) + 1;
  const int N = chain1->nj * chain1->nk;
  const double sqrtOneMinusBeta2 = sqrt(1.0 - chain1->rwmh_stepsize * chain1->rwmh_stepsize);
  double *u = (double *)malloc(sizeof(double) * chain1->nj * chain1->nk);
  fftw_complex *uk = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * chain1->nj * maxk);
  
  memcpy(u, chain1->current_physical_state, sizeof(double) * chain1->nj * chain1->nk);
  fftw_execute_dft_r2c(chain1->_r2c, u, chain1->current_spectral_state);
  
  memcpy(u, chain2->current_physical_state, sizeof(double) * chain2->nj * chain2->nk);
  fftw_execute_dft_r2c(chain2->_r2c, u, chain2->current_spectral_state);
    
  // Draw from prior distribution
  random_divfree_prior_draw(chain1, chain2);
  
  for(j = 0; j < chain1->nj; j++) {
    for(k = 0; k < maxk; k++) {
      chain1->current_spectral_state[maxk*j+k] /= N;
      chain2->current_spectral_state[maxk*j+k] /= N;
      chain1->proposed_spectral_state[maxk*j+k] = sqrtOneMinusBeta2 * chain1->current_spectral_state[maxk*j+k] + chain1->rwmh_stepsize * chain1->prior_draw[maxk*j+k];
      //chain1->proposed_spectral_state[maxk*j+k] /= N;
      chain2->proposed_spectral_state[maxk*j+k] = sqrtOneMinusBeta2 * chain2->current_spectral_state[maxk*j+k] + chain2->rwmh_stepsize * chain2->prior_draw[maxk*j+k];
      //chain2->proposed_spectral_state[maxk*j+k] /= N;
    }
  }
  
  memcpy(uk, chain1->proposed_spectral_state, sizeof(fftw_complex) * chain1->nj * maxk);
  fftw_execute_dft_c2r(chain1->_c2r, uk, chain1->proposed_physical_state);
  memcpy(uk, chain2->proposed_spectral_state, sizeof(fftw_complex) * chain1->nj * maxk);
  fftw_execute_dft_c2r(chain2->_c2r, uk, chain2->proposed_physical_state);
  
  fftw_free(uk);
  free(u);
}

void mcmc_update_avgs(mcmc_infchain *chain) {
  int j, k;
  double deltar;
  fftw_complex deltac;
  chain->current_iter++;
  
  /*
   Update physical vectors
   */
  if (chain->current_iter == 1) {
    for (j = 0; j < chain->nj; j++) {
      for (k = 0; k < chain->nk; k++) {
        deltar = chain->current_physical_state[chain->nk * j + k] - chain->avg_physical_state[chain->nk * j + k];
        chain->avg_physical_state[chain->nk * j + k] += (deltar / chain->current_iter);
        chain->_M2[chain->nk * j + k] += (deltar * (chain->current_physical_state[chain->nk * j + k] - chain->avg_physical_state[chain->nk * j + k]));
        chain->var_physical_state[chain->nk * j + k] = -1.0;
      }
    }
  }
  else {
    for (j = 0; j < chain->nj; j++) {
      for (k = 0; k < chain->nk; k++) {
        deltar = chain->current_physical_state[chain->nk * j + k] - chain->avg_physical_state[chain->nk * j + k];
        chain->avg_physical_state[chain->nk * j + k] += (deltar / chain->current_iter);
        chain->_M2[chain->nk * j + k] += (deltar * (chain->current_physical_state[chain->nk * j + k] - chain->avg_physical_state[chain->nk * j + k]));
        chain->var_physical_state[chain->nk * j + k] = chain->_M2[chain->nk * j + k] / (chain->current_iter - 1);
      }
    }
  }
  
  /*
   Update spectral vectors
  */
  if (chain->current_iter == 1) {
    for (j = 0; j < chain->nj; j++) {
      for (k = 0; k < chain->nk/2 + 1; k++) {
        deltac = chain->current_spectral_state[(chain->nk/2 + 1) * j + k] - chain->avg_spectral_state[(chain->nk/2 + 1) * j + k];
        chain->avg_spectral_state[(chain->nk/2 + 1) * j + k] += (deltac / chain->current_iter);
      }
    }
  }
  else {
    for (j = 0; j < chain->nj; j++) {
      for (k = 0; k < chain->nk/2 + 1; k++) {
        deltac = chain->current_spectral_state[(chain->nk/2 + 1) * j + k] - chain->avg_spectral_state[(chain->nk/2 + 1) * j + k];
        chain->avg_spectral_state[(chain->nk/2 + 1) * j + k] += (deltac / chain->current_iter);
      }
    }
  }
  
  /*
   Update scalars
   */
  chain->avg_acc_prob += ((chain->acc_prob - chain->avg_acc_prob) / chain->current_iter);
}

void mcmc_update_RWMH(mcmc_infchain *chain, double logLHDOfProposal) {
  double alpha;
  
  alpha = exp(chain->log_likelihood_current_state - logLHDOfProposal);
  
  if (alpha > 1.0) {
    alpha = 1.0;
  }
  // fixme: set acc prob here instead
  if (gsl_rng_uniform(chain->r) < alpha) {
    memcpy(chain->current_spectral_state, chain->proposed_spectral_state, sizeof(fftw_complex) * chain->nj * ((chain->nk >> 1) + 1));
    memcpy(chain->current_physical_state, chain->proposed_physical_state, sizeof(double) * chain->nj * chain->nk);
    chain->acc_prob = alpha;
    chain->log_likelihood_current_state = logLHDOfProposal;
  }
  
  mcmc_update_avgs(chain);
}

void mcmc_update_vectorfield_RWMH(mcmc_infchain *chain1, mcmc_infchain *chain2, double logLHDOfProposal) {
  double alpha;
  
  // log likelihoods will be the same for both chains
	alpha = exp(chain1->log_likelihood_current_state - logLHDOfProposal);
	
  if (alpha > 1.0) {
    alpha = 1.0;
  }
  
  chain1->accepted = 0;
  chain2->accepted = 0;
  chain1->acc_prob = alpha;
  chain2->acc_prob = alpha;
  
  if (gsl_rng_uniform(chain1->r) < alpha) {
    chain1->accepted = 1;
    chain2->accepted = 1;
    memcpy(chain1->current_spectral_state, chain1->proposed_spectral_state, sizeof(fftw_complex) * chain1->nj * ((chain1->nk >> 1) + 1));
    memcpy(chain1->current_physical_state, chain1->proposed_physical_state, sizeof(double) * chain1->nj * chain1->nk);
    chain1->log_likelihood_current_state = logLHDOfProposal;
    memcpy(chain2->current_spectral_state, chain2->proposed_spectral_state, sizeof(fftw_complex) * chain2->nj * ((chain2->nk >> 1) + 1));
    memcpy(chain2->current_physical_state, chain2->proposed_physical_state, sizeof(double) * chain2->nj * chain2->nk);
    chain2->log_likelihood_current_state = logLHDOfProposal;
  }
  
  mcmc_update_avgs(chain1);
  mcmc_update_avgs(chain2);
}

void mcmc_set_RWMH_stepsize(mcmc_infchain *chain, double beta) {
  chain->rwmh_stepsize = beta;
}

void mcmc_set_prior_alpha(mcmc_infchain *chain, double alpha) {
  chain->alpha_prior = alpha;
}

void mcmc_set_prior_var(mcmc_infchain *chain, double var) {
  chain->prior_var = var;
  chain->prior_std = sqrt(var);
}

double L2Field(fftw_complex *uk, int nj, int nk) {
  int j, k;
  const int maxk = (nk >> 1) + 1;
  double sum = 0.0;
  
  for (j = 0; j < nj; j++) {
    for (k = 0; k < maxk; k++) {
      sum += cabs(uk[maxk*j+k]) * cabs(uk[maxk*j+k]);
    }
  }
  
  return 2.0 * sum;
}

double mcmc_current_L2(mcmc_infchain *chain) {
  return L2Field(chain->current_spectral_state, chain->nj, chain->nk);
}

double mcmc_proposed_L2(mcmc_infchain *chain) {
  return L2Field(chain->proposed_spectral_state, chain->nj, chain->nk);
}

double mcmc_prior_L2(mcmc_infchain *chain) {
  return L2Field(chain->prior_draw, chain->nj, chain->nk);
}































void randomPriorDrawOLD(gsl_rng *r, double PRIOR_ALPHA, fftw_complex *randDrawCoeffs) {
  int j, k;
  double xrand, yrand, c;//, scale;
  //const double one = 1.0;
  
  c = 4.0 * M_PI * M_PI;
  
  for(j = 0; j < nj1; j++) {
    for(k = 0; k < nk1/2 + 1; k++) {
      if((j == 0) && (k == 0)) {
        randDrawCoeffs[0] = 0.0;
      }
      else if((j == nj1/2) && (k == nk1/2)) {
        xrand = gsl_ran_gaussian_ziggurat(r, 1.0);
        randDrawCoeffs[(nk1/2 + 1) * nj1/2 + nk1/2] = xrand / pow((c * ((j * j) + (k * k))), (double)PRIOR_ALPHA/2.0);
      }
      else if((j == 0) && (k == nk1/2)) {
        xrand = gsl_ran_gaussian_ziggurat(r, 1.0);
        randDrawCoeffs[nk1/2] = xrand / pow((c * ((j * j) + (k * k))), (double)PRIOR_ALPHA/2.0);
      }
      else if((j == nj1/2) && (k == 0)) {
        xrand = gsl_ran_gaussian_ziggurat(r, 1.0);
        randDrawCoeffs[(nk1/2 + 1) * nj1/2] = xrand / pow((c * ((j * j) + (k * k))), (double)PRIOR_ALPHA/2.0);
      }
      else {
        xrand = gsl_ran_gaussian_ziggurat(r, 1.0) / sqrt(2.0);
        yrand = gsl_ran_gaussian_ziggurat(r, 1.0) / sqrt(2.0);
        randDrawCoeffs[(nk1/2 + 1) * j + k] = (xrand + I * yrand) / pow((c * ((j * j) + (k * k))), (double)PRIOR_ALPHA/2.0);
        if(j > nj1/2) {
          randDrawCoeffs[(nk1/2 + 1) * j + k] = conj(randDrawCoeffs[(nk1/2+1)*(nj1-j)+k]);
        }
      }
    }
  }
  
  /*
  for(j = 1; j < nj1 / 2; j++) {
    for(k = 0; k < nk1 / 2 + 1; k++) {
      xrand = gsl_ran_gaussian_ziggurat(r, 1.0) / M_SQRT2;
      yrand = gsl_ran_gaussian_ziggurat(r, 1.0) / M_SQRT2;
      scale = pow(c * ((j * j) + (k * k)), (double)PRIOR_ALPHA/2.0);
      randDrawCoeffs[(nk1/2 + 1) * j + k] = (xrand + I * yrand) / scale;
      randDrawCoeffs[(nk1/2 + 1) * (nj1-j) + k] = (xrand - I * yrand) / scale;
    }
  }
  
  for(k = 1; k < nk1 / 2 + 1; k++) {
    xrand = gsl_ran_gaussian_ziggurat(r, 1.0) / M_SQRT2;
    randDrawCoeffs[k] = (xrand + I * yrand) / pow(c * (k * k), (double)PRIOR_ALPHA/2.0);
  }
  
  for(k = 0; k < nk1/2; k++) {
    xrand = gsl_ran_gaussian_ziggurat(r, 1.0) / M_SQRT2;
    randDrawCoeffs[(nk1/2+1)*(nj1/2)+k] = (xrand + I * yrand) / pow(c * ((nj1 * nj1 / 4) + (k * k)), (double)PRIOR_ALPHA/2.0);
  }
  
  randDrawCoeffs[0] = 0.0;
  
  xrand = gsl_ran_gaussian_ziggurat(r, 1.0);
  randDrawCoeffs[(nk1/2+1)*(nj1/2)+(nk1/2)] = xrand / pow(c * ((j * j) + (k * k)), (double)PRIOR_ALPHA/2.0);
  */
}

void setRWMHStepSize(mcmc_infchain *chain, double stepSize) {
  chain->rwmh_stepsize = stepSize;
}

void resetAvgs(mcmc_infchain *chain) {
  /*
    Sets avg_physical_state to current_physical_state and
         avg_spectral_state to current_spectral_state
  */
  const size_t size_doublenj = sizeof(double) * chain->nj;
  memcpy(chain->avg_physical_state, chain->current_physical_state, size_doublenj * chain->nk);
  memcpy(chain->avg_spectral_state, chain->current_spectral_state, size_doublenj * ((chain->nk >> 1) + 1));
}

void resetVar(mcmc_infchain *chain) {
  // Sets var_physical_state to 0
  memset(chain->var_physical_state, 0, sizeof(double) * chain->nj * chain->nk);
}

void proposeIndependence(mcmc_infchain *chain) {
  const int maxk = (chain->nk >> 1) + 1;
  
  memcpy(chain->proposed_spectral_state, chain->prior_draw, sizeof(fftw_complex) * chain->nj * maxk);
}

double lsqFunctional(const double * const data, const double * const obsVec, const int obsVecSize, const double obs_std_dev) {
  int i;
  double temp1, sum1 = 0.0;
  
  for(i = 0; i < obsVecSize; i++) {
    temp1 = (data[i] - obsVec[i]);
    sum1 += temp1 * temp1;
  }
  
  return sum1 / (2.0 * obs_std_dev * obs_std_dev);
}

void acceptReject(mcmc_infchain *chain) {
  const double phi1 = lsqFunctional(chain->data, chain->current_state_observations, chain->sizeObsVector, chain->obs_std_dev);
  const double phi2 = lsqFunctional(chain->data, chain->proposed_state_observations, chain->sizeObsVector, chain->obs_std_dev);
  double tempAccProb = exp(phi1 - phi2);
  
  if(tempAccProb > 1.0) {
    tempAccProb = 1.0;
  }
  
  if(gsl_rng_uniform(chain->r) < tempAccProb) {
    memcpy(chain->current_spectral_state, chain->proposed_spectral_state, sizeof(fftw_complex) * chain->nj * ((chain->nk >> 1) + 1));
    memcpy(chain->current_physical_state, chain->proposed_physical_state, sizeof(double) * chain->nj * chain->nk);
    memcpy(chain->current_state_observations, chain->proposed_state_observations, sizeof(double) * chain->sizeObsVector);
    
    chain->acc_prob = tempAccProb;
    chain->current_LSQFunctional = phi2;
  }
  else {
    chain->current_LSQFunctional = phi1;
  }
}

//
//void updateChain(mcmc_infchain *chain) {
//chain->currentIter++;
//void updateAvgs(mcmc_infchain *chain);
//void updateVar(mcmc_infchain *chain);
/*
int main(void) {
  int nj = 32, nk = 32, sizeObsVector = 0;
  unsigned long int randseed = 0;
  
  FILE *fp;
  mcmc_infchain *chain;
  
  C = (mcmc_infchain *)malloc(sizeof(mcmc_infchain));
  
  initChain(C, nj, nk, sizeObsVector, randseed);
  
  randomPriorDraw2(C);
  
  fp = fopen("2.dat", "w");
  fwrite(chain->priorDraw, sizeof(fftw_complex), nj * (nk / 2 + 1), fp);
  fclose(fp);
  
  return 0;
}
*/
