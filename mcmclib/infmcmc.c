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

void infmcmc_initChain(mcmc_infchain *C, const int nj, const int nk) {
  const int maxk      = (nk >> 1) + 1;
  const int sspectral = sizeof(fftw_complex) * nj * maxk;
  const int sphysical = sizeof(double      ) * nj * nk;
  //const int obsVecMem = sizeof(double      ) * sizeObsVector;
  FILE *fp;
  unsigned long int seed;
  
  // Set up variables
  C->nj = nj;
  C->nk = nk;
  //C->sizeObsVector = sizeObsVector;
  C->current_iter = 0;
  C->accepted = 0;
  C->_short_time_acc_prob_avg = 0.0;
  C->_bLow = 0.0;
  C->_bHigh = 1.0;
  
  // Allocate a ton of memory
  C->current_physical_state      = (double *)malloc(sphysical);
  C->avg_physical_state          = (double *)malloc(sphysical);
  C->var_physical_state          = (double *)malloc(sphysical);
  C->proposed_physical_state     = (double *)malloc(sphysical);
  C->_M2                       = (double *)malloc(sphysical);
  //C->current_state_observations  = (double *)malloc(obsVecMem);
  //C->proposed_state_observations = (double *)malloc(obsVecMem);
  //C->data                      = (double *)malloc(obsVecMem);
  
  C->current_spectral_state  = (fftw_complex *)fftw_malloc(sspectral);
  C->avg_spectral_state      = (fftw_complex *)fftw_malloc(sspectral);
  C->prior_draw             = (fftw_complex *)fftw_malloc(sspectral);
  C->proposed_spectral_state = (fftw_complex *)fftw_malloc(sspectral);
  
  memset(C->current_physical_state,  0, sphysical);
  memset(C->avg_physical_state,      0, sphysical);
  memset(C->var_physical_state,      0, sphysical);
  memset(C->proposed_physical_state, 0, sphysical);
  memset(C->_M2,                   0, sphysical);
  memset(C->current_spectral_state,  0, sspectral);
  memset(C->avg_spectral_state,      0, sspectral);
  memset(C->proposed_spectral_state, 0, sspectral);
  
  C->acc_prob = 0.0;
  C->avg_acc_prob = 0.0;
  C->log_likelihood_current_state = 0.0;
  
  /*
   * Set some default values
   */
  C->alpha_prior = 3.0;
  C->rwmh_stepsize = 1e-4;
  C->prior_var = 1.0;
  C->prior_std = 1.0;
  
  C->r = gsl_rng_alloc(gsl_rng_taus2);
  
  fp = fopen("/dev/urandom", "rb");
  
  if (fp != NULL) {
    fread(&seed, sizeof(unsigned long int), 1, fp);
    gsl_rng_set(C->r, seed);
    fclose(fp);
    printf("Using random seed\n");
  }
  else {
    gsl_rng_set(C->r, 0);
    printf("Using zero seed\n");
  }
  
  C->_c2r = fftw_plan_dft_c2r_2d(nj, nk, C->proposed_spectral_state, C->proposed_physical_state, FFTW_MEASURE);
  C->_r2c = fftw_plan_dft_r2c_2d(nj, nk, C->current_physical_state, C->current_spectral_state, FFTW_MEASURE);
}

void infmcmc_freeChain(mcmc_infchain *C) {
  // Free all allocated memory used by the chain
  free(C->current_physical_state);
  free(C->avg_physical_state);
  free(C->var_physical_state);
  free(C->proposed_physical_state);
  free(C->_M2);
  //free(C->current_state_observations);
  //free(C->proposed_state_observations);
  
  fftw_free(C->current_spectral_state);
  fftw_free(C->avg_spectral_state);
  fftw_free(C->prior_draw);
  fftw_free(C->proposed_spectral_state);
  
  gsl_rng_free(C->r);
}

void infmcmc_resetChain(mcmc_infchain *C) {
  infmcmc_freeChain(C);
  infmcmc_initChain(C, C->nj, C->nk);
}

void infmcmc_writeChain(const mcmc_infchain *C, FILE *fp) {
  const int s = C->nj * C->nk;
  
  fwrite(&(C->nj),                 sizeof(int),    1,         fp);
  fwrite(&(C->nk),                 sizeof(int),    1,         fp);
  fwrite(&(C->current_iter),        sizeof(int),    1,         fp);
  fwrite(C->current_physical_state,  sizeof(double), s,         fp);
  fwrite(C->avg_physical_state,      sizeof(double), s,         fp);
  fwrite(C->var_physical_state,      sizeof(double), s,         fp);
  fwrite(&(C->log_likelihood_current_state), sizeof(double), 1,         fp);
  fwrite(&(C->acc_prob),            sizeof(double), 1,         fp);
  fwrite(&(C->avg_acc_prob),         sizeof(double), 1,         fp);
}

void infmcmc_writeChainInfo(const mcmc_infchain *C, FILE *fp) {
  fwrite(&(C->nj),             sizeof(int), 1, fp);
  fwrite(&(C->nk),             sizeof(int), 1, fp);
}

void infmcmc_writeVFChain(const mcmc_infchain *U, const mcmc_infchain *V, FILE *fp) {
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

void infmcmc_printChain(mcmc_infchain *C) {
  printf("Iteration %d\n", C->current_iter);
  printf("-- Length is         %d x %d\n", C->nj, C->nk);
  printf("-- llhd val is       %lf\n", C->log_likelihood_current_state);
  printf("-- Acc. prob is      %.10lf\n", C->acc_prob);
  printf("-- Avg. acc. prob is %.10lf\n", C->avg_acc_prob);
  printf("-- Beta is           %.10lf\n\n", C->rwmh_stepsize);
  //finmcmc_printCurrentState(C);
  //finmcmc_printAvgState(C);
  //finmcmc_printVarState(C);
}

void randomPriorDraw(mcmc_infchain *C) {
  int j, k;
  const int maxk = (C->nk >> 1) + 1;
  const int nko2 = C->nk >> 1;
  const int njo2 = C->nj >> 1;
  double xrand, yrand, c;
  //const double one = 1.0;
  
  c = 4.0 * M_PI * M_PI;
  
  for(j = 0; j < C->nj; j++) {
    for(k = 0; k < maxk; k++) {
      xrand = gsl_ran_gaussian_ziggurat(C->r, C->prior_std);
      if((j == 0) && (k == 0)) {
        C->prior_draw[0] = 0.0;
      }
      else if((j == njo2) && (k == nko2)) {
        C->prior_draw[maxk*njo2+nko2] = /*C->nj */ xrand / pow(c * ((j * j) + (k * k)), (double)C->alpha_prior/2.0);
      }
      else if((j == 0) && (k == nko2)) {
        C->prior_draw[nko2] = /*C->nj */ xrand / pow(c * ((j * j) + (k * k)), (double)C->alpha_prior/2.0);
      }
      else if((j == njo2) && (k == 0)) {
        C->prior_draw[maxk*njo2] = /*C->nj */ xrand / pow(c * ((j * j) + (k * k)), (double)C->alpha_prior/2.0);
      }
      else {
        xrand /= sqrt(2.0);
        yrand = gsl_ran_gaussian_ziggurat(C->r, C->prior_std) / sqrt(2.0);
        C->prior_draw[maxk*j+k] = /*C->nj */ (xrand + I * yrand) / pow(c * ((j * j) + (k * k)), (double)C->alpha_prior/2.0);
        if(j > njo2) {
          C->prior_draw[maxk*j+k] = conj(C->prior_draw[maxk*(C->nj-j)+k]);
        }
      }
    }
  }
}

void randomDivFreePriorDraw(mcmc_infchain *C1, mcmc_infchain *C2) {
  int j, k;
  const int maxk = (C1->nk >> 1) + 1;
  const int njo2 = C2->nj >> 1;
  const int nko2 = C1->nk >> 1;
  double modk;
  
  randomPriorDraw(C1);
  
  for (j = 0; j < C1->nj; j++) {
    for (k = 0; k < maxk; k++) {
      if (j == 0 && k == 0) {
        C1->prior_draw[0] = 0.0;
        C2->prior_draw[0] = 0.0;
        continue;
      }
      
      modk = sqrt(j * j + k * k);
      if (j < njo2) {
        C2->prior_draw[maxk*j+k] = -j * C1->prior_draw[maxk*j+k] / modk;
      }
      else if (j > njo2) {
        C2->prior_draw[maxk*j+k] = -(j - C1->nj) * C1->prior_draw[maxk*j+k] / modk;
      }
      else {
        C2->prior_draw[maxk*j+k] = 0.0;
      }
      
      if (k < nko2) {
        C1->prior_draw[maxk*j+k] *= k / modk;
      }
      else {
        C1->prior_draw[maxk*j+k] = 0.0;
      }
    }
  }
}

void infmcmc_seedWithPriorDraw(mcmc_infchain *C) {
  const int size = sizeof(fftw_complex) * C->nj * ((C->nk >> 1) + 1);
  fftw_complex *uk = (fftw_complex *)fftw_malloc(size);
  const fftw_plan p = fftw_plan_dft_c2r_2d(C->nj, C->nk, uk, C->current_physical_state, FFTW_ESTIMATE);
  
  randomPriorDraw(C);
  memcpy(uk, C->prior_draw, size);
  
  //fixme: put into current spectral state
  
  fftw_execute(p);
  fftw_destroy_plan(p);
  fftw_free(uk);
}

void infmcmc_seedWithDivFreePriorDraw(mcmc_infchain *C1, mcmc_infchain *C2) {
  const int size = sizeof(fftw_complex) * C1->nj * ((C1->nk >> 1) + 1);
  fftw_complex *uk = (fftw_complex *)fftw_malloc(size);
  const fftw_plan p = fftw_plan_dft_c2r_2d(C1->nj, C1->nk, uk, C1->current_physical_state, FFTW_ESTIMATE);
  
  randomDivFreePriorDraw(C1, C2);
  
  memcpy(uk, C1->prior_draw, size);
  fftw_execute_dft_c2r(p, uk, C1->current_physical_state);
  
  memcpy(uk, C2->prior_draw, size);
  fftw_execute_dft_c2r(p, uk, C2->current_physical_state);
  
  memcpy(C1->current_spectral_state, C1->prior_draw, size);
  memcpy(C2->current_spectral_state, C2->prior_draw, size);
  
  fftw_destroy_plan(p);
  fftw_free(uk);
}

void infmcmc_proposeRWMH(mcmc_infchain *C) {
  int j, k;
  const int maxk = (C->nk >> 1) + 1;
  const int N = C->nj * C->nk;
  const double sqrtOneMinusBeta2 = sqrt(1.0 - C->rwmh_stepsize * C->rwmh_stepsize);
  double *u = (double *)malloc(sizeof(double) * C->nj * C->nk);
  fftw_complex *uk = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * C->nj * maxk);
  
  memcpy(u, C->current_physical_state, sizeof(double) * C->nj * C->nk);
  fftw_execute_dft_r2c(C->_r2c, u, C->current_spectral_state);
  
  // Draw from prior distribution
  randomPriorDraw(C);
  
  for(j = 0; j < C->nj; j++) {
    for(k = 0; k < maxk; k++) {
      C->proposed_spectral_state[maxk*j+k] = sqrtOneMinusBeta2 * C->current_spectral_state[maxk*j+k] + C->rwmh_stepsize * C->prior_draw[maxk*j+k];
      C->proposed_spectral_state[maxk*j+k] /= N;
    }
  }
  
  memcpy(uk, C->proposed_spectral_state, sizeof(fftw_complex) * C->nj * maxk);
  fftw_execute_dft_c2r(C->_c2r, uk, C->proposed_physical_state);
  fftw_free(uk);
  free(u);
}

void infmcmc_adaptRWMHStepSize(mcmc_infchain *C, double inc) {
  // Adapt to stay in 20-30% range.
  int adaptFreq = 100;
  double rate;
  
  if (C->current_iter > 0 && C->current_iter % adaptFreq == 0) {
    rate = (double) C->_short_time_acc_prob_avg / adaptFreq;
    
    if (rate < 0.2) {
      //C->_bHigh = C->rwmh_stepsize;
      //C->rwmh_stepsize = (C->_bLow + C->_bHigh) / 2.0;
      C->rwmh_stepsize -= inc;
    }
    else if (rate > 0.3) {
      //C->_bLow = C->rwmh_stepsize;
      //C->rwmh_stepsize = (C->_bLow + C->_bHigh) / 2.0;
      C->rwmh_stepsize += inc;
    }
    
    C->_short_time_acc_prob_avg = 0.0;
  }
  else {
    C->_short_time_acc_prob_avg += C->acc_prob;
  }
}

void infmcmc_proposeDivFreeRWMH(mcmc_infchain *C1, mcmc_infchain *C2) {
  int j, k;
  const int maxk = (C1->nk >> 1) + 1;
  const int N = C1->nj * C1->nk;
  const double sqrtOneMinusBeta2 = sqrt(1.0 - C1->rwmh_stepsize * C1->rwmh_stepsize);
  double *u = (double *)malloc(sizeof(double) * C1->nj * C1->nk);
  fftw_complex *uk = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * C1->nj * maxk);
  
  memcpy(u, C1->current_physical_state, sizeof(double) * C1->nj * C1->nk);
  fftw_execute_dft_r2c(C1->_r2c, u, C1->current_spectral_state);
  
  memcpy(u, C2->current_physical_state, sizeof(double) * C2->nj * C2->nk);
  fftw_execute_dft_r2c(C2->_r2c, u, C2->current_spectral_state);
    
  // Draw from prior distribution
  randomDivFreePriorDraw(C1, C2);
  
  for(j = 0; j < C1->nj; j++) {
    for(k = 0; k < maxk; k++) {
      C1->current_spectral_state[maxk*j+k] /= N;
      C2->current_spectral_state[maxk*j+k] /= N;
      C1->proposed_spectral_state[maxk*j+k] = sqrtOneMinusBeta2 * C1->current_spectral_state[maxk*j+k] + C1->rwmh_stepsize * C1->prior_draw[maxk*j+k];
      //C1->proposed_spectral_state[maxk*j+k] /= N;
      C2->proposed_spectral_state[maxk*j+k] = sqrtOneMinusBeta2 * C2->current_spectral_state[maxk*j+k] + C2->rwmh_stepsize * C2->prior_draw[maxk*j+k];
      //C2->proposed_spectral_state[maxk*j+k] /= N;
    }
  }
  
  memcpy(uk, C1->proposed_spectral_state, sizeof(fftw_complex) * C1->nj * maxk);
  fftw_execute_dft_c2r(C1->_c2r, uk, C1->proposed_physical_state);
  memcpy(uk, C2->proposed_spectral_state, sizeof(fftw_complex) * C1->nj * maxk);
  fftw_execute_dft_c2r(C2->_c2r, uk, C2->proposed_physical_state);
  
  fftw_free(uk);
  free(u);
}

void infmcmc_updateAvgs(mcmc_infchain *C) {
  int j, k;
  double deltar;
  fftw_complex deltac;
  C->current_iter++;
  
  /*
   Update physical vectors
   */
  if (C->current_iter == 1) {
    for (j = 0; j < C->nj; j++) {
      for (k = 0; k < C->nk; k++) {
        deltar = C->current_physical_state[C->nk * j + k] - C->avg_physical_state[C->nk * j + k];
        C->avg_physical_state[C->nk * j + k] += (deltar / C->current_iter);
        C->_M2[C->nk * j + k] += (deltar * (C->current_physical_state[C->nk * j + k] - C->avg_physical_state[C->nk * j + k]));
        C->var_physical_state[C->nk * j + k] = -1.0;
      }
    }
  }
  else {
    for (j = 0; j < C->nj; j++) {
      for (k = 0; k < C->nk; k++) {
        deltar = C->current_physical_state[C->nk * j + k] - C->avg_physical_state[C->nk * j + k];
        C->avg_physical_state[C->nk * j + k] += (deltar / C->current_iter);
        C->_M2[C->nk * j + k] += (deltar * (C->current_physical_state[C->nk * j + k] - C->avg_physical_state[C->nk * j + k]));
        C->var_physical_state[C->nk * j + k] = C->_M2[C->nk * j + k] / (C->current_iter - 1);
      }
    }
  }
  
  /*
   Update spectral vectors
  */
  if (C->current_iter == 1) {
    for (j = 0; j < C->nj; j++) {
      for (k = 0; k < C->nk/2 + 1; k++) {
        deltac = C->current_spectral_state[(C->nk/2 + 1) * j + k] - C->avg_spectral_state[(C->nk/2 + 1) * j + k];
        C->avg_spectral_state[(C->nk/2 + 1) * j + k] += (deltac / C->current_iter);
      }
    }
  }
  else {
    for (j = 0; j < C->nj; j++) {
      for (k = 0; k < C->nk/2 + 1; k++) {
        deltac = C->current_spectral_state[(C->nk/2 + 1) * j + k] - C->avg_spectral_state[(C->nk/2 + 1) * j + k];
        C->avg_spectral_state[(C->nk/2 + 1) * j + k] += (deltac / C->current_iter);
      }
    }
  }
  
  /*
   Update scalars
   */
  C->avg_acc_prob += ((C->acc_prob - C->avg_acc_prob) / C->current_iter);
}

void infmcmc_updateRWMH(mcmc_infchain *C, double logLHDOfProposal) {
  double alpha;
  
  alpha = exp(C->log_likelihood_current_state - logLHDOfProposal);
  
  if (alpha > 1.0) {
    alpha = 1.0;
  }
  // fixme: set acc prob here instead
  if (gsl_rng_uniform(C->r) < alpha) {
    memcpy(C->current_spectral_state, C->proposed_spectral_state, sizeof(fftw_complex) * C->nj * ((C->nk >> 1) + 1));
    memcpy(C->current_physical_state, C->proposed_physical_state, sizeof(double) * C->nj * C->nk);
    C->acc_prob = alpha;
    C->log_likelihood_current_state = logLHDOfProposal;
  }
  
  infmcmc_updateAvgs(C);
}

void infmcmc_updateVectorFieldRWMH(mcmc_infchain *C1, mcmc_infchain *C2, double logLHDOfProposal) {
  double alpha;
  
  // log likelihoods will be the same for both chains
	alpha = exp(C1->log_likelihood_current_state - logLHDOfProposal);
	
  if (alpha > 1.0) {
    alpha = 1.0;
  }
  
  C1->accepted = 0;
  C2->accepted = 0;
  C1->acc_prob = alpha;
  C2->acc_prob = alpha;
  
  if (gsl_rng_uniform(C1->r) < alpha) {
    C1->accepted = 1;
    C2->accepted = 1;
    memcpy(C1->current_spectral_state, C1->proposed_spectral_state, sizeof(fftw_complex) * C1->nj * ((C1->nk >> 1) + 1));
    memcpy(C1->current_physical_state, C1->proposed_physical_state, sizeof(double) * C1->nj * C1->nk);
    C1->log_likelihood_current_state = logLHDOfProposal;
    memcpy(C2->current_spectral_state, C2->proposed_spectral_state, sizeof(fftw_complex) * C2->nj * ((C2->nk >> 1) + 1));
    memcpy(C2->current_physical_state, C2->proposed_physical_state, sizeof(double) * C2->nj * C2->nk);
    C2->log_likelihood_current_state = logLHDOfProposal;
  }
  
  infmcmc_updateAvgs(C1);
  infmcmc_updateAvgs(C2);
}

void infmcmc_setRWMHStepSize(mcmc_infchain *C, double beta) {
  C->rwmh_stepsize = beta;
}

void infmcmc_setPriorAlpha(mcmc_infchain *C, double alpha) {
  C->alpha_prior = alpha;
}

void infmcmc_setPriorVar(mcmc_infchain *C, double var) {
  C->prior_var = var;
  C->prior_std = sqrt(var);
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

double infmcmc_L2Current(mcmc_infchain *C) {
  return L2Field(C->current_spectral_state, C->nj, C->nk);
}

double infmcmc_L2Proposed(mcmc_infchain *C) {
  return L2Field(C->proposed_spectral_state, C->nj, C->nk);
}

double infmcmc_L2Prior(mcmc_infchain *C) {
  return L2Field(C->prior_draw, C->nj, C->nk);
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

void setRWMHStepSize(mcmc_infchain *C, double stepSize) {
  C->rwmh_stepsize = stepSize;
}

void resetAvgs(mcmc_infchain *C) {
  /*
    Sets avg_physical_state to current_physical_state and
         avg_spectral_state to current_spectral_state
  */
  const size_t size_doublenj = sizeof(double) * C->nj;
  memcpy(C->avg_physical_state, C->current_physical_state, size_doublenj * C->nk);
  memcpy(C->avg_spectral_state, C->current_spectral_state, size_doublenj * ((C->nk >> 1) + 1));
}

void resetVar(mcmc_infchain *C) {
  // Sets var_physical_state to 0
  memset(C->var_physical_state, 0, sizeof(double) * C->nj * C->nk);
}

void proposeIndependence(mcmc_infchain *C) {
  const int maxk = (C->nk >> 1) + 1;
  
  memcpy(C->proposed_spectral_state, C->prior_draw, sizeof(fftw_complex) * C->nj * maxk);
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

void acceptReject(mcmc_infchain *C) {
  const double phi1 = lsqFunctional(C->data, C->current_state_observations, C->sizeObsVector, C->obs_std_dev);
  const double phi2 = lsqFunctional(C->data, C->proposed_state_observations, C->sizeObsVector, C->obs_std_dev);
  double tempAccProb = exp(phi1 - phi2);
  
  if(tempAccProb > 1.0) {
    tempAccProb = 1.0;
  }
  
  if(gsl_rng_uniform(C->r) < tempAccProb) {
    memcpy(C->current_spectral_state, C->proposed_spectral_state, sizeof(fftw_complex) * C->nj * ((C->nk >> 1) + 1));
    memcpy(C->current_physical_state, C->proposed_physical_state, sizeof(double) * C->nj * C->nk);
    memcpy(C->current_state_observations, C->proposed_state_observations, sizeof(double) * C->sizeObsVector);
    
    C->acc_prob = tempAccProb;
    C->current_LSQFunctional = phi2;
  }
  else {
    C->current_LSQFunctional = phi1;
  }
}

//
//void updateChain(mcmc_infchain *C) {
//C->currentIter++;
//void updateAvgs(mcmc_infchain *C);
//void updateVar(mcmc_infchain *C);
/*
int main(void) {
  int nj = 32, nk = 32, sizeObsVector = 0;
  unsigned long int randseed = 0;
  
  FILE *fp;
  mcmc_infchain *C;
  
  C = (mcmc_infchain *)malloc(sizeof(mcmc_infchain));
  
  initChain(C, nj, nk, sizeObsVector, randseed);
  
  randomPriorDraw2(C);
  
  fp = fopen("2.dat", "w");
  fwrite(C->priorDraw, sizeof(fftw_complex), nj * (nk / 2 + 1), fp);
  fclose(fp);
  
  return 0;
}
*/
