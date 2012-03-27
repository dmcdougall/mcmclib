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

void infmcmc_initChain(INFCHAIN *C, const int nj, const int nk) {
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
  C->currentIter = 0;
  C->accepted = 0;
  C->_shortTimeAccProbAvg = 0.0;
  C->_bLow = 0.0;
  C->_bHigh = 1.0;
  
  // Allocate a ton of memory
  C->currentPhysicalState      = (double *)malloc(sphysical);
  C->avgPhysicalState          = (double *)malloc(sphysical);
  C->varPhysicalState          = (double *)malloc(sphysical);
  C->proposedPhysicalState     = (double *)malloc(sphysical);
  C->_M2                       = (double *)malloc(sphysical);
  //C->currentStateObservations  = (double *)malloc(obsVecMem);
  //C->proposedStateObservations = (double *)malloc(obsVecMem);
  //C->data                      = (double *)malloc(obsVecMem);
  
  C->currentSpectralState  = (fftw_complex *)fftw_malloc(sspectral);
  C->avgSpectralState      = (fftw_complex *)fftw_malloc(sspectral);
  C->priorDraw             = (fftw_complex *)fftw_malloc(sspectral);
  C->proposedSpectralState = (fftw_complex *)fftw_malloc(sspectral);
  
  memset(C->currentPhysicalState,  0, sphysical);
  memset(C->avgPhysicalState,      0, sphysical);
  memset(C->varPhysicalState,      0, sphysical);
  memset(C->proposedPhysicalState, 0, sphysical);
  memset(C->_M2,                   0, sphysical);
  memset(C->currentSpectralState,  0, sspectral);
  memset(C->avgSpectralState,      0, sspectral);
  memset(C->proposedSpectralState, 0, sspectral);
  
  C->accProb = 0.0;
  C->avgAccProb = 0.0;
  C->logLHDCurrentState = 0.0;
  
  /*
   * Set some default values
   */
  C->alphaPrior = 3.0;
  C->rwmhStepSize = 1e-4;
  C->priorVar = 1.0;
  C->priorStd = 1.0;
  
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
  
  C->_c2r = fftw_plan_dft_c2r_2d(nj, nk, C->proposedSpectralState, C->proposedPhysicalState, FFTW_MEASURE);
  C->_r2c = fftw_plan_dft_r2c_2d(nj, nk, C->currentPhysicalState, C->currentSpectralState, FFTW_MEASURE);
}

void infmcmc_freeChain(INFCHAIN *C) {
  // Free all allocated memory used by the chain
  free(C->currentPhysicalState);
  free(C->avgPhysicalState);
  free(C->varPhysicalState);
  free(C->proposedPhysicalState);
  free(C->_M2);
  //free(C->currentStateObservations);
  //free(C->proposedStateObservations);
  
  fftw_free(C->currentSpectralState);
  fftw_free(C->avgSpectralState);
  fftw_free(C->priorDraw);
  fftw_free(C->proposedSpectralState);
  
  gsl_rng_free(C->r);
}

void infmcmc_writeChain(const INFCHAIN *C, FILE *fp) {
  const int s = C->nj * C->nk;
  
  fwrite(&(C->nj),                 sizeof(int),    1,         fp);
  fwrite(&(C->nk),                 sizeof(int),    1,         fp);
  fwrite(&(C->currentIter),        sizeof(int),    1,         fp);
  fwrite(C->currentPhysicalState,  sizeof(double), s,         fp);
  fwrite(C->avgPhysicalState,      sizeof(double), s,         fp);
  fwrite(C->varPhysicalState,      sizeof(double), s,         fp);
  fwrite(&(C->logLHDCurrentState), sizeof(double), 1,         fp);
  fwrite(&(C->accProb),            sizeof(double), 1,         fp);
  fwrite(&(C->avgAccProb),         sizeof(double), 1,         fp);
}

void infmcmc_writeChainInfo(const INFCHAIN *C, FILE *fp) {
  fwrite(&(C->nj),             sizeof(int), 1, fp);
  fwrite(&(C->nk),             sizeof(int), 1, fp);
}

void infmcmc_writeVFChain(const INFCHAIN *U, const INFCHAIN *V, FILE *fp) {
  const int s = U->nj * U->nk;
  
  fwrite(U->currentPhysicalState,  sizeof(double), s,         fp);
  fwrite(U->avgPhysicalState,      sizeof(double), s,         fp);
  fwrite(U->varPhysicalState,      sizeof(double), s,         fp);
  fwrite(V->currentPhysicalState,  sizeof(double), s,         fp);
  fwrite(V->avgPhysicalState,      sizeof(double), s,         fp);
  fwrite(V->varPhysicalState,      sizeof(double), s,         fp);
  fwrite(&(U->logLHDCurrentState), sizeof(double), 1,         fp);
  fwrite(&(U->accProb),            sizeof(double), 1,         fp);
  fwrite(&(U->avgAccProb),         sizeof(double), 1,         fp);
}

void infmcmc_printChain(INFCHAIN *C) {
  printf("Iteration %d\n", C->currentIter);
  printf("-- Length is         %d x %d\n", C->nj, C->nk);
  printf("-- llhd val is       %lf\n", C->logLHDCurrentState);
  printf("-- Acc. prob is      %.10lf\n", C->accProb);
  printf("-- Avg. acc. prob is %.10lf\n", C->avgAccProb);
  printf("-- Beta is           %.10lf\n\n", C->rwmhStepSize);
  //finmcmc_printCurrentState(C);
  //finmcmc_printAvgState(C);
  //finmcmc_printVarState(C);
}

void randomPriorDraw(INFCHAIN *C) {
  int j, k;
  const int maxk = (C->nk >> 1) + 1;
  const int nko2 = C->nk >> 1;
  const int njo2 = C->nj >> 1;
  double xrand, yrand, c;
  //const double one = 1.0;
  
  c = 4.0 * M_PI * M_PI;
  
  for(j = 0; j < C->nj; j++) {
    for(k = 0; k < maxk; k++) {
      xrand = gsl_ran_gaussian_ziggurat(C->r, C->priorStd);
      if((j == 0) && (k == 0)) {
        C->priorDraw[0] = 0.0;
      }
      else if((j == njo2) && (k == nko2)) {
        C->priorDraw[maxk*njo2+nko2] = /*C->nj */ xrand / pow(c * ((j * j) + (k * k)), (double)C->alphaPrior/2.0);
      }
      else if((j == 0) && (k == nko2)) {
        C->priorDraw[nko2] = /*C->nj */ xrand / pow(c * ((j * j) + (k * k)), (double)C->alphaPrior/2.0);
      }
      else if((j == njo2) && (k == 0)) {
        C->priorDraw[maxk*njo2] = /*C->nj */ xrand / pow(c * ((j * j) + (k * k)), (double)C->alphaPrior/2.0);
      }
      else {
        xrand /= sqrt(2.0);
        yrand = gsl_ran_gaussian_ziggurat(C->r, C->priorStd) / sqrt(2.0);
        C->priorDraw[maxk*j+k] = /*C->nj */ (xrand + I * yrand) / pow(c * ((j * j) + (k * k)), (double)C->alphaPrior/2.0);
        if(j > njo2) {
          C->priorDraw[maxk*j+k] = conj(C->priorDraw[maxk*(C->nj-j)+k]);
        }
      }
    }
  }
}

void randomDivFreePriorDraw(INFCHAIN *C1, INFCHAIN *C2) {
  int j, k;
  const int maxk = (C1->nk >> 1) + 1;
  const int njo2 = C2->nj >> 1;
  const int nko2 = C1->nk >> 1;
  double modk;
  
  randomPriorDraw(C1);
  
  for (j = 0; j < C1->nj; j++) {
    for (k = 0; k < maxk; k++) {
      if (j == 0 && k == 0) {
        C1->priorDraw[0] = 0.0;
        C2->priorDraw[0] = 0.0;
        continue;
      }
      
      modk = sqrt(j * j + k * k);
      if (j < njo2) {
        C2->priorDraw[maxk*j+k] = -j * C1->priorDraw[maxk*j+k] / modk;
      }
      else if (j > njo2) {
        C2->priorDraw[maxk*j+k] = -(j - C1->nj) * C1->priorDraw[maxk*j+k] / modk;
      }
      else {
        C2->priorDraw[maxk*j+k] = 0.0;
      }
      
      if (k < nko2) {
        C1->priorDraw[maxk*j+k] *= k / modk;
      }
      else {
        C1->priorDraw[maxk*j+k] = 0.0;
      }
    }
  }
}

void infmcmc_seedWithPriorDraw(INFCHAIN *C) {
  const int size = sizeof(fftw_complex) * C->nj * ((C->nk >> 1) + 1);
  fftw_complex *uk = (fftw_complex *)fftw_malloc(size);
  const fftw_plan p = fftw_plan_dft_c2r_2d(C->nj, C->nk, uk, C->currentPhysicalState, FFTW_ESTIMATE);
  
  randomPriorDraw(C);
  memcpy(uk, C->priorDraw, size);
  
  //fixme: put into current spectral state
  
  fftw_execute(p);
  fftw_destroy_plan(p);
  fftw_free(uk);
}

void infmcmc_seedWithDivFreePriorDraw(INFCHAIN *C1, INFCHAIN *C2) {
  const int size = sizeof(fftw_complex) * C1->nj * ((C1->nk >> 1) + 1);
  fftw_complex *uk = (fftw_complex *)fftw_malloc(size);
  const fftw_plan p = fftw_plan_dft_c2r_2d(C1->nj, C1->nk, uk, C1->currentPhysicalState, FFTW_ESTIMATE);
  
  randomDivFreePriorDraw(C1, C2);
  
  memcpy(uk, C1->priorDraw, size);
  fftw_execute_dft_c2r(p, uk, C1->currentPhysicalState);
  
  memcpy(uk, C2->priorDraw, size);
  fftw_execute_dft_c2r(p, uk, C2->currentPhysicalState);
  
  memcpy(C1->currentSpectralState, C1->priorDraw, size);
  memcpy(C2->currentSpectralState, C2->priorDraw, size);
  
  fftw_destroy_plan(p);
  fftw_free(uk);
}

void infmcmc_proposeRWMH(INFCHAIN *C) {
  int j, k;
  const int maxk = (C->nk >> 1) + 1;
  const int N = C->nj * C->nk;
  const double sqrtOneMinusBeta2 = sqrt(1.0 - C->rwmhStepSize * C->rwmhStepSize);
  double *u = (double *)malloc(sizeof(double) * C->nj * C->nk);
  fftw_complex *uk = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * C->nj * maxk);
  
  memcpy(u, C->currentPhysicalState, sizeof(double) * C->nj * C->nk);
  fftw_execute_dft_r2c(C->_r2c, u, C->currentSpectralState);
  
  // Draw from prior distribution
  randomPriorDraw(C);
  
  for(j = 0; j < C->nj; j++) {
    for(k = 0; k < maxk; k++) {
      C->proposedSpectralState[maxk*j+k] = sqrtOneMinusBeta2 * C->currentSpectralState[maxk*j+k] + C->rwmhStepSize * C->priorDraw[maxk*j+k];
      C->proposedSpectralState[maxk*j+k] /= N;
    }
  }
  
  memcpy(uk, C->proposedSpectralState, sizeof(fftw_complex) * C->nj * maxk);
  fftw_execute_dft_c2r(C->_c2r, uk, C->proposedPhysicalState);
  fftw_free(uk);
  free(u);
}

void infmcmc_adaptRWMHStepSize(INFCHAIN *C, double inc) {
  // Adapt to stay in 20-30% range.
  int adaptFreq = 100;
  double rate;
  
  if (C->currentIter > 0 && C->currentIter % adaptFreq == 0) {
    rate = (double) C->_shortTimeAccProbAvg / adaptFreq;
    
    if (rate < 0.2) {
      //C->_bHigh = C->rwmhStepSize;
      //C->rwmhStepSize = (C->_bLow + C->_bHigh) / 2.0;
      C->rwmhStepSize -= inc;
    }
    else if (rate > 0.3) {
      //C->_bLow = C->rwmhStepSize;
      //C->rwmhStepSize = (C->_bLow + C->_bHigh) / 2.0;
      C->rwmhStepSize += inc;
    }
    
    C->_shortTimeAccProbAvg = 0.0;
  }
  else {
    C->_shortTimeAccProbAvg += C->accProb;
  }
}

void infmcmc_proposeDivFreeRWMH(INFCHAIN *C1, INFCHAIN *C2) {
  int j, k;
  const int maxk = (C1->nk >> 1) + 1;
  const int N = C1->nj * C1->nk;
  const double sqrtOneMinusBeta2 = sqrt(1.0 - C1->rwmhStepSize * C1->rwmhStepSize);
  double *u = (double *)malloc(sizeof(double) * C1->nj * C1->nk);
  fftw_complex *uk = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * C1->nj * maxk);
  
  memcpy(u, C1->currentPhysicalState, sizeof(double) * C1->nj * C1->nk);
  fftw_execute_dft_r2c(C1->_r2c, u, C1->currentSpectralState);
  
  memcpy(u, C2->currentPhysicalState, sizeof(double) * C2->nj * C2->nk);
  fftw_execute_dft_r2c(C2->_r2c, u, C2->currentSpectralState);
    
  // Draw from prior distribution
  randomDivFreePriorDraw(C1, C2);
  
  for(j = 0; j < C1->nj; j++) {
    for(k = 0; k < maxk; k++) {
      C1->currentSpectralState[maxk*j+k] /= N;
      C2->currentSpectralState[maxk*j+k] /= N;
      C1->proposedSpectralState[maxk*j+k] = sqrtOneMinusBeta2 * C1->currentSpectralState[maxk*j+k] + C1->rwmhStepSize * C1->priorDraw[maxk*j+k];
      //C1->proposedSpectralState[maxk*j+k] /= N;
      C2->proposedSpectralState[maxk*j+k] = sqrtOneMinusBeta2 * C2->currentSpectralState[maxk*j+k] + C2->rwmhStepSize * C2->priorDraw[maxk*j+k];
      //C2->proposedSpectralState[maxk*j+k] /= N;
    }
  }
  
  memcpy(uk, C1->proposedSpectralState, sizeof(fftw_complex) * C1->nj * maxk);
  fftw_execute_dft_c2r(C1->_c2r, uk, C1->proposedPhysicalState);
  memcpy(uk, C2->proposedSpectralState, sizeof(fftw_complex) * C1->nj * maxk);
  fftw_execute_dft_c2r(C2->_c2r, uk, C2->proposedPhysicalState);
  
  fftw_free(uk);
  free(u);
}

void infmcmc_updateAvgs(INFCHAIN *C) {
  int j, k;
  double deltar;
  fftw_complex deltac;
  C->currentIter++;
  
  /*
   Update physical vectors
   */
  if (C->currentIter == 1) {
    for (j = 0; j < C->nj; j++) {
      for (k = 0; k < C->nk; k++) {
        deltar = C->currentPhysicalState[C->nk * j + k] - C->avgPhysicalState[C->nk * j + k];
        C->avgPhysicalState[C->nk * j + k] += (deltar / C->currentIter);
        C->_M2[C->nk * j + k] += (deltar * (C->currentPhysicalState[C->nk * j + k] - C->avgPhysicalState[C->nk * j + k]));
        C->varPhysicalState[C->nk * j + k] = -1.0;
      }
    }
  }
  else {
    for (j = 0; j < C->nj; j++) {
      for (k = 0; k < C->nk; k++) {
        deltar = C->currentPhysicalState[C->nk * j + k] - C->avgPhysicalState[C->nk * j + k];
        C->avgPhysicalState[C->nk * j + k] += (deltar / C->currentIter);
        C->_M2[C->nk * j + k] += (deltar * (C->currentPhysicalState[C->nk * j + k] - C->avgPhysicalState[C->nk * j + k]));
        C->varPhysicalState[C->nk * j + k] = C->_M2[C->nk * j + k] / (C->currentIter - 1);
      }
    }
  }
  
  /*
   Update spectral vectors
  */
  if (C->currentIter == 1) {
    for (j = 0; j < C->nj; j++) {
      for (k = 0; k < C->nk/2 + 1; k++) {
        deltac = C->currentSpectralState[(C->nk/2 + 1) * j + k] - C->avgSpectralState[(C->nk/2 + 1) * j + k];
        C->avgSpectralState[(C->nk/2 + 1) * j + k] += (deltac / C->currentIter);
      }
    }
  }
  else {
    for (j = 0; j < C->nj; j++) {
      for (k = 0; k < C->nk/2 + 1; k++) {
        deltac = C->currentSpectralState[(C->nk/2 + 1) * j + k] - C->avgSpectralState[(C->nk/2 + 1) * j + k];
        C->avgSpectralState[(C->nk/2 + 1) * j + k] += (deltac / C->currentIter);
      }
    }
  }
  
  /*
   Update scalars
   */
  C->avgAccProb += ((C->accProb - C->avgAccProb) / C->currentIter);
}

void infmcmc_updateRWMH(INFCHAIN *C, double logLHDOfProposal) {
  double alpha;
  
  alpha = exp(C->logLHDCurrentState - logLHDOfProposal);
  
  if (alpha > 1.0) {
    alpha = 1.0;
  }
  // fixme: set acc prob here instead
  if (gsl_rng_uniform(C->r) < alpha) {
    memcpy(C->currentSpectralState, C->proposedSpectralState, sizeof(fftw_complex) * C->nj * ((C->nk >> 1) + 1));
    memcpy(C->currentPhysicalState, C->proposedPhysicalState, sizeof(double) * C->nj * C->nk);
    C->accProb = alpha;
    C->logLHDCurrentState = logLHDOfProposal;
  }
  
  infmcmc_updateAvgs(C);
}

void infmcmc_updateVectorFieldRWMH(INFCHAIN *C1, INFCHAIN *C2, double logLHDOfProposal) {
  double alpha;
  
  // log likelihoods will be the same for both chains
	alpha = exp(C1->logLHDCurrentState - logLHDOfProposal);
	
  if (alpha > 1.0) {
    alpha = 1.0;
  }
  
  C1->accepted = 0;
  C2->accepted = 0;
  C1->accProb = alpha;
  C2->accProb = alpha;
  
  if (gsl_rng_uniform(C1->r) < alpha) {
    C1->accepted = 1;
    C2->accepted = 1;
    memcpy(C1->currentSpectralState, C1->proposedSpectralState, sizeof(fftw_complex) * C1->nj * ((C1->nk >> 1) + 1));
    memcpy(C1->currentPhysicalState, C1->proposedPhysicalState, sizeof(double) * C1->nj * C1->nk);
    C1->logLHDCurrentState = logLHDOfProposal;
    memcpy(C2->currentSpectralState, C2->proposedSpectralState, sizeof(fftw_complex) * C2->nj * ((C2->nk >> 1) + 1));
    memcpy(C2->currentPhysicalState, C2->proposedPhysicalState, sizeof(double) * C2->nj * C2->nk);
    C2->logLHDCurrentState = logLHDOfProposal;
  }
  
  infmcmc_updateAvgs(C1);
  infmcmc_updateAvgs(C2);
}

void infmcmc_setRWMHStepSize(INFCHAIN *C, double beta) {
  C->rwmhStepSize = beta;
}

void infmcmc_setPriorAlpha(INFCHAIN *C, double alpha) {
  C->alphaPrior = alpha;
}

void infmcmc_setPriorVar(INFCHAIN *C, double var) {
  C->priorVar = var;
  C->priorStd = sqrt(var);
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

double infmcmc_L2Current(INFCHAIN *C) {
  return L2Field(C->currentSpectralState, C->nj, C->nk);
}

double infmcmc_L2Proposed(INFCHAIN *C) {
  return L2Field(C->proposedSpectralState, C->nj, C->nk);
}

double infmcmc_L2Prior(INFCHAIN *C) {
  return L2Field(C->priorDraw, C->nj, C->nk);
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

void setRWMHStepSize(CHAIN *C, double stepSize) {
  C->rwmhStepSize = stepSize;
}

void resetAvgs(CHAIN *C) {
  /*
    Sets avgPhysicalState to currentPhysicalState and
         avgSpectralState to currentSpectralState
  */
  const size_t size_doublenj = sizeof(double) * C->nj;
  memcpy(C->avgPhysicalState, C->currentPhysicalState, size_doublenj * C->nk);
  memcpy(C->avgSpectralState, C->currentSpectralState, size_doublenj * ((C->nk >> 1) + 1));
}

void resetVar(CHAIN *C) {
  // Sets varPhysicalState to 0
  memset(C->varPhysicalState, 0, sizeof(double) * C->nj * C->nk);
}

void proposeIndependence(CHAIN *C) {
  const int maxk = (C->nk >> 1) + 1;
  
  memcpy(C->proposedSpectralState, C->priorDraw, sizeof(fftw_complex) * C->nj * maxk);
}

double lsqFunctional(const double * const data, const double * const obsVec, const int obsVecSize, const double obsStdDev) {
  int i;
  double temp1, sum1 = 0.0;
  
  for(i = 0; i < obsVecSize; i++) {
    temp1 = (data[i] - obsVec[i]);
    sum1 += temp1 * temp1;
  }
  
  return sum1 / (2.0 * obsStdDev * obsStdDev);
}

void acceptReject(CHAIN *C) {
  const double phi1 = lsqFunctional(C->data, C->currentStateObservations, C->sizeObsVector, C->obsStdDev);
  const double phi2 = lsqFunctional(C->data, C->proposedStateObservations, C->sizeObsVector, C->obsStdDev);
  double tempAccProb = exp(phi1 - phi2);
  
  if(tempAccProb > 1.0) {
    tempAccProb = 1.0;
  }
  
  if(gsl_rng_uniform(C->r) < tempAccProb) {
    memcpy(C->currentSpectralState, C->proposedSpectralState, sizeof(fftw_complex) * C->nj * ((C->nk >> 1) + 1));
    memcpy(C->currentPhysicalState, C->proposedPhysicalState, sizeof(double) * C->nj * C->nk);
    memcpy(C->currentStateObservations, C->proposedStateObservations, sizeof(double) * C->sizeObsVector);
    
    C->accProb = tempAccProb;
    C->currentLSQFunctional = phi2;
  }
  else {
    C->currentLSQFunctional = phi1;
  }
}

//
//void updateChain(CHAIN *C) {
//C->currentIter++;
//void updateAvgs(CHAIN *C);
//void updateVar(CHAIN *C);
/*
int main(void) {
  int nj = 32, nk = 32, sizeObsVector = 0;
  unsigned long int randseed = 0;
  
  FILE *fp;
  CHAIN *C;
  
  C = (CHAIN *)malloc(sizeof(CHAIN));
  
  initChain(C, nj, nk, sizeObsVector, randseed);
  
  randomPriorDraw2(C);
  
  fp = fopen("2.dat", "w");
  fwrite(C->priorDraw, sizeof(fftw_complex), nj * (nk / 2 + 1), fp);
  fclose(fp);
  
  return 0;
}
*/
