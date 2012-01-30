#include "finmcmc.h"

void finmcmc_initChain(FINCHAIN *C, int n) {
  FILE *fp;
  unsigned long int seed;
  
  // Set up variables
  C->length = n;
  C->currentIter = 0;
  
  // Allocate a ton of memory
  C->currentState      = (double *)malloc(sizeof(double) * n);
  C->avgState          = (double *)malloc(sizeof(double) * n);
  C->varState          = (double *)malloc(sizeof(double) * n);
  C->proposedState     = (double *)malloc(sizeof(double) * n);
  C->_M2               = (double *)malloc(sizeof(double) * n);
  
  memset(C->currentState,  0, sizeof(double) * n);
  memset(C->avgState,      0, sizeof(double) * n);
  memset(C->varState,      0, sizeof(double) * n);
  memset(C->proposedState, 0, sizeof(double) * n);
  memset(C->_M2,           0, sizeof(double) * n);
  
  C->accProb = 0.0;
  C->avgAccProb = 0.0;
  C->logLHDCurrentState = 0.0;
  
  C->r = gsl_rng_alloc(gsl_rng_taus2);
  
  fp = fopen("/dev/urandom", "rb");
  
  if (fp != NULL) {
    fread(&seed, sizeof(unsigned long int), 1, fp);
    gsl_rng_set(C->r, seed);
    fclose(fp);
  }
  else {
    gsl_rng_set(C->r, 0);
  }
}

void finmcmc_freeChain(FINCHAIN *C) {
  // Free all allocated memory used by the chain
  free(C->currentState);
  free(C->avgState);
  free(C->varState);
  free(C->proposedState);
  free(C->_M2);
  gsl_rng_free(C->r);
}

void finmcmc_printCurrentState(FINCHAIN C) {
  int i;
  
  printf("-- Current state is:\n");
  for (i = 0; i < C.length; i++) {
    printf("---- Comp. %d is %lf\n", i, C.currentState[i]);
  }
  printf("\n");
}

void finmcmc_printAvgState(FINCHAIN C) {
  int i;
  
  printf("-- Average state is:\n");
  for (i = 0; i < C.length; i++) {
    printf("---- Comp. %d is %lf\n", i, C.avgState[i]);
  }
  printf("\n");
}

void finmcmc_printVarState(FINCHAIN C) {
  int i;
  
  printf("-- Variance state is:\n");
  for (i = 0; i < C.length; i++) {
    printf("---- Comp. %d is %lf\n", i, C.varState[i]);
  }
  printf("\n");
}

void finmcmc_printChain(FINCHAIN C) {
  printf("Iteration %d\n", C.currentIter);
  printf("-- Length is         %d\n", C.length);
  printf("-- llhd val is       %lf\n", C.logLHDCurrentState);
  printf("-- Acc. prob is      %lf\n", C.accProb);
  printf("-- Avg. acc. prob is %lf\n\n", C.avgAccProb);
  finmcmc_printCurrentState(C);
  finmcmc_printAvgState(C);
  finmcmc_printVarState(C);
}

void finmcmc_writeChain(const FINCHAIN *C, FILE *fp) {
  fwrite(&(C->length),             sizeof(int),    1,         fp);
  fwrite(&(C->currentIter),        sizeof(int),    1,         fp);
  fwrite(C->currentState,          sizeof(double), C->length, fp);
  fwrite(C->avgState,              sizeof(double), C->length, fp);
  fwrite(C->varState,              sizeof(double), C->length, fp);
  fwrite(&(C->logLHDCurrentState), sizeof(double), 1,         fp);
  fwrite(&(C->accProb),            sizeof(double), 1,         fp);
  fwrite(&(C->avgAccProb),         sizeof(double), 1,         fp);
}

void finmcmc_gausRanVec(const FINCHAIN *C, double *x, double stdDev) {
  int i;
  
  for (i = 0; i < C->length; i++) {
    x[i] = gsl_ran_gaussian(C->r, stdDev);
  }
}

void finmcmc_proposeRWMH(const FINCHAIN *C, double stdDev, double beta) {
  int i;
  const double sq1mbeta2 = sqrt(1.0 - (beta * beta));
  
  for (i = 0; i < C->length; i++) {
    C->proposedState[i] = (sq1mbeta2 * C->currentState[i]) + (beta * gsl_ran_gaussian(C->r, stdDev));
  }
}

/*
  Assumes non-average quantities have been updated
*/
void finmcmc_updateAvgs(FINCHAIN *C) {
  int i;
  double delta;
  
  C->currentIter++;
  
  /*
    Update vectors
  */
  if (C->currentIter == 1) {
    for (i = 0; i < C->length; i++) {
      delta = C->currentState[i] - C->avgState[i];
      C->avgState[i] += (delta / C->currentIter);
      C->_M2[i] += (delta * (C->currentState[i] - C->avgState[i]));
      C->varState[i] = -1.0;
    }
  }
  else {
    for (i = 0; i < C->length; i++) {
      delta = C->currentState[i] - C->avgState[i];
      C->avgState[i] += (delta / C->currentIter);
      C->_M2[i] += (delta * (C->currentState[i] - C->avgState[i]));
      C->varState[i] = C->_M2[i] / (C->currentIter - 1);
    }
  }
  
  /*
    Update scalars
  */
  C->avgAccProb += ((C->accProb - C->avgAccProb) / C->currentIter);
}

void finmcmc_updateRWMH(FINCHAIN *C, double logLHDOfProposal) {
  double alpha;
  
  alpha = exp(C->logLHDCurrentState - logLHDOfProposal);
  
  if (alpha > 1.0) {
    alpha = 1.0;
  }
  
  if (gsl_rng_uniform(C->r) < alpha) {
    memcpy(C->currentState, C->proposedState, sizeof(double) * C->length);
    C->accProb = alpha;
    C->logLHDCurrentState = logLHDOfProposal;
  }
  
  finmcmc_updateAvgs(C);
}

/*
void setRWMHStepSize(CHAIN *C, double stepSize) {
  C->rwmhStepSize = stepSize;
}

void resetAvgs(CHAIN *C) {
  //
    //Sets avgPhysicalState to currentPhysicalState and
      //   avgSpectralState to currentSpectralState
  //
  const size_t size_doublenj = sizeof(double) * C->nj;
  memcpy(C->avgPhysicalState, C->currentPhysicalState, size_doublenj * C->nk);
  memcpy(C->avgSpectralState, C->currentSpectralState, size_doublenj * ((C->nk >> 2) + 1));
}

void resetVar(CHAIN *C) {
  // Sets varPhysicalState to 0
  memset(C->varPhysicalState, 0, sizeof(double) * C->nj * C->nk);
}

void proposeRWMH(CHAIN *C) {
  int j, k;
  const int maxk = (C->nk >> 2) + 1;
  const double sqrtOneMinusBeta2 = sqrt((1.0 - C->rwmhStepSize * C->rwmhStepSize));
  
  for(j = 0; j < C->nj; j++) {
    for(k = 0; k < maxk; k++) {
      C->proposedSpectralState[maxk*j+k] = sqrtOneMinusBeta2 * C->currentSpectralState[maxk*j+k] + C->rwmhStepSize * C->priorDraw[maxk*j+k];
    }
  }
}

void proposeIndependence(CHAIN *C) {
  const int maxk = (C->nk >> 2) + 1;
  
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
    memcpy(C->currentSpectralState, C->proposedSpectralState, sizeof(fftw_complex) * C->nj * ((C->nk >> 2) + 1));
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

int main(void) {
  int nj = 32, nk = 32, sizeObsVector = 0;
  unsigned long int randseed = 0;
  
  FILE *fp;
  CHAIN *C;
  
  C = (CHAIN *)malloc(sizeof(CHAIN));
  
  initChain(C, nj, nk, sizeObsVector, randseed);
  
  //randomPriorDraw2(C);
  
  //fp = fopen("2.dat", "w");
  //fwrite(C->priorDraw, sizeof(fftw_complex), nj * (nk / 2 + 1), fp);
  //fclose(fp);
  
  return 0;
}
*/