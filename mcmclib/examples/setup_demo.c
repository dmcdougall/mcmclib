#include <infmcmc.h>

int main(int argc, char **argv) {
  mcmc_infchain *chain;
  chain = (mcmc_infchain *)malloc(sizeof(mcmc_infchain));

  mcmc_init_infchain(chain, 32, 32);

  return 0;
}
