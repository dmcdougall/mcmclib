#include <CUnit/Basic.h>
#include "infmcmc.h"

void setter_test(void) {
  double var = 1.2;
  mcmc_infchain chain;
  mcmc_set_prior_var(&chain, var);
  CU_ASSERT(var == chain.prior_var);
}

int main(int argc, char *argv[]) {
  CU_pSuite pSuite = NULL;

  // Initialise CUnit registry
  if (CUE_SUCCESS != CU_initialize_registry()) {
    return CU_get_error();
  }

  // Add a suite to the registry
  pSuite = CU_add_suite("Suite 1", NULL, NULL);
  if (NULL == pSuite) {
    CU_cleanup_registry();
    return CU_get_error();
  }

  // Add the tests to the suite
  if (NULL == CU_add_test(pSuite, "Setter test", setter_test)) {
    CU_cleanup_registry();
    return CU_get_error();
  }

  // Run tests using basic interface
  CU_basic_set_mode(CU_BRM_VERBOSE);
  CU_basic_run_tests();
  CU_cleanup_registry();
  return CU_get_error();
}
