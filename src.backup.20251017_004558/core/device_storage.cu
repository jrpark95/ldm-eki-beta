// Device storage for global constant arrays
// These are allocated once in device memory and passed as pointers to kernels
// to avoid invalid device symbol errors in non-RDC mode

// REMOVED: d_flex_hgt - now allocated with cudaMalloc in LDM class
// REMOVED: d_T_const - now allocated with cudaMalloc in LDM class
