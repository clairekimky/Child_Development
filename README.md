# Child_Development

This paper uses data from the Longitudinal Study of Australian Children (LSAC). The dataset is freely available through the Australian Data Archive Dataverse platform. However, access is restricted to approved researchers affiliated with government, academic, or non-profit institutions. Therefore, I provide only the replication codes, not the raw data.

# Codes, Estimates, and Tables

1) Folder: "STATA"
   The file "clean.do" cleans the raw data. The file "panel.do" merges cross-sectional datasets across multiple periods to create a panel. The file "analysis_bp.do" constructs the analysis sample, generates descriptive statistics, and produces key empirical patterns motivating the structural model. The file "export.do" exports cleaned data as CSV files for structural estimation.

2) Folder: "LSAC/Julia/Code" contains all Julia scripts used for the structural model estimation.
   The file "compute_1st.jl" estimates production function parameters using GMM. The file "compute_2nd_fin.jl" estimates preference parameters using SMM and produces model fit and counterfactual results. The file "umax_fin.jl" solves the optimization problem of parents and children (e.g., study time, conditional allowances, educational investments) using backward induction and both analytical and numerical methods. The file "smm_fin.jl" computes simulated and sample moments and performs SMM estimation. The file "quantcpm.jl" quantifies the contribution of conditional allowances to child skill development by comparing the baseline model to a counterfactual where the allowance channel is shut down. The file "cpm_eval.jl" converts model-implied skill gains from conditional allowances into monetary equivalents in terms of educational investment and unconditional income transfers. The file "decompose.jl" differentiates non-adopters of conditional allowances between those facing financial barriers versus those with low preferences for child skills, identifying key drivers of adoption decisions.
