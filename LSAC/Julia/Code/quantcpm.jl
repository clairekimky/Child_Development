# ==============================================================================
# Project Title: The Dynamics of Parent-Child Interactions in Shaping Cognitive and
# Non-cognitive Development during Adolescence
# Author: Claire Kim
# Institution: University of Wisconsin-Madison
# Start Date: 09/30/2025
# Description:
#
# This script estimates a dynamic model of child development with a focus on
# parental incentive and self-investment on cognitive and non-cognitive traits
# using Simulated Method of Moments
# ==============================================================================


using Random
using LinearAlgebra
using Statistics
using DataFrames
using CSV  # Added this for loading the CSV
using Distributions

# Set "include" path
cd("/Users/clairekim/Desktop/Project/Child Development/LSAC/code/")

# ==============================================================================
# Counterfactual Experiments
# ------------------------------------------------------------------------------

# Value Functions using estimated parameters

using NLopt, StaticArrays

# ============================================================================
# STEP 1: Identify Never-Adopters
# ============================================================================
# A child is a "never adopter" if they never receive CPM in baseline
#never_adopters = [all(CPM_opt[i, :] .== 0) for i in 1:n_c]

# ============================================================================
# STEP 2: Counterfactual Function (revised adoption logic)
# ============================================================================
function value_func_cf_comparative(
    Z, R_bar, params1c, params, n_c, n_age, s, Time, Y, CT,
    β_p, β_c, φ, M_opt, CPM_opt, τ_opt, R_opt
)
    """
    Solve model when CPM is NOT AVAILABLE (forced off).
    Parents can only choose (c_p, M, PM) and child chooses τ freely.
    """
    # Storage
    CPM_baseline = copy(CPM_opt)
    τ_baseline   = copy(τ_opt)
    M_baseline   = copy(M_opt)
    PM_baseline  = copy(R_opt)

    CPM_noCPM = zeros(Int, n_c, n_age)
    τ_noCPM    = zeros(n_c, n_age)
    M_noCPM    = zeros(n_c, n_age)
    PM_noCPM   = zeros(n_c, n_age)
    
    ψcc_store = zeros(n_c, n_age)
    ψpc_store = zeros(n_c, n_age)
    
    μ = params[1:3]
    Σ = [
        (params[4])^2  params[8]  params[9];
        params[8]  (params[5])^2  params[7];
        params[9]  params[7]  (params[6])^2
    ] # Covariance matrix
    Σ = make_pos_def(Σ)
    L = cholesky(Symmetric(Σ)).L
    
    Threads.@threads for i in 1:n_c

        z_i = @view Z[:, i]

        ν = μ .+ L * z_i                  # ν uses current μ,Σ but same z_i

        # Random draw from preference parameters
        α_0 = 1 # Normalization
        α_1 = exp(ν[1])

        if hyp[3*(i-1)+1] >= 1.0 && hyp[3*(i-1)+1] <= 5.0 # Low Self-Regulation
            # First compute base normalized values
            λ_sum_base = 1 + exp(ν[2]) + exp(ν[3])
            λ_0_base = exp(ν[2]) / λ_sum_base
            λ_1_base = exp(ν[3]) / λ_sum_base
            λ_2_base = 1 / λ_sum_base
            
            # Now apply transformation and renormalize
            λ_1_tilde = (1 + params[11]) * λ_1_base
            #λ_1_tilde = (1 + params[10]) * λ_1_base
            scale = 1.0 / (λ_0_base + λ_1_tilde + λ_2_base)
            
            λ_0 = λ_0_base * scale
            λ_1 = λ_1_tilde * scale
            λ_2 = λ_2_base * scale
            
        elseif hyp[3*(i-1)+1] >= 6.0 && hyp[3*(i-1)+1] <= 11.0  # High Self-Regulation
            λ_sum = 1 + exp(ν[2]) + exp(ν[3])
            λ_0 = exp(ν[2]) / λ_sum
            λ_1 = exp(ν[3]) / λ_sum
            λ_2 = 1 / λ_sum
        end
        # Store the current values of alpha and lambda parameters
        λ_1_vec = λ_1
        λ_0_vec = λ_0
        λ_2_vec = λ_2

        for t = n_age:-1:1
            # Compute ψ weights (same as before)
            if t == n_age
                ψcc_store[i, t] = λ_0_vec * (1 - (β_c)^(t + 1)) / (1 - β_c) # ψ_{c, 4}^C
                ψpc_store[i, t] = (α_0*(1 - φ)*(1 - (β_p)^(t + 1)) / (1 - β_p)) + φ*ψcc_store[i, t] # ψ_{p, 4}^C
            elseif t == 1 || t == 2 # ψ_{c, 3}^C, ψ_{c, 3}^N, ψ_{p, 3}^C, ψ_{p, 3}^N, ψ_{c, 2}^C, ψ_{c, 2}^N, ψ_{p, 2}^C, ψ_{p, 2}^N
                ψcc_store[i, t] = λ_0_vec + β_c*(params1c[t*4+1]*ψcc_store[i, t+1]) # ψ_{c, T}^C
                ψpc_store[i, t] = (1-φ)*α_0 + φ*λ_0_vec + β_p*(params1c[t*4+1]*ψpc_store[i, t+1]) # ψ_{p, T + 1}^C
            end
            
            ψcc, ψpc = ψcc_store[i, t], ψpc_store[i, t]
            Y_it = Y[(i-1)*n_age+t]
            s_it = s[(i-1)*n_age+t]
            Rbar_it = R_bar[(i-1)*n_age+t]
            CT_it = CT[(i-1)*n_age+t]
            
            χ = (1-φ)*α_1 + φ*λ_2_vec + β_p*(ψpc*params1c[t*4-1])
            M_star = β_p*(ψpc*params1c[t*4-1])*(Y_it+CT_it) / χ # Educational Investment Goods
            R_star = max(0.0, ((φ*λ_2_vec*(Y_it+CT_it) / χ) - Rbar_it))

            A = β_c*(ψcc*params1c[t*4-2])
            #println("A: ", A)
            τ_star = A*(Time-s_it) / (λ_1_vec+A) # Closed-form optimal child effort (no CPM)
            
            # --- Scenario 2: No CPM Exists ---
            CPM_noCPM[i, t] = 0
            τ_noCPM[i, t] = τ_star
            M_noCPM[i, t] = M_star
            PM_noCPM[i, t] = R_star
        end
    end
    
    return Dict(
        :baseline => (CPM_baseline, τ_baseline, M_baseline, PM_baseline),
        :noCPM  => (CPM_noCPM, τ_noCPM, M_noCPM, PM_noCPM)
    )
end

# ============================================================================
# Simulate Skills Forward
# ============================================================================

function simulate_skills_comparative(results, params1c, n_c, n_age, df)
    """
    Simulate cognitive and non-cognitive skills using production functions
    Production function is: ln(Z_t) = f(ln(Z_{t-1}), ln(τ), ln(M))
    So Z_t = exp(f(...))
    """
    skills = Dict()
    
    for scen in [:baseline, :noCPM]
        CPM_scen, τ_scen, M_scen, PM_scen = results[scen][1], results[scen][2], results[scen][3], results[scen][4]
        
        # Initialize skill matrix
        ln_Z_C = zeros(n_c, n_age + 1)
        ln_Z_C[:, 1] .= log.(df[df[:, 2] .== 1, 10])

        # Flatten arrays
        sr_vec = repeat(df[1:n_c, 13], inner=n_age)
        
        # Accumulate skills forward
        for t = 1:n_age
            for i = 1:n_c
            # Only simulate counterfactual if child i did NOT adopt CPM in period t
                τ_it = τ_scen[i, t]
                M_it = M_scen[i, t]
                sr_it = reshape(df[:, 11], n_c, n_age)
                
                # Cognitive skill production function (in logs)
                # ln(Z_C[t+1]) = α₀ + α₁*ln(Z_C[t]) + α₂*ln(τ) + α₃*ln(M) + α₄*ln(Z_N[t])
                ln_Z_C[i, t+1] = params1c[t*4-3] * ln_Z_C[i, t] +         # lagged cognitive
                                params1c[t*4-2] * log(τ_it) +    # effort
                                params1c[t*4-1] * log(M_it) +    # investment
                                params1c[t*4]   * log(sr_it[i, t])          # lagged non-cognitive
            end
        end
        
        # Store all periods (including initial)
        skills[scen] = Dict(
            :cognitive => ln_Z_C       # n_c × (n_age + 1)
        )
    end
    
    return skills
end

# ============================================================================
# Compute Treatment Effects - REVISED VERSION
# ============================================================================

function compute_treatment_effects_by_age(skills, CPM_baseline; hyperactivity=nothing)
    n_c, n_age_plus1 = size(skills[:baseline][:cognitive])
    n_age = n_age_plus1 - 1  # number of transitions
    effects = Dict(
        :Period => 1:(n_age+1),  # include initial period
        :BaselineMean => zeros(n_age+1),
        :NoCPMMean => zeros(n_age+1),
        :Change => zeros(n_age+1),
        :Change_SD => zeros(n_age+1),
    )

    # Heterogeneity groups
    if hyperactivity !== nothing
        effects[:BaselineMean_low] = zeros(n_age+1)
        effects[:NoCPMMean_low]   = zeros(n_age+1)
        effects[:Change_low]      = zeros(n_age+1)
        effects[:Change_SD_low]   = zeros(n_age+1)

        effects[:BaselineMean_high] = zeros(n_age+1)
        effects[:NoCPMMean_high]   = zeros(n_age+1)
        effects[:Change_high]      = zeros(n_age+1)
        effects[:Change_SD_high]   = zeros(n_age+1)

        idx_low  = findall((hyperactivity .>= 1) .& (hyperactivity .<= 5))
        idx_high = findall((hyperactivity .>= 10) .& (hyperactivity .<= 11))
    end

    for t in 1:(n_age+1)
        ln_base = skills[:baseline][:cognitive][:, t]  # t-th column
        ln_no   = skills[:noCPM][:cognitive][:, t]

        μ_base = mean(ln_base)
        μ_no   = mean(ln_no)
        sd_base = std(ln_base)
        sd_base = sd_base == 0.0 ? eps() : sd_base

        effects[:BaselineMean][t] = μ_base
        effects[:NoCPMMean][t]   = μ_no
        effects[:Change][t]      = μ_base - μ_no
        effects[:Change_SD][t]   = (μ_base - μ_no) / sd_base

        if hyperactivity !== nothing
            # Low-SR
            μ_base_low = mean(ln_base[idx_low])
            μ_no_low   = mean(ln_no[idx_low])
            effects[:BaselineMean_low][t] = μ_base_low
            effects[:NoCPMMean_low][t]   = μ_no_low
            effects[:Change_low][t]      = μ_base_low - μ_no_low
            effects[:Change_SD_low][t]   = (μ_base_low - μ_no_low) / sd_base

            # High-SR
            μ_base_high = mean(ln_base[idx_high])
            μ_no_high   = mean(ln_no[idx_high])
            effects[:BaselineMean_high][t] = μ_base_high
            effects[:NoCPMMean_high][t]   = μ_no_high
            effects[:Change_high][t]      = μ_base_high - μ_no_high
            effects[:Change_SD_high][t]   = (μ_base_high - μ_no_high) / sd_base
        end
    end

    return effects
end
