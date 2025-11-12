
# ==============================================================================
# Project Title: The Dynamics of Parent-Child Interactions in Shaping Cognitive and
# Non-cognitive Development during Adolescence
# Author: Claire Kim
# Institution: University of Wisconsin-Madison
# Start Date: 03/03/2025
# Description: Compute Code
#
# This script estimates a dynamic model of child development with a focus on
# parental incentive and self-investment on cognitive and non-cognitive traits
# using Simulated Method of Moments
# ==============================================================================

# Load packages
using Random
using LinearAlgebra
using Statistics
using DataFrames
using CSV
using Distributions
using Optim

# Set "include" path
cd("/Users/clairekim/Desktop/Project/Child Development/LSAC/code/")

# Load data from the CSV file
df1 = CSV.File("cogparams.csv")
df1 = DataFrame(df1)

# ==============================================================================
# 1. Model Estimation
# ------------------------------------------------------------------------------

df_matrix1 = Matrix(df1)

# Set initial parameters
δ_c_11 = 0.75   
δ_c_21 = 0.20
δ_c_31 = 0.15
δ_c_41 = 0.14

δ_c_12 = 0.72  
δ_c_22 = 0.23
δ_c_32 = 0.13
δ_c_42 = 0.15

δ_c_13 = 0.70   
δ_c_23 = 0.25
δ_c_33 = 0.11
δ_c_43 = 0.16

params1 = [δ_c_11, δ_c_21, δ_c_31, δ_c_41, 
           δ_c_12, δ_c_22, δ_c_32, δ_c_42, 
           δ_c_13, δ_c_23, δ_c_33, δ_c_43]

# Add 4 CRE parameters (initially zero)
params1_CRE = vcat(params1, [-0.01, -0.01, -0.01, -0.005])

# DEFINE PARAMETER NAMES EARLY (before using them)
param_names = ["δ₁₁", "δ₂₁", "δ₃₁", "δ₄₁", 
               "δ₁₂", "δ₂₂", "δ₃₂", "δ₄₂",
               "δ₁₃", "δ₂₃", "δ₃₃", "δ₄₃", 
               "γ₁", "γ₂", "γ₃", "γ₄"]

# ==============================================================================
# PRECOMPUTE CHILD AVERAGES (CRE) FROM ORIGINAL DATA - ONCE
# ==============================================================================
function compute_child_averages(df)
    child_ids = unique(df[:,1])
    avg_dict = Dict{Int, Vector{Float64}}()
    for id in child_ids
        rows = findall(df[:,1] .== id)
        avg_vals = Float64[]
        for col in [7, 10, 3, 11]  # z, tau, M, hyp
            valid_rows = [r for r in rows if df[r,col] > 0]
            push!(avg_vals, isempty(valid_rows) ? 0.0 : mean(log.(df[valid_rows, col])))
        end
        avg_dict[id] = [v <= 0 ? 1e-8 : v for v in avg_vals]
    end
    return avg_dict
end

# Compute ONCE from original data
global_avg_dict = compute_child_averages(df_matrix1)

# ==============================================================================
# MODIFIED MOMENT CONDITIONS - NOW TAKES PRE-COMPUTED avg_dict
# ==============================================================================
function moment_conditions_cog_CRE(params, df, avg_dict)
    groups = [1, 2, 3]
    all_moments = []

    for (i, group) in enumerate(groups)
        valid_rows = filter(row -> row[2]==group &&
            row[3] > 0 && row[4] > 0 &&
            row[7] > 0 && row[8] > 0 &&
            row[9] > 0 && row[10] > 0 &&
            row[11] > 0, eachrow(df))

        if isempty(valid_rows)
            continue
        end

        # Current observations
        z_f     = [row[4] for row in valid_rows]
        z       = [row[7] for row in valid_rows]
        tau     = [row[10] for row in valid_rows]
        M       = [row[3] for row in valid_rows]
        hyp     = [row[11] for row in valid_rows]

        # Instruments
        inst1 = [row[8] for row in valid_rows]
        inst2 = [row[9] for row in valid_rows]
        instruments = hcat(inst1, inst2)

        # USE PRE-COMPUTED child averages (passed in as argument)
        child_avg = [avg_dict[row[1]] for row in valid_rows]
        avg_matrix = hcat(child_avg...)'  # Nx4

        # Fitted values: ln_A + δ*current + γ*child_avg
        fitted = params[1+4*(i-1)] .* log.(z) .+
                 params[2+4*(i-1)] .* log.(tau) .+
                 params[3+4*(i-1)] .* log.(M) .+
                 params[4+4*(i-1)] .* log.(hyp) .+
                 params[13] .* (avg_matrix[:,1]) .+
                 params[14] .* (avg_matrix[:,2]) .+
                 params[15] .* (avg_matrix[:,3]) .+
                 params[16] .* (avg_matrix[:,4])

        residuals = log.(z_f) .- fitted

        # Moments: residuals times instruments
        moments = hcat([residuals .* instruments[:,j] for j in 1:size(instruments,2)]...)
        push!(all_moments, moments)
    end

    return vcat(all_moments...)
end

# ==============================================================================
# UPDATED OBJECTIVE FUNCTIONS - NOW TAKE avg_dict
# ==============================================================================
function gmm_objective(params, data, avg_dict)
    m = moment_conditions_cog_CRE(params, data, avg_dict)
    g = vec(mean(m, dims=1))
    return dot(g,g)  # Identity weighting
end

function compute_optimal_weight_matrix(params, data, avg_dict)
    m = moment_conditions_cog_CRE(params, data, avg_dict)
    g = m .- mean(m, dims=1)
    W = (g' * g) / size(g,1)
    return inv(W)
end

function gmm_objective_weighted(params, data, W, avg_dict)
    m = moment_conditions_cog_CRE(params, data, avg_dict)
    g = vec(mean(m, dims=1))
    return g' * W * g
end

# ==============================================================================
# ESTIMATION WITH ORIGINAL DATA
# ==============================================================================
println("Starting first-step GMM estimation...")
res_step1 = Optim.optimize(p -> gmm_objective(p, df_matrix1, global_avg_dict), params1_CRE, BFGS())
params_step1 = Optim.minimizer(res_step1)

println("Computing optimal weighting matrix...")
W_opt = compute_optimal_weight_matrix(params_step1, df_matrix1, global_avg_dict)

println("Starting efficient GMM estimation...")
res_step2 = Optim.optimize(p -> gmm_objective_weighted(p, df_matrix1, W_opt, global_avg_dict), params_step1, BFGS())
params_step2 = Optim.minimizer(res_step2)

println("\nOriginal estimates:")
param_names = ["δ₁₁", "δ₂₁", "δ₃₁", "δ₄₁", 
               "δ₁₂", "δ₂₂", "δ₃₂", "δ₄₂",
               "δ₁₃", "δ₂₃", "δ₃₃", "δ₄₃", 
               "γ₁", "γ₂", "γ₃", "γ₄"]
for i in 1:16
    println("$(param_names[i]): $(round(params_step2[i]; digits=4))")
end

# ==============================================================================
# BOOTSTRAP
# ==============================================================================
num_bootstrap = 100
param_boot = zeros(num_bootstrap, length(params1_CRE))
child_ids = unique(df_matrix1[:,1])
H = length(child_ids)

println("\n=== Starting Bootstrap (resampling $H children) ===")

for b in 1:num_bootstrap
    if b % 10 == 0
        println("Bootstrap iteration $b / $num_bootstrap")
    end

    # Resample households WITH REPLACEMENT (keep original IDs)
    sampled_ids = sample(child_ids, H, replace=true)
    
    df_b = DataFrame()
    for original_id in sampled_ids
        rows = df_matrix1[df_matrix1[:,1] .== original_id, :]
        df_b = vcat(df_b, DataFrame(rows, :auto))
    end
    
    df_b_matrix = Matrix(df_b)

    try
        # First stage (using GLOBAL avg_dict)
        res1 = Optim.optimize(
            p -> gmm_objective(p, df_b_matrix, global_avg_dict), 
            params_step2,  # Start from original estimates
            BFGS(),
            Optim.Options(g_tol=1e-4, f_tol=1e-6)
        )
        params_step1_b = Optim.minimizer(res1)

        # Second stage (using GLOBAL avg_dict)
        W_b = compute_optimal_weight_matrix(params_step1_b, df_b_matrix, global_avg_dict)
        res2 = Optim.optimize(
            p -> gmm_objective_weighted(p, df_b_matrix, W_b, global_avg_dict), 
            params_step1_b, 
            BFGS(),
            Optim.Options(g_tol=1e-4, f_tol=1e-6)
        )
        param_boot[b,:] .= Optim.minimizer(res2)

    catch e
        println("Bootstrap iteration $b failed: ", e)
        param_boot[b,:] .= NaN
    end
end

# ==============================================================================
# BOOTSTRAP DIAGNOSTICS
# ==============================================================================
num_successful = sum([all(.!isnan.(param_boot[b,:])) for b in 1:num_bootstrap])
println("\nBootstrap success rate: $(num_successful)/$(num_bootstrap)")

# Filter valid bootstrap results
valid_rows = findall(row -> all(.!isnan.(row)), eachrow(param_boot))
param_boot_clean = param_boot[valid_rows,:]

# CHECK: Bootstrap parameter distributions
println("\n=== Bootstrap Parameter Distributions ===")
for i in 1:length(params_step2)
    boot_mean = mean(param_boot_clean[:,i])
    boot_sd = std(param_boot_clean[:,i])
    boot_min = minimum(param_boot_clean[:,i])
    boot_max = maximum(param_boot_clean[:,i])
    original = params_step2[i]
    println("$(param_names[i]): Original=$(round(original, digits=4)), " *
            "Boot Mean=$(round(boot_mean, digits=4)), " *
            "Boot SD=$(round(boot_sd, digits=4)), " *
            "Range=$(round(boot_max - boot_min, digits=4))")
end

# Compute standard errors
boot_se = mapslices(std, param_boot_clean; dims=1)

println("\n=== FINAL RESULTS ===")
for i in 1:16
    println("$(param_names[i]): $(round(params_step2[i]; digits=4)) (SE = $(round(boot_se[i]; digits=4)))")
end

# Save the results

flatten_params1 = vec(params_step2)  # ensures it's a 1D vector

# Save estimated parameters
df_params1 = DataFrame(estimated_params1 = flatten_params1)
CSV.write("params1.csv", df_params1)

# Save bootstrapped SEs for the non-cognitive skill parameters
boot_se_vec = vec(boot_se)

df_se_cog = DataFrame(
    param_id = 1:length(boot_se_vec),
    se = boot_se_vec
)

CSV.write("params1_se.csv", df_se_cog)
