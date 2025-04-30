
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
using CSV  # Added this for loading the CSV
using Distributions
using Optim
using GLM
using MomentOpt
using StatsModels
using ForwardDiff
using StatsBase

# Set "include" path
cd("/Users/clairekim/Desktop/Project/Child Development/LSAC/code/")
include("backward_induction.jl")
include("smm.jl")
include("theta.jl")
#include("smm_DW.jl")

# Load data from the CSV file
df1 = CSV.File("data_1st.csv") # First-stage estimation
df2 = CSV.File("data_2nd.csv")

# Convert to a DataFrame
df1 = DataFrame(df1)
df2 = DataFrame(df2)

# Define new column names
#colnames = ["cid", "cage", "mage", "Ec", "Ep", "female", "CPM", "read", "num", "write", "Y", "s", "τ", "medu"]

# ==============================================================================
# 1. Model Estimation
# ------------------------------------------------------------------------------

# Step 0: Measurement parameters

df_matrix1 = Matrix(df1)
n_age = 3   # Length of Age
n_c = length(unique(df_matrix1[:, 1]))  # Number of children

μ_0, μ_1 = factor(df_matrix1, n_age)
signal, noise = signal_and_noise(df_matrix1, n_age)

# Plot the change in average cognitive skills over time
avgcog = mean_theta(df_matrix1, n_age)

# Define the periods with age ranges as strings
periods = ["10-11", "12-13", "14-15"]

# Create the plot with two lines, one for avgcogstar and one for avgcog
plot(periods, [avgcog[1], avgcog[2], avgcog[3]], 
     linestyle=:solid, marker=:circle, color=:blue, 
     xlabel="Age", ylabel="Average Log Cognitive Skills", legend=false)

# Increase the plot size (more space to the right)
#plot!(size=(800, 600))  # Increase width to make room for right margin

# Save the plot as a PNG file
savefig("/Users/clairekim/Desktop/Project/Child Development/LSAC/code/figs/avgcog.png")

# Add error-free non-cognitive skills column at the end

# Run the simulation
simulated_skills = recover_latent_skills_from_sdq(n_c, n_age, df_matrix1; mu_0 = -2.94, mu_1 = 1.0, p_min = 0.05)
# Initialize an empty vector to store the flattened values
simulated_skills_flat = Float64[]

# Loop over each household (n_c) and each time period (n_age)
for h in 1:n_c
    for t in 1:n_age
        push!(simulated_skills_flat, simulated_skills[h, t])
    end
end

# Ensure the lengths match before appending the new column
if length(simulated_skills_flat) == size(df_matrix1, 1)
    # Add the flattened simulated_skills_flat as the 13th column to df_matrix2
    df_matrix1 = hcat(df_matrix1, simulated_skills_flat)
else
    println("Error: Lengths do not match!")
end

ncog = df_matrix1[:, 12]         # Error-free non-cognitive skills

# Plot the change in average cognitive skills over time
avgncog1 = mean(df_matrix1[(df_matrix1[:, 2] .== 1), 12])
avgncog2 = mean(df_matrix1[(df_matrix1[:, 2] .== 2), 12])
avgncog3 = mean(df_matrix1[(df_matrix1[:, 2] .== 3), 12])

# Define the periods with age ranges as strings
periods = ["10-11", "12-13", "14-15"]

# Create the plot with two lines, one for avgcogstar and one for avgcog
plot(periods, [avgncog1, avgncog2, avgncog3], 
     linestyle=:solid, marker=:circle, color=:blue, 
     xlabel="Age", ylabel="Average Log Non-Cognitive Skills", legend=false)

# Increase the plot size (more space to the right)
#plot!(size=(800, 600))  # Increase width to make room for right margin

# Save the plot as a PNG file
savefig("/Users/clairekim/Desktop/Project/Child Development/LSAC/code/figs/avgncog.png")

# Step 1: Estimate production parameters

# Set initial parameters
A_c_1 = -0.0866 #(9) TFP at age 10 - 11 for cognitive skills
δ_c_11 = 0.573 #(10) marginal effect of lagged cognitive skills at age 10 - 11
δ_c_12 = 0.076 #(11) marginal effect of study time at age 10 - 11
δ_c_13 = 0.062 #(12) marginal effect of education-related goods at age 10 - 11
δ_c_14 = 0.135 #(13) marginal effect of the interaction between lagged non-cognitive skills and study time at age 10 - 11
A_c_2 = -0.0864 #(14) TFP at age 12 - 13 for cognitive skills
δ_c_21 = 0.533 #(15) marginal effect of lagged cognitive skills at age 12 - 13
δ_c_22 = 0.082 #(16) marginal effect of study time at age 12 - 13
δ_c_23 = 0.068 #(17) marginal effect of education-related goods at age 12 - 13
δ_c_24 = 0.149 #(18) marginal effect of the interaction between lagged non-cognitive skills and study time at age 12 - 13
A_c_3 = -0.0858 #(19) TFP at age 14 - 15 for cognitive skills
δ_c_31 = 0.494 #(20) marginal effect of lagged cognitive skills at age 14 - 15
δ_c_32 = 0.090 #(21) marginal effect of study time at age 14 - 15
δ_c_33 = 0.069 #(22) marginal effect of education-related goods at age 14 - 15
δ_c_34 = 0.163 #(23) marginal effect of the interaction between lagged non-cognitive skills and study time at age 14 - 15

params1 = [A_c_1, δ_c_11, δ_c_12, δ_c_13, δ_c_14, A_c_2, δ_c_21, δ_c_22, δ_c_23, δ_c_24,
 A_c_3, δ_c_31, δ_c_32, δ_c_33, δ_c_34]

params1[5] = 0.121

function moment_conditions(params1, df_matrix1)
    groups = [1, 2, 3]
    all_moments = []

    μ_0, μ_1 = factor(df_matrix1, n_age)

    for (i, group) in enumerate(groups)
        # Filter valid rows
        valid_rows = filter(row -> row[2] == group && row[3] != -999 && 
                                       row[11] != -999, eachrow(df_matrix1))

        if length(valid_rows) == 0
            continue
        end

        # Extract variables
        z_f = log.([row[5] for row in valid_rows])                      # future proxy (θ_{j,t+1})
        z = log.([row[8] for row in valid_rows])              # current proxy (θ_{j,t})
        log_tau = log.([row[11] for row in valid_rows])               # log(τ_{c,t})
        log_M = log.([row[3] for row in valid_rows])                  # log(M_t)
        log_hyp = log.([row[12] for row in valid_rows])
        interaction = log_tau .* log_hyp         # log(τ_{c,t}) * log(θ_{t}^N)

        # Rescale proxies
        z_f_tilde = (z_f .- μ_0[1, i]) ./ μ_1[1, i]
        z_tilde = (z .- μ_0[1, i]) ./ μ_1[1, i]

        # Instruments: lagged alternate proxies (assumed uncorrelated measurement errors)
        inst1 = log.([row[9] for row in valid_rows])
        inst2 = log.([row[10] for row in valid_rows])
        instruments = hcat(inst1, inst2)

        # Parameter block for group
        offset = (i - 1) * 5
        ln_A = params1[offset + 1]
        δ1 = params1[offset + 2]
        δ2 = params1[offset + 3]
        δ3 = params1[offset + 4]
        δ4 = params1[offset + 5]

        # Predicted values
        fitted = ln_A .+ δ1 .* z_tilde .+ δ2 .* log_tau .+ δ3 .* log_M .+ δ4 .* interaction

        # Residuals: full composite error term
        residuals = z_f_tilde .- fitted

        # Moment conditions: residuals × instruments (broadcast correctly)
        moments = hcat([residuals .* instruments[:, j] for j in 1:size(instruments, 2)]...)
        push!(all_moments, moments)
    end

    return vcat(all_moments...)
end

function gmm_objective(params, data)
    m = moment_conditions(params, data)
    g = vec(mean(m, dims=1))
    return dot(g, g)  # Simple GMM with identity weighting matrix
end

# Initial parameter guess: 5 parameters per group × 3 groups = 15
res_prod = optimize(
    p -> gmm_objective(p, df_matrix1),
    params1,
    NelderMead(),
    Optim.Options(iterations = 5000, f_tol = 1e-8, x_tol = 1e-8)
)

println(res_prod.minimizer)

params1[1:15] = res_prod.minimizer[1:15]

# Bootstrapped CI
num_bootstrap = 100  # Number of bootstrap repetitions
param_boot = zeros(num_bootstrap, length(params1))

# Get unique household IDs
household_ids = unique(df_matrix1[:, 1])
H = length(household_ids)

for b in 1:num_bootstrap

    # Print progress every 10 iterations
    if b % 10 == 0
        println("Bootstrap iteration $b / $num_bootstrap")
    end

    # Resample household IDs with replacement
    sampled_ids = sample(household_ids, H, replace=true)
    
    # Build resampled dataframe
    df_b = vcat([
        df_matrix1[df_matrix1[:, 1] .== hh, :] for hh in sampled_ids
    ]...)

    # Try-catch in case optimization fails
    try
        res_b = optimize(
            p -> gmm_objective(p, df_b),
            params1,  # Use original as starting point
            NelderMead(),
            Optim.Options(iterations=1000)
        )

        param_boot[b, :] .= res_b.minimizer
    catch e
        println("Bootstrap iteration $b failed: ", e)
        param_boot[b, :] .= NaN
    end
end

# Filter successful bootstrap samples
valid_rows = findall(row -> all(.!isnan.(row)), eachrow(param_boot))
param_boot_clean = param_boot[valid_rows, :]

# 95% confidence intervals
ci_lower = mapslices(p -> quantile(p, 0.025), param_boot_clean; dims=1)
ci_upper = mapslices(p -> quantile(p, 0.975), param_boot_clean; dims=1)

# Combine into readable output
confidence_intervals = hcat(ci_lower', ci_upper')  # Each row: [lower upper] for a param

# Display
for (i, ci) in enumerate(eachrow(confidence_intervals))
    println("Param $i: 95% CI = [", round(ci[1], digits=4), ", ", round(ci[2], digits=4), "]")
end


# Step 2: Estimate the model using SMM

df_matrix2 = Matrix(df2)

# Extract columns
cid = df_matrix2[:, 1]        # Child's ID
age = df_matrix2[:, 2]       # Child's Age (time period)
mage = df_matrix2[:, 3]          # Mother's Age
hyp = df_matrix2[:, 4]         # Child's Sustained attention
cage = df_matrix2[:, 5]       # Child's Actual Age
female = df_matrix2[:, 6]    # Child Gender
CPM = df_matrix2[:, 7]       # CPM Choice   
Y = df_matrix2[:, 8]       # Household Weekly Income
s = df_matrix2[:, 9]         # School Hours
τ = df_matrix2[:, 10]         # Study Time
medu = df_matrix2[:, 11]         # Mother's Educational Attainment
pm = df_matrix2[:, 12]         # Weekly Amount of Pockey Money
M = df_matrix2[:, 13]         # Weekly Educational Investment

# Primitives
β_c = 0.90            # Child's Discount Factor
β_p = 0.95            # Parent's Discount Factor
Time = 70               # Total available hours per week
φ = 0.333 # Parental Altruism

# Grids and dimensions
n_c = length(unique(df_matrix2[:, 1]))  # Number of children
n_age = 3   # Length of Age

# Add error-free non-cognitive skills column at the end

# Run the simulation
simulated_skills = recover_latent_skills_from_sdq(n_c, n_age, df_matrix2; mu_0 = -2.94, mu_1 = 1.0, p_min = 0.05)
# Initialize an empty vector to store the flattened values
simulated_skills_flat = Float64[]

# Loop over each household (n_c) and each time period (n_age)
for h in 1:n_c
    for t in 1:n_age
        push!(simulated_skills_flat, simulated_skills[h, t])
    end
end

# Ensure the lengths match before appending the new column
if length(simulated_skills_flat) == size(df_matrix2, 1)
    # Add the flattened simulated_skills_flat as the 14th column to df_matrix2
    df_matrix2 = hcat(df_matrix2, simulated_skills_flat)
else
    println("Error: Lengths do not match!")
end

# Ensure the flattened simulated skills are correctly aligned with the original 2D matrix
println("Min of simulated_skills before flattening: ", minimum(simulated_skills))
println("Min of simulated_skills_flat before adding to df_matrix2: ", minimum(simulated_skills_flat))

df_matrix2[:, 14] = simulated_skills_flat
println("Min of df_matrix2[:, 14] after direct assignment: ", minimum(df_matrix2[:, 14]))

ncog = df_matrix2[:, 14]         # Error-free non-cognitive skills
println("Min of ncog after direct assignment: ", minimum(ncog))

# Set initial parameters
μ_α0 = 0.124
μ_α1 = 2.545 #(1) Parents' preference over their private consumption
μ_λ0 = 1.092
μ_λ1 = 1.808 #(2) Child's preference over their own leisure
μ_λ2 = 1.100 #(3) Child's preference over their non-cognitive skills
σ_α0 = 0.118
σ_α1 = 0.943 #(4) Parents' preference over their private consumption
σ_λ0 = 1.078
σ_λ1 = 1.233 #(5) Child's preference over their own leisure
σ_λ2 = 1.084 #(6) Child's preference over their non-cognitive skills
σ_ζ = 0.065 #(7) CPM Cost distribution
ν_γ = -0.1966311 #(8) Scaling factor of leisure preference given CPM

A_n_1 = -0.0883 #(24) TFP at age 10 - 11 for non-cognitive skills
δ_n_11 = 0.058597336182281946 #(25) marginal effect of lagged non-cognitive skills at age 10 - 11
δ_n_12 = -0.011738844431260798 #(26) marginal effect of CPM at age 10 - 11
δ_n_13 = 0.037837741472181428 #(26) marginal effect of τ at age 10 - 11
A_n_2 = -0.0867 #(27) TFP at age 12 - 13 for non-cognitive skills
δ_n_21 = 0.010096173584764312 #(28) marginal effect of lagged non-cognitive skills at age 12 - 13
δ_n_22 = -0.05479509757062367 #(29) marginal effect of CPM at age 12 - 13
δ_n_23 = 0.14564950891079229 #(29) marginal effect of \tau at age 12 - 13
A_n_3 = -0.0823 #(30) TFP at age 14 - 15 for non-cognitive skills
δ_n_31 = 0.002 #(31) marginal effect of lagged non-cognitive skills at age 14 - 15
δ_n_32 = -0.100 #(32) marginal effect of CPM at age 14 - 15
δ_n_33 = 0.27 #(32) marginal effect of \tau at age 14 - 15

params2 = [μ_α1, μ_λ1, μ_λ2, σ_α1, σ_λ1, σ_λ2, σ_ζ, ν_γ,
 A_n_1, δ_n_11, δ_n_12, A_n_2, δ_n_21, δ_n_22, A_n_3, δ_n_31, δ_n_32, μ_α0, μ_λ0, σ_α0, σ_λ0, δ_n_13, δ_n_23, δ_n_33]
ReMatrix(GenAvgMoments(params1, params2, df_matrix2))

#(1) Use only 1000 iterations to get better guesses of the parameters

# Define the optimization options
#options1 = Optim.Options(
#    iterations = 1000,      # Max number of iterations       # Relative tolerance for convergence criterion (change for Subplex)
#    g_tol = 1e-6          # Gradient tolerance (convergence criterion)
#)
# Ensure that each parameter doesn't go below its corresponding lower bound
#lower_bounds = [0.0, 0.0, 0.0, NaN, NaN, 0.0, NaN, NaN, NaN, 0.0, 0.0, 0.0,
#0.0, 0.0, 0.0, 0.0, 0.0, 0.0, NaN, NaN, 
#NaN, NaN, NaN, NaN, 0.0, NaN, NaN, NaN]

# Replace NaNs with -Inf (free variables) for those parameters
#lower_bounds = [isnan(x) ? -Inf : x for x in lower_bounds]

function objective_function_lb(params1, params2, df_matrix, lower_bounds)
    
    params2 = max.(params2, lower_bounds)  # Apply lower bounds element-wise

    # Call the Calibrate function, passing the parameters (which are now constrained)
    return Calibrate_w(params1, params2, df_matrix)
end

function objective_function(params1, params2, df_matrix)
    
    # Call the Calibrate function, passing the parameters (which are now constrained)
    return Calibrate_w(params1, params2, df_matrix)
end

#objective_function(params1, params2, df_matrix2)

# Optimize using Nelder-Mead
# Set the options
opts = Optim.Options(
    iterations = 1000,       # Maximum number of iterations
    f_tol = 1e-6,           # Function tolerance (change in the function value)
    show_trace = true,       # Show optimization progress
)

# Run the optimizer with Nelder-Mead method and the correct options
result1 = optimize(
    p -> objective_function(params1, p, df_matrix2),  # Objective function
    params2,  # Initial guess for the parameters
    NelderMead(),  # Derivative-free method
    opts   # Pass the options
)

#compute_final_estimates_and_standard_errors(params, df_matrix, 100, 1000)

# Retrieve the optimized parameters
parametervals = Optim.minimizer(result1)

opts2 = Optim.Options(
    iterations = 5000,       # Maximum number of iterations
    f_tol = 1e-6,           # Function tolerance (change in the function value)
    show_trace = true,       # Show optimization progress
)

# Call the optimization function
result2 = optimize(
    p -> objective_function(params1, p, df_matrix2),  # Objective function
    parametervals,  # Initial guess for the parameters
    NelderMead(),  # Derivative-free method
    opts2   # Pass the options
)

#result2 = optimize(v -> Calibrate(v, df_matrix), parametervals, iterations = 5000, g_tol = 1e-6)
#result2 = optimize(v -> Calibrate(v, df_matrix), parametervals, 
#                   lower_bounds = lower_bounds, Fminbox(NelderMead()),
#                   iterations = 5000, g_tol = 1e-6)
# Optimize using Nelder-Mead with bounds
# Retrieve the (final) optimized parameters
params_fin = Optim.minimizer(result2)

function bootstrap_CIs(params1, params2, df_matrix::Matrix, B::Int)
    estimates = []

    # Get unique child IDs from the first column (assumed)
    unique_ids = unique(df_matrix[:, 1])

    for b in 1:B
        println("Bootstrap replication $b...")

        try
            # Step 1: Resample child IDs with replacement
            resampled_ids = rand(unique_ids, length(unique_ids))

            # Step 2: Construct bootstrap sample by ID
            bootstrap_sample = vcat([df_matrix[df_matrix[:, 1] .== id, :] for id in resampled_ids]...)

            # Step 3: Re-estimate parameters on bootstrap sample
            obj_fun = p -> objective_function(params1, p, bootstrap_sample)

            result = optimize(
                obj_fun,
                params2,  # Initial guess
                NelderMead(),
                Optim.Options(iterations=1000, f_tol=1e-6, show_trace=false)
            )

            est_params2 = Optim.minimizer(result)
            # If `params1` are fixed, you can just return those alongside:
            push!(estimates, vcat(params1, est_params2))

        catch e
            println("Bootstrap replication $b failed with error: $e")
        end
    end

    # Step 4: Convert list of estimates into a matrix
    estimates_mat = hcat(estimates...)'  # B x (length(params1) + length(params2))

    # Step 5: Compute confidence intervals (2.5% and 97.5% quantiles)
    lower_bound = mapslices(x -> quantile(x, 0.025), estimates_mat, dims=1)
    upper_bound = mapslices(x -> quantile(x, 0.975), estimates_mat, dims=1)

    return lower_bound, upper_bound, estimates_mat
end

result3 = optimize(
    p -> objective_function(p, df_matrix, lower_bounds),  # Objective function
    params_fin,  # Initial guess for the parameters
    NelderMead(),  # Derivative-free method
    opts2   # Pass the options
)

flatten_params1 = vcat(params1...)
flatten_params2 = vcat(params_fin...)

df_params1 = DataFrame(estimated_params1 = flatten_params1)
df_params2 = DataFrame(estimated_params2 = flatten_params2)

# Save to CSV
CSV.write("params1.csv", df_params1)
CSV.write("params2.csv", df_params2)

#result3 = optimize(objective_function, params_fin, NelderMead())
#params_temp = Optim.minimizer(result3)
#sim_data = Simulate(params_temp, n_c, n_age, Ep, s, Time, df_matrix) 
#sim_moments = ReMatrix(GenAvgMoments(params_temp, df_matrix)) 

# ==============================================================================
# 2. Model Fit
# ------------------------------------------------------------------------------

# Generate true data moments
true_moments = ReMatrix(GenMoments(df_matrix2))
true_moments = ReMatrix_est(GenMoments_est(df_matrix2))

# Generate simulated data and average simulated data moments
#sim_data = Simulate(parametervals, n_c, n_age, Ep, s, Time, df_matrix)
#sim_data = Simulate(params_fin, n_c, n_age, Ep, s, Time, df_matrix)

#sim_moments = ReMatrix(GenAvgMoments(params_fin, df_matrix))
sim_moments = ReMatrix_est(GenAvgMoments_est(params1, params_fin, df_matrix2))
#sim_moments = ReMatrix(GenAvgMoments(params1, params2, df_matrix2))
#sim_moments = ReMatrix(GenAvgMoments(params1, parametervals, df_matrix2)) # Temporary
#sim_moments = ReMatrix_est(GenAvgMoments_est(params1, parametervals, df_matrix2))

using CSV

# Assuming true_moments and sim_moments are vectors of vectors
# Flatten the vectors
flatten_true_moments = vcat(true_moments...)
#data_mom_temp = DataFrame(true_moments = flatten_true_moments)
#CSV.write("data_mom_temp.csv", data_mom_temp)

flatten_sim_moments = vcat(sim_moments...)
#model_mom_temp = DataFrame(sim_moments = flatten_sim_moments)
#CSV.write("model_mom_temp.csv", model_mom_temp)

# Combine them into a 2-column DataFrame
df_moments = DataFrame(true_moments = flatten_true_moments, sim_moments = flatten_sim_moments)

# Save to CSV
CSV.write("moments_data.csv", df_moments)

# Call the optimization function
#result3 = optimize(
#    p -> objective_function(p, df_matrix, lower_bounds),  # Objective function
#    params_fin,  # Initial guess for the parameters
#    iterations = 5000,  # Number of iterations
#    g_tol = 1e-6          # Gradient tolerance (convergence criterion)
#)

#params_fin = Optim.minimizer(result3)

# ==============================================================================
# 3. Counterfactual Experiments
# ------------------------------------------------------------------------------

#(A) Quantifying the effect of CPM

# Overall Effect
cf_result = GenAvgMoments_cf(params1, params_fin, df_matrix2)
#println(GenAvgMoments_cf(params1, parametervals, df_matrix2))
println(cf_result)

# Inequality
cf1_result = GenAvgMoments_cf2(params1, params_fin, df_matrix2)
#cf1_result = GenAvgMoments_cf2(params1, parametervals, df_matrix2)
println(cf1_result)

cf2_result = GenAvgMoments_cf1(params1, params_fin, df_matrix2)
println(cf2_result) # Temporary... need to combine every code














# STEP 2: Compute standard errors

# Define the number of bootstrap samples
n_bootstrap = 1000  # Number of bootstrap iterations

# Function to perform cluster bootstrap and estimate parameters
using Random
using Statistics

# Bootstrap EM for parameter estimation and standard errors
function bootstrap(params, df, num_bootstraps::Int)
    #println("Bootstrapping with parameters V: ", V)
    
    individuals = unique(df[:, 1])  # Unique individuals
    num_params = length(params)          # Number of parameters

    bootstrapped_params = zeros(num_params, num_bootstraps)

    for b in 1:num_bootstraps
        # Generate bootstrap sample by sampling individuals with replacement
        bootstrap_ids = sample(individuals, length(individuals), replace=true)
        
        bootstrap_sample = filter(row -> in(row[1], bootstrap_ids), df)

        # MDE on bootstrap sample
        params_b = params

        opt_b = optimize(
            p -> objective_function(p, df_matrix, lower_bounds),  # Objective function
            params_fin,  # Initial guess for the parameters
            iterations = 5000,  # Number of iterations
            g_tol = 1e-6          # Gradient tolerance (convergence criterion)
        )
        V_bootstrap = opt_b.minimizer
        
        # Extract optimal parameters
        V_bootstrap = Optim.minimizer(opt_b)

        # Store the estimated parameters for the bootstrap sample
        bootstrapped_params[:, b] = V_bootstrap
    end

    return bootstrapped_params
end

# Compute Bootstrapped Standard Errors and Final Parameters
function compute_final_estimates_and_standard_errors(params, df, num_bootstraps::Int)
    num_params = length(params)        # Number of parameters
    # Estimate parameters on the original data
    #estimated_params = estimate_parameters(params, df)
    
    # Compute bootstrapped standard errors
    bootstrapped_est = estimated_params(params, df, num_bootstraps::Int)
    std_errors = std(bootstrapped_est, dims=2)

    return estimated_params, std_errors
end

# Load the final estimated parameters

result_se = compute_final_estimates_and_standard_errors(params_fin, df_matrix, 100)

# Display results
println("Standard Errors: ", standard_errors)
println("95% Confidence Intervals:")
for (param, lower, upper) in zip(params_fin, ci_lower, ci_upper)
    println("Parameter: $param => 95% CI: [$lower, $upper]")
end

# Optionally, save the results to CSV for further inspection
results_se_df = DataFrame(parameter=params_fin, standard_error=standard_errors, 
    ci_lower=ci_lower, ci_upper=ci_upper)
CSV.write("bootstrap_results.csv", results_se_df)

