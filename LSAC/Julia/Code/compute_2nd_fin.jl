
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
using Statistics
using DataFrames
using CSV  # Added this for loading the CSV
using Distributions
using Optim
using Base.Threads
using BlackBoxOptim

# Set "include" path
cd("/Users/clairekim/Desktop/Project/Child Development/LSAC/code/")
include("umax_fin.jl")
include("smm_fin.jl")
include("quantcpm.jl")
include("cpm_eval.jl")
include("decompose.jl")

# Load the CSV file (replace 'your_file.csv' with the actual file path)
params1c = CSV.read("params1.csv", DataFrame) # Cognitive Skill Production Parameters

# Convert the first column of the DataFrame into a vector (assuming the values are in the first column)
params1c = params1c[!, 1]  # The '!' symbol means to extract the entire column

# Now `params` contains your 15 elements from the CSV file
println(params1c)

# Load data from the CSV file
df_all = CSV.File("prefparams.csv") # First-stage estimation

# Convert to a DataFrame
df_all = DataFrame(df_all)

# ==============================================================================
# 1. Model Estimation
# ------------------------------------------------------------------------------

df_matrix_all = Matrix(df_all)

# Extract columns
cid = @view df_matrix_all[:, 1]        # Child's ID
age = @view df_matrix_all[:, 2]       # Child's Age (time period)
#cage = @view df_matrix_all[:, 3]          # Child's Actual Age
medu = @view df_matrix_all[:, 3]          # Mother's education
CPM = @view df_matrix_all[:, 4]       # CPM Choice   
Y = @view df_matrix_all[:, 5]       # Household Weekly Income   
s = @view df_matrix_all[:, 6]      # School Hours
τ = @view df_matrix_all[:, 7]         # Study Time
#medu = @view df_matrix_all[:, 8]        # Mother's Educational Attainment
pm = @view df_matrix_all[:, 8]          # Weekly Amount of Pocket Money
M = @view df_matrix_all[:, 9]        # Weekly Educational Investment
read = @view df_matrix_all[:, 10]         # Reading Score
hyp = @view df_matrix_all[:, 11]         # hyperactivity
R_bar = @view df_matrix_all[:, 12]         # Baseline Child-related goods spending
hyp_init = @view df_matrix_all[:, 13]

# Primitives
β_c = 0.90            # Child's Discount Factor
β_p = 0.95            # Parent's Discount Factor
Time = 960               # Total available hours per day (Assume sleeping time 8 hours)
φ = (1/3) # Parental Altruism

# Grids and dimensions
n_c = length(unique(df_matrix_all[:, 1]))  # Number of children
n_age = 3   # Length of Age

# Set initial parameters
μ_α0 = 1.4818176034627867      # parents' consumption preference
μ_λ0 = -0.60249571172782     # child's cognition preference
μ_λ1 = 2.273323139443664      # child's leisure preference

σ_α0 = 0.01375019009361514     # parents' consumption preference
σ_λ0 = 0.029798948662958993     # child's cognition preference
σ_λ1 = 0.035859267516245     # child's leisure preference

λ01 = -0.864233250414191     
α0λ0 = 0.0280041740506624     
α0λ1 = 0.43923386756067     

ζ = 0.0903377511645257       # CPM cost
g = 0.1328011422569909        # Low SR leisure weight

params_all = [μ_α0, μ_λ0, μ_λ1, σ_α0, σ_λ0, σ_λ1, λ01, α0λ0, α0λ1, ζ, g]

# ==============================================================================
# 1. Baseline Model
# ------------------------------------------------------------------------------

CT = zeros(n_c*n_age)
# Define objective with tracking
rng = MersenneTwister(1234)
Z = randn(rng, 3, n_c)             # base normals for ν (5 dims)

# Quick Check to see if NaN values exist

temp=value_func(Z, params1c, params_all, n_c, n_age, s, Time, Y, CT, R_bar, hyp)
M_opt, CPM_opt, τ_opt, R_opt = temp[3], temp[1], temp[2], temp[4]

sim_moments = simulated_moments(CPM_opt, τ_opt, R_opt, M_opt, df_matrix_all, n_c, n_age, core_only=false)
println("sim_moments: ", sim_moments)

emp_moms = data_moments(df_matrix_all, n_age, core_only = false)
println("emp_moments: ", emp_moms)

# Check preference parameters
println("\n--- Step 3: Preference Parameter Check ---")
println("α_1: mean=$(round(temp[5], digits=3)), std=$(round(temp[6], digits=3))")
println("λ_0: mean=$(round(temp[7], digits=3)), std=$(round(temp[8], digits=3))")
println("λ_1: mean=$(round(temp[9], digits=3)), std=$(round(temp[10], digits=3))")
println("λ_2: mean=$(round(temp[11], digits=3)), std=$(round(temp[12], digits=3))")

# Check CPM adoption
println("\n--- Step 3: CPM Adoption ---")
println("Overall CPM rate: $(round(mean(CPM_opt)*100, digits=2))% (Target: 16-19%)")
for t in 1:n_age
    cpm_rate = mean(CPM_opt[:, t])
    println("  Period $t: $(round(cpm_rate*100, digits=2))%")
end

# Check study time
println("\n--- Step 3: Study Time by CPM Status ---")
for cpm_status in 0:1
    idx = CPM_opt .== cpm_status
    if any(idx)
        avg_tau = mean(τ_opt[idx])
        println("  CPM=$cpm_status: $(round(avg_tau, digits=2)) min/week")
    end
end

# Check key moments
println("\n--- Step 3: Key Moment Comparison ---")
for (label, indices) in zip(moment_labels, moment_indices)
    println("\n$label:")
    for (j, idx) in enumerate(indices)
        sim_val = sim_moments[idx]
        emp_val = emp_moms[idx]
        diff = sim_val - emp_val
        pct_diff = 100 * diff / (abs(emp_val) + 1e-10)
        println("  $(j): Sim=$(round(sim_val, digits=2)) vs Emp=$(round(emp_val, digits=2)) [Diff=$(round(diff, digits=2)), $(round(pct_diff, digits=1))%]")
    end
end

println("\n" * "="^80)
println("END OF STEP 3")
println("="^80)

# Compute weighting matrix using original data
child_data = index_by_child(df_matrix_all, n_c)
moment_std_boot = bootstrap_moment_std(child_data, n_c, n_age, 100; core_only = false)

# Compute variance estimates
moment_var_boot = moment_std_boot .^ 2

# Weight matrix for SMM: inverse of bootstrapped variances
W = Diagonal(1 ./ moment_std_boot)
diag(W)

# Put more weights on main moments

# All Moments
for i = 1:46
    W[i, i] = 1.0
end

# Study Time moments
for i = 1:12
    W[i, i] = 10.0
end

# CPM moments
for i = 13:30
    W[i, i] = 5000.0  
end

# Pocket Money moments
for i = 31:40
    W[i, i] = 5000.0 
end

# Define initial parameter vector
initial_params = copy(params_all)

function obj(params)
    error = distance_function_optimized(
    params1c, params, W, n_c, n_age,
    s, Time, Y, CT, Z, emp_moms, df_matrix_all, hyp; core_only=false)

    return error
end

res_opt = Optim.optimize(
    obj,          # objective function
    initial_params,
    NelderMead(),         # Nelder-Mead algorithm
    Optim.Options(
        iterations = 5000,
        f_tol = 1e-4,
        x_tol = 1e-4,
        g_tol = 1e-6,
        show_trace = true,
        show_every = 10
        #trace_simplex = true
        #time_limit = 7200
    )
)

println("Nelder-Mead solution: ", res_opt.minimizer)

baseline_results = value_func(Z, params1c, res_opt.minimizer, n_c, n_age, s, Time, Y, CT, R_bar, hyp)

derived_baseline = [
    baseline_results[5], baseline_results[6],
    baseline_results[7], baseline_results[8],
    baseline_results[9], baseline_results[10],
    baseline_results[11], baseline_results[12]
]

println("Best derived_baseline: ", derived_baseline)

# Save results

# Flatten parameters
flatten_params_fin = vcat(res_opt.minimizer...)

# Create DataFrame for parameters
df_params_fin = DataFrame(estimated_params_fin = flatten_params_fin)

# Save parameters to CSV
CSV.write("params_fin.csv", df_params_fin)

# Create DataFrame for derived_baseline
df_derived_baseline = DataFrame(derived_baseline = derived_baseline)

# Save derived_baseline to CSV
CSV.write("derived_baseline.csv", df_derived_baseline)

# ==============================================================================
# 2. Model Fit
# ------------------------------------------------------------------------------

M_opt, CPM_opt, τ_opt, R_opt = baseline_results[3], baseline_results[1], baseline_results[2], baseline_results[4]
sim_moments = simulated_moments(CPM_opt, τ_opt, R_opt, M_opt, df_matrix_all, n_c, n_age, core_only = false)
println("sim_moments: ", sim_moments)
emp_moms = data_moments(df_matrix_all, n_age, core_only = false)
println("emp_moments: ", emp_moms)

# Assuming true_moments and sim_moments are vectors of vectors
# Flatten the vectors
flatten_true_moments_est = vcat(true_moments...)
flatten_sim_moments_est = vcat(sim_moments...)

# Combine them into a 2-column DataFrame
df_moments = DataFrame(true_moments_est = flatten_true_moments_est, sim_moments_est = flatten_sim_moments_est)

# Save to CSV
CSV.write("moments_data.csv", df_moments)

# Plotting Model Fit for child's study hours

# Divide moments by 60 to convert to hours
sim_moments_hrs = zeros(12)
emp_moms_hrs = zeros(12)

sim_moments_hrs .= sim_moments[1:12] ./ 60
emp_moms_hrs .= emp_moms[1:12] ./ 60

# Define the groups
groups = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
]

group_labels = ["High SR - No CPM", "High SR - CPM", "Low SR - No CPM", "Low SR - CPM"]

periods = 1:3  # just 3 x-ticks
age_labels = ["10-11", "12-13", "14-15"]  # x-axis labels

# Create a 2x2 layout for 4 plots
plot_layout = @layout [a b; c d]
p = plot(layout = plot_layout, size=(800,600))

for i in 1:4
    group_idx = groups[i]
    
    # Create plot
    p = plot(
        1:3, emp_moms_hrs[group_idx],
        label="Data",
        markershape=:circle,
        markersize=5,      # <-- bigger markers
        color=:royalblue1,
        lw=2,               # <-- thicker lines
        ylims=(0, 2.0),
        xticks=(periods, age_labels),
        xlabel="Child's Ages",
        ylabel="Study Hours",
        legendfontsize=12,  # <-- legend font size
        guidefontsize=14,   # <-- axis labels font size
        tickfontsize=12     # <-- tick labels font size
    )
    
    plot!(1:3, sim_moments_hrs[group_idx],
          label="Model",
          markershape=:diamond,
          markersize=5,    # bigger markers
          lw=2,
          color=:violet)
    
    gui(p)  # display plot
    
    # Save plot
    filename = "tau_fit_" * replace(group_labels[i], " " => "_") * ".png"
    savefig(p, filename)
    
    println("Plot $i saved as $filename. Press Enter to see the next plot...")
    readline()
end


# ==============================================================================
# 3. Counterfactual Policies
# ------------------------------------------------------------------------------

# Load parameters
best_params = CSV.read("params_fin.csv", DataFrame)[!, 1]

# Run baseline to get optimal choices
println("Running baseline model...")
baseline_results = value_func(Z, params1c, best_params, n_c, n_age, s, Time, Y, CT, R_bar, hyp)

derived_baseline = [
    baseline_results[5], baseline_results[6],
    baseline_results[7], baseline_results[8],
    baseline_results[9], baseline_results[10],
    baseline_results[11], baseline_results[12]
]

M_opt = baseline_results[3]
CPM_opt = baseline_results[1]
τ_opt = baseline_results[2]
R_opt = baseline_results[4]

println("Baseline CPM adoption: $(round(mean(CPM_opt)*100, digits=1))%")
println("Baseline Pocket Money Amount (CPM only): $(round(mean(R_opt[CPM_opt .== 1]), digits=1))")

# ==============================================================================
# COUNTERFACTUAL 1: No CPM Available
# ==============================================================================

println("\n" * "="^80)
println("COUNTERFACTUAL 1: Skills Without CPM")
println("="^80)

# Run counterfactuals (assuming you have the baseline results already)
results = value_func_cf_comparative(
    Z, R_bar, params1c, best_params, n_c, n_age, s, Time, Y, CT,
    β_p, β_c, φ, M_opt, CPM_opt, τ_opt, R_opt
)

# Simulate skills forward
skills = simulate_skills_comparative(results, params1c, n_c, n_age, df_matrix_all)

# Compute treatment effects
hyperactivity_child = reshape(hyp_init, n_age, n_c)'[:, 1]
effects = compute_treatment_effects_by_age(skills, CPM_opt; hyperactivity=hyperactivity_child)

# ============================================================================
# REPORT RESULTS
# ============================================================================

using Printf

println("="^80)
println("COUNTERFACTUAL ANALYSIS: CPM's Role in Skill Formation")
println("="^80)

println("\nThe table below shows the average log cognitive skills under Baseline (with CPM) and No CPM scenarios,")
println("as well as the corresponding skill gaps and standard deviation units for all children and by self-regulation group.\n")

for t in 1:(n_age+1)  # include all periods, including period 4
    println("\nPeriod $(t):")
    println("  Average skills:")
    println("    Baseline (all): $(round(effects[:BaselineMean][t], digits=3))")
    println("    No CPM (all):   $(round(effects[:NoCPMMean][t], digits=3))")
    println("    Change (Δln θ): $(round(effects[:Change][t], digits=3)) log points")
    println("    Change (Δln θ SD): $(round(effects[:Change_SD][t], digits=3)) SD")

    println("  Heterogeneous effects:")
    println("    Low SR Δln θ:  $(round(effects[:Change_low][t], digits=3))")
    println("    High SR Δln θ: $(round(effects[:Change_high][t], digits=3))")
    println("    Low SR Δln θ SD:  $(round(effects[:Change_SD_low][t], digits=3)) SD")
    println("    High SR Δln θ SD: $(round(effects[:Change_SD_high][t], digits=3)) SD")
end

# ============================================================================
# PLOT SKILL DEVELOPMENT PATH
# ============================================================================

using Plots
gr()

# Periods correspond to next-period skills (t=2,3,4)
periods = 1:3  # just 3 x-ticks
age_labels = ["10-11", "12-13", "14-15"]  # x-axis labels

# Take only columns 2:end from effects
baseline_next = effects[:BaselineMean][2:end]
noCPM_next   = effects[:NoCPMMean][2:end]

# All children plot
cf1 = plot(
    periods, baseline_next,
    label="Baseline",
    lw=2,
    marker=:circle,
    legend=:bottomright,
    grid=true
)
plot!(periods, noCPM_next,
      label="No CPM",
      lw=2,
      marker=:diamond)

xticks!(periods, age_labels)
xlabel!("Child's Age")
ylabel!("Next-Period Log Cognitive Skill")

gui(cf1)
savefig(cf1, "skill_development.png")

# Subgroup plot
baseline_low  = effects[:BaselineMean_low][2:end]
baseline_high = effects[:BaselineMean_high][2:end]
noCPM_low     = effects[:NoCPMMean_low][2:end]
noCPM_high    = effects[:NoCPMMean_high][2:end]

# Compute gaps for shading
gap_baseline = baseline_low .- baseline_high
gap_noCPM   = noCPM_low .- noCPM_high

cf1_sr = plot(periods, baseline_low, label="Baseline Low SR", color=:blue, marker=:square, legend=:bottomright,lw=2)
plot!(periods, baseline_high, label="Baseline High SR", color=:red, marker=:square, legend=:bottomright,lw=2)
plot!(periods, noCPM_low, label="No CPM Low SR", color=:green, marker=:circle, legend=:bottomright,lw=2, linestyle=:dash)
plot!(periods, noCPM_high, label="No CPM High SR", color=:orange, marker=:circle, legend=:bottomright,lw=2, linestyle=:dash)

# Optional ribbons
plot!(periods, (baseline_low + baseline_high) ./ 2, ribbon=gap_baseline ./ 2, alpha=0.1, color=:pink, label="")
plot!(periods, (noCPM_low + noCPM_high) ./ 2, ribbon=gap_noCPM ./ 2, alpha=0.1, color=:yellow, label="")

xticks!(periods, age_labels)
xlabel!("Child's Age")
ylabel!("Next-Period Log Cognitive Skill")
gui(cf1_sr)
savefig(cf1_sr, "skill_development_sr.png")

# ==============================================================================
# COUNTERFACTUAL 2: Monetary Equivalence
# ==============================================================================

println("\n" * "="^80)
println("COUNTERFACTUAL 2: Monetary Equivalence")
println("="^80)

println("Converting CPM skill gains into equivalent educational investment (ΔM)")
println("and household income transfers (ΔY) needed to replicate the same gains")
println("when CPM is not available.\n")

# Define groups using INITIAL PERIOD hyperactivity
period_labels = ["10-11", "12-13", "14-15"]
skill_gain_labels = ["12-13", "14-15", "16-17"]  # Add this line

all_idx = collect(1:n_c)
idx_low = collect(findall((hyperactivity_child .>= 1) .& (hyperactivity_child .<= 5)))
idx_high = collect(findall((hyperactivity_child .>= 10) .& (hyperactivity_child .<= 11)))

println("Sample sizes:")
println("  All children: $(length(all_idx))")
println("  Low SR (score 1-5): $(length(idx_low))")
println("  High SR (score 6-11): $(length(idx_high))")

# Compute CPM-induced skill gains for next period
ΔZ_CPM_age_next      = vec(mean(skills[:baseline][:cognitive][:, 2:end] .- 
                                skills[:noCPM][:cognitive][:, 2:end], dims=1))
ΔZ_CPM_age_low_next  = vec(mean(skills[:baseline][:cognitive][idx_low, 2:end] .- 
                                skills[:noCPM][:cognitive][idx_low, 2:end], dims=1))
ΔZ_CPM_age_high_next = vec(mean(skills[:baseline][:cognitive][idx_high, 2:end] .- 
                                skills[:noCPM][:cognitive][idx_high, 2:end], dims=1))

# Store results
equiv_results = DataFrame(
    Group = String[],
    Age = String[],
    ΔZ_Target = Float64[],
    ΔM = Float64[],
    ΔY = Float64[],
    ΔM_pct_income = Float64[],
    ΔY_pct_income = Float64[]
)

# Loop over groups
for (group_name, hidx, ΔZ_group) in zip(
        ["All Children", "Low SR", "High SR"], 
        [all_idx, idx_low, idx_high],
        [ΔZ_CPM_age_next, ΔZ_CPM_age_low_next, ΔZ_CPM_age_high_next]
    )

    println("\n" * "="^70)
    println("Group: $group_name (n = $(length(hidx)))")
    println("="^70)

    avg_income = mean([mean(Y[(i-1)*n_age + 1 : (i-1)*n_age + (n_age-1)]) for i in hidx])
    println("Average income for periods 1:3: \$", round(avg_income, digits=2))

    for (t_idx, t_period) in enumerate(1:n_age)
        ΔZ_target = ΔZ_group[t_idx]

        println("\nInvest in Period $(period_labels[t_idx]) to achieve skill gain in $(skill_gain_labels[t_idx]):")
        println("  Target skill gain: $(round(ΔZ_target, digits=4)) log points")

        # ΔM equivalence
        ΔM = find_equiv_M(
            ΔZ_target, hidx, results, skills, 
            params1c, n_c, n_age, df_matrix_all;
            tol=1e-3, max_iter=10000,
            invest_period=t_period
        )

        # ΔY equivalence
        ΔY = find_equiv_Y(
            ΔZ_target, hidx, results, skills,
            params1c, best_params, n_c, n_age,
            s, Time, Y, CT, Z, R_bar, hyp, β_p, β_c, φ, df_matrix_all;
            tol=1e-3, max_iter=10000,
            invest_period=t_period
        )

        ΔM_pct = 100 * ΔM / avg_income
        ΔY_pct = 100 * ΔY / avg_income

        println(@sprintf("  ΔM ≈ \$%7.2f/week (%.2f%% of income)", ΔM, ΔM_pct))
        println(@sprintf("  ΔY ≈ \$%7.2f/week (%.2f%% of income)", ΔY, ΔY_pct))

        push!(equiv_results, (
            group_name, 
            skill_gain_labels[t_idx],  # Use skill_gain_labels instead
            ΔZ_target, 
            ΔM, 
            ΔY,
            ΔM_pct,
            ΔY_pct
        ))
    end
end

println("\n" * "="^80)
println("Summary Table")
println("="^80)
println(equiv_results)

# ==============================================================================
# COUNTERFACTUAL 3: Decompose Non-Adopters & Target Support
# ==============================================================================

println("\n" * "="^80)
println("COUNTERFACTUAL 3: Decomposing Non-Adoption and Type A Income Support Analysis")
println("="^80)

# Identify never-adopters
never_adopters = findall(vec(sum(results[:baseline][1], dims=2) .== 0))

println("Total never-adopters: $(length(never_adopters))")
println("\nClassifying households...")
println("(Testing if they would adopt CPM with income = \$5000/week)")

# Storage
classification_results = DataFrame(
    HouseholdID = Int[],
    TypeA = Bool[],
    Reason = String[],
    PeriodsWouldAdopt = String[],
    AvgIncome = Float64[],
    SelfReg = Float64[]
)

# Classify each never-adopter
for (idx, i) in enumerate(never_adopters)
    if idx % 100 == 0
        println("  Processed $idx / $(length(never_adopters)) households...")
    end
    
    is_typeA, reason, periods = classify_non_adopter_improved(
        i, results, params1c, best_params, n_c, n_age,
        s, Time, Y, CT, Z, R_bar, hyp, β_p, β_c, φ
    )
    
    avg_income = mean(Y[(i-1)*n_age + 1 : i*n_age])
    sr_i = hyp[3*(i-1)+1]
    periods_str = isempty(periods) ? "none" : join(periods, ",")
    
    push!(classification_results, (i, is_typeA, reason, periods_str, avg_income, sr_i))
end

# Aggregate results
typeA_households = classification_results[classification_results.TypeA, :HouseholdID]
typeB_households = classification_results[.!classification_results.TypeA, :HouseholdID]

println("\n" * "="^70)
println("Classification Summary")
println("="^70)
println("Type A (Constrained): $(length(typeA_households)) households ($(round(100*length(typeA_households)/length(never_adopters), digits=1))%)")
println("Type B (Preference): $(length(typeB_households)) households ($(round(100*length(typeB_households)/length(never_adopters), digits=1))%)")

# Statistics by type
idx_low = findall((hyperactivity_child .>= 1) .& (hyperactivity_child .<= 5))
idx_high = findall((hyperactivity_child .>= 10) .& (hyperactivity_child .<= 11))

for (type_name, type_ids) in [("Type A", typeA_households), ("Type B", typeB_households)]
    if isempty(type_ids)
        println("\n$type_name: No households in this category")
        continue
    end
    
    println("\n$type_name Statistics:")
    
    type_subset = classification_results[in.(classification_results.HouseholdID, Ref(type_ids)), :]
    
    avg_income = mean(type_subset.AvgIncome)
    println("  Avg income: \$$(round(avg_income, digits=2))/week")
    
    n_low_SR = sum(in.(type_subset.SelfReg, Ref(1.0:5.0)))
    n_high_SR = sum(in.(type_subset.SelfReg, Ref(10.0:11.0)))
    println("  Low SR (1-5): $(n_low_SR) ($(round(100*n_low_SR/length(type_ids), digits=1))%)")
    println("  High SR (10-11): $(n_high_SR) ($(round(100*n_high_SR/length(type_ids), digits=1))%)")
end

# ---------------------------
# Step 5: Income Support for Type A
# ---------------------------
if !isempty(typeA_households)
    println("\n" * "="^70)
    println("Income Support Analysis (Type A Only)")
    println("="^70)
    
    println("Computing minimum income supplements needed...")
    
    income_support_results = DataFrame(
        HouseholdID = Int[],
        ΔY_needed = Float64[],
        Baseline_Income = Float64[],
        ΔY_pct = Float64[],
        SelfReg = Float64[]
    )
    
    for (idx, i) in enumerate(typeA_households)
        if idx % 20 == 0
            println("  Processed $idx / $(length(typeA_households))...")
        end
        
        ΔY = find_minimum_income_typeA(
            i, results, params1c, best_params, n_c, n_age,
            s, Time, Y, CT, Z, R_bar, hyp, β_p, β_c, φ
        )
        
        baseline_inc = mean(Y[(i-1)*n_age + 1 : i*n_age])
        ΔY_pct = 100 * ΔY / baseline_inc
        sr_i = hyp[3*(i-1)+1]
        
        push!(income_support_results, (i, ΔY, baseline_inc, ΔY_pct, sr_i))
    end
    
    println("\nIncome Support Summary (All Type A):")
    println("  Mean ΔY: \$$(round(mean(income_support_results.ΔY_needed), digits=2))/week")
    println("  Median ΔY: \$$(round(median(income_support_results.ΔY_needed), digits=2))/week")
    println("  Mean % of income: $(round(mean(income_support_results.ΔY_pct), digits=1))%")
    
    for (sr_name, sr_range) in [("Low SR", 1.0:5.0), ("High SR", 10.0:11.0)]
        subset = income_support_results[in.(income_support_results.SelfReg, Ref(sr_range)), :]
        if nrow(subset) > 0
            println("\n$sr_name Type A Families (n=$(nrow(subset))):")
            println("  Mean ΔY: \$$(round(mean(subset.ΔY_needed), digits=2))/week")
            println("  Mean % of income: $(round(mean(subset.ΔY_pct), digits=1))%")
        end
    end
else
    println("\n" * "="^70)
    println("⚠ No Type A households found!")
    println("="^70)
    println("\nThis means all non-adopters have preferences against CPM,")
    println("even when given very high income (\$5000/week).")
    println("\nPossible explanations:")
    println("  1. Enforcement costs (params[10]) are too high")
    println("  2. Parents have low altruism (φ) toward child outcomes")
    println("  3. Model misspecification in CPM decision")
    println("\nConsider:")
    println("  - Checking params[10] (enforcement cost parameter)")
    println("  - Examining φ (altruism parameter)")
    println("  - Reviewing solve_CPM_grid implementation")
end

println("\n" * "="^80)

# ==============================================================================
# 4. Bootsrapped Standard Errors
# ------------------------------------------------------------------------------

num_bootstrap = 100
param_boot = zeros(num_bootstrap, length(params_all))
derived_boot = zeros(num_bootstrap, 8)  # 8 derived parameters from value_func

child_ids = unique(df_matrix_all[:,1])
H = length(child_ids)

initial_params = copy(params_all)

for b in 1:num_bootstrap
    if b % 10 == 0
        println("Bootstrap iteration $b / $num_bootstrap")
    end

    # Resample children
    sampled_ids = sample(child_ids, H, replace=true)
    df_b = vcat([df_matrix_all[df_matrix_all[:,1] .== id, :] for id in sampled_ids]...)
    
    # Extract resampled data
    s_b = df_b[:, 6]
    Y_b = df_b[:, 5]
    CT_b = zeros(size(df_b, 1))
    R_bar_b = df_b[:, 12]

    try
        res_b = Optim.optimize(
            obj,          # objective function
            initial_params,
            NelderMead(),         # Nelder-Mead algorithm
            Optim.Options(
                iterations = 1000,
                f_tol = 1e-4,
                x_tol = 1e-4,
                show_trace = true,
                #trace_simplex = true
                #time_limit = 7200
            )
        )

        # Store latent parameters
        param_boot[b, :] .= best_params
        
        # Compute and store derived parameters
        boot_results = value_func(Z, params1c, best_params, n_c, n_age, s, Time, Y, CT, R_bar, hyp)
        derived_boot[b, :] .= [boot_results[5:12]...]

    catch e
        println("Bootstrap iteration $b failed: ", e)
        param_boot[b, :] .= NaN
        derived_boot[b, :] .= NaN
    end
end

# Filter successful samples
valid_rows = findall(row -> all(.!isnan.(row)), eachrow(param_boot))
param_boot_clean = param_boot[valid_rows, :]
derived_boot_clean = derived_boot[valid_rows, :]

# Compute SEs for both latent and derived parameters
latent_se = vec(std(param_boot_clean; dims=1))
derived_se = vec(std(derived_boot_clean; dims=1))

# Report latent parameters
param_names = ["μ_α1", "μ_λ0", "μ_λ1",
    "σ_α1", "σ_λ0", "σ_λ1",
    "λ01", "α1λ0", "α1λ1", "ζ", "g"]

println("\n=== Latent Parameters ===")
for i in 1:length(params_all)
    println("$(param_names[i]): $(round(best_params[i]; digits=4)) (SE = $(round(latent_se[i]; digits=4)))")
end

# Report derived parameters
derived_names = ["mean_α1", "std_α1", "mean_λ0", "std_λ0", 
"mean_λ1", "std_λ1", "mean_λ2", "std_λ2"]

println("\n=== Derived Parameters ===")
for i in 1:8
    println("$(derived_names[i]): $(round(derived_baseline[i]; digits=4)) (SE = $(round(derived_se[i]; digits=4)))")
end

latent_se_vec = vec(latent_se)
derived_se_vec = vec(derived_se)

df_se_latent = DataFrame(
    param_id = 1:length(latent_se_vec),
    se = latent_se_vec
)

df_se_derived = DataFrame(
    param_id = 1:length(derived_se_vec),
    se = derived_se_vec
)

CSV.write("latent_params_pref_se.csv", df_se_latent)
CSV.write("params_pref_se.csv", df_se_derived)
