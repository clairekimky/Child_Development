#using Distributed
#addprocs(4)  # Adjust based on your available CPUs

cd("/Users/clairekim/Desktop/Project/Child Development/julia")

#@everywhere using Optim, LinearAlgebra, Statistics, Distributions, CSV, DataFrames, Random, ForwardDiff, StatsBase
using Optim, LinearAlgebra, Statistics, Distributions, CSV, DataFrames, Random, ForwardDiff, StatsBase

# Load necessary files on all workers
#@everywhere include("data_meas_f_temp.jl")
#@everywhere include("data_meas_m_temp.jl")
#@everywhere include("em_mvn.jl")

include("data_f.jl")
include("em.jl")

# Read data (only on the master process)
data_f_1 = CSV.File(joinpath("/Users/clairekim/Desktop/Project/Child Development/julia/", "data_n_meas_1_f.csv")) |> DataFrame # Female
data_f_2 = CSV.File(joinpath("/Users/clairekim/Desktop/Project/Child Development/julia/", "data_n_meas_2_f.csv")) |> DataFrame # Female
data_f_3 = CSV.File(joinpath("/Users/clairekim/Desktop/Project/Child Development/julia/", "data_n_meas_3_f.csv")) |> DataFrame # Female


# Broadcast data to all workers
#@everywhere data_f = $data_f
#@everywhere data_m = $data_m

# Get unique individual IDs
unique_female_ids_1 = unique(data_f_1.pid)
unique_female_ids_2 = unique(data_f_2.pid)
unique_female_ids_3 = unique(data_f_3.pid)

# Set the initial parameters
k1_init = randn(Float64, 2) * 0.1
k2_init = randn(Float64, 2) * 0.1
k3_init = randn(Float64, 2) * 0.1
k4_init = randn(Float64, 2) * 0.1

Δ1_init = rand() * I(2)
Δ2_init = rand() * I(2)
Δ3_init = rand() * I(2)
Δ4_init = rand() * I(2)

# Define initial parameters
# (1) κ
κ_init = [0.25, 0.25, 0.25, 0.25]
# (2) k
#k1_init = [101.708433333333333, 2.1127415]
#k2_init = [102.6209, 2.048843]
#k3_init = [106.5457, 1.562963]
#k4_init = [95.9587, 2.126289]
# (3) Δ
#B1 =  generate_invertible_matrix()
#Δ1_init = transpose(B1) * B1
#B2 =  generate_invertible_matrix()
#Δ2_init = transpose(B2) * B2
#B3 =  generate_invertible_matrix()
#Δ3_init = transpose(B3) * B3
#B4 =  generate_invertible_matrix()
#Δ4_init = transpose(B4) * B4

# Estimation

# (1) Female
#θ = draw_samples_positive(k1_init, Δ1_init, k2_init, Δ2_init, k3_init, Δ3_init, k4_init, Δ4_init, κ_init, 10, data_f_1) ## n by R by 2 array
#cont = likelihood_cont(μ, α, σ, θ, 10, data_f_1)
#cens = likelihood_cens(μ, α, σ, θ, 10, data_f_1)
#expectation_step_n(k1_init, Δ1_init, k2_init, Δ2_init, k3_init, Δ3_init, k4_init, Δ4_init, κ_init, θ, 10, data_f_1)
#pdf(Normal(μ[3] + α[3] * θ[1, 1, 1], 95.9587), data_f_1[1, Symbol("meas", 3)]) 
#cont[1, 1] * cens[1, 1]

## Age 10

μ = [mean(data_f_1[:, Symbol("meas", j)]) for j in 1:7]
α = [randn() * 0.1 + 1 for j in 1:7]
α[3:4] .= 1.0
σ = [std(data_f_1[:, Symbol("meas", j)]) for j in 1:7]

# Estimate Parameters with Bootstrapped Standard Errors
results1_f = compute_final_estimates_and_standard_errors(k1_init, Δ1_init, k2_init, Δ2_init, k3_init, Δ3_init, k4_init, Δ4_init, κ_init, 5, data_f_1, 100, 1000)

# Extract estimated parameters into DataFrame
df_dist_f = DataFrame(Ψ_A_f = results1_f)

# Write the DataFrame to a CSV file
CSV.write("input1_f.csv", df_dist_f)

## Age 11 - 12

μ = [mean(data_f_1[:, Symbol("meas", j)]) for j in 1:7]
α = [randn() * 0.1 + 1 for j in 1:7]
α[3:4] .= 1.0
σ = [std(data_f_2[:, Symbol("meas", j)]) for j in 1:7]

# Estimate Parameters with Bootstrapped Standard Errors
results2_f = compute_final_estimates_and_standard_errors(k1_init, Δ1_init, k2_init, Δ2_init, k3_init, Δ3_init, k4_init, Δ4_init, κ_init, 5, data_f_2, 100, 1000)

# Extract estimated parameters into DataFrame
df_dist_f = DataFrame(Ψ_A_f = results2_f)

# Write the DataFrame to a CSV file
CSV.write("input2_f.csv", df_dist_f)

## Age 13- 14

μ = [mean(data_f_1[:, Symbol("meas", j)]) for j in 1:7]
α = [randn() * 0.1 + 1 for j in 1:7]
α[3:4] .= 1.0
σ = [std(data_f_3[:, Symbol("meas", j)]) for j in 1:7]

# Estimate Parameters with Bootstrapped Standard Errors
results3_f = compute_final_estimates_and_standard_errors(k1_init, Δ1_init, k2_init, Δ2_init, k3_init, Δ3_init, k4_init, Δ4_init, κ_init, 5, data_f_3, 100, 1000)

# Extract estimated parameters into DataFrame
df_dist_f = DataFrame(Ψ_A_f = results3_f)

# Write the DataFrame to a CSV file
CSV.write("input3_f.csv", df_dist_f)