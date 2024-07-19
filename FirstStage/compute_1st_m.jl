using Distributed
addprocs(4)  # Adjust based on your available CPUs

@everywhere begin
    using Optim, LinearAlgebra, Statistics, Distributions, CSV, DataFrames, Random, ForwardDiff, StatsBase, Clustering, Plots
    include("data_m_all.jl")
    include("em.jl")
end

# Define the path
data_path = "/Users/clairekim/Desktop/Project/Child Development/julia/"

# Read data
data_m = CSV.File(joinpath(data_path, "data_n_meas_m.csv")) |> DataFrame
#data_f_1 = CSV.File(joinpath(data_path, "data_n_meas_1_f.csv")) |> DataFrame
#data_f_2 = CSV.File(joinpath(data_path, "data_n_meas_2_f.csv")) |> DataFrame
#data_f_3 = CSV.File(joinpath(data_path, "data_n_meas_3_f.csv")) |> DataFrame

μ = randn(Float64, 21)
for j in 1:21
    μ[j] = mean(data_f[:, Symbol("meas", j)])
end
μ[1] = μ[4] = μ[7] = μ[10] = μ[14] = μ[18] = 0.0

α = fill(1e-6, 21)
α[1] = α[4] = α[7] = α[10] = α[14] = α[18] = 1.0

# Identify σ
σ = zeros(Float64, 21)
for j in 1:21
    σ[j] = sqrt(var(data_f[:, Symbol("meas", j)]))
end

# Initialize Factor means
k1_init = [
    mean(data_f[:, Symbol("meas", 1)]),
    mean(data_f[:, Symbol("meas", 4)]),
    mean(data_f[:, Symbol("meas", 7)]),
    mean(data_f[:, Symbol("meas", 10)]),
    mean(data_f[:, Symbol("meas", 14)]),
    mean(data_f[:, Symbol("meas", 18)])
]

k2_init = [
    mean(data_f[:, Symbol("meas", 1)]),
    mean(data_f[:, Symbol("meas", 4)]),
    mean(data_f[:, Symbol("meas", 7)]),
    mean(data_f[:, Symbol("meas", 10)]),
    mean(data_f[:, Symbol("meas", 14)]),
    mean(data_f[:, Symbol("meas", 18)])
]

Δ1_init = diagm([σ[1], σ[4], σ[7], σ[10], σ[14], σ[18]]) .+ 1e-6
Δ2_init = diagm([σ[1], σ[4], σ[7], σ[10], σ[14], σ[18]]) .+ 1e-6

#κ_init = [0.25, 0.25, 0.25, 0.25]
κ_init = rand(Float64, 2)
κ_init = κ_init / sum(κ_init)

# Estimate Parameters with Bootstrapped Standard Errors
#@everywhere function bootstrap_estimate(k1, Δ1, k2, Δ2, k3, Δ3, k4, Δ4, κ, data, n_bootstrap, n_iter)
#    compute_final_estimates_and_standard_errors(k1, Δ1, k2, Δ2, k3, Δ3, k4, Δ4, κ, 2, data, n_bootstrap, n_iter)
#end
@everywhere function bootstrap_estimate(k1, Δ1, k2, Δ2, κ, μ, α, σ, data, n_bootstrap, n_iter)
    compute_final_estimates_and_standard_errors(k1, Δ1, k2, Δ2, κ, μ, α, σ, 5, data, n_bootstrap, n_iter)
end

#θ_samples = draw_samples_positive_fast(k1_init, Δ1_init, k2_init, Δ2_init, k3_init, Δ3_init, k4_init, Δ4_init, κ_init, 5, data_m)
#cont = likelihood_cont(μ, α, σ, θ_samples, 5, data_m)
#cens = likelihood_cont(μ, α, σ, θ_samples, 5, data_m)
#posterior = expectation_step_n(k1_init, Δ1_init, k2_init, Δ2_init, k3_init, Δ3_init, k4_init, Δ4_init, κ_init, θ_samples, 5, data_m)
#m_step(posterior, θ_samples, 5, data_m)
#em_algorithm(k1_init, Δ1_init, k2_init, Δ2_init, k3_init, Δ3_init, k4_init, Δ4_init, κ_init, 5, data_m; max_iters=1000)
#em_algorithm(k1_init, Δ1_init, k2_init, Δ2_init, κ_init, 5, data_m; max_iters=1000)

#μ = [mean(data_m[:, Symbol("meas", j)]) for j in 1:21]
#σ = [std(data_m[:, Symbol("meas", j)]) for j in 1:21]   # Calculate standard deviations for meas1 to meas9
#α = ones(size(μ))

#results1_m = bootstrap_estimate(k1_init, Δ1_init, k2_init, Δ2_init, k3_init, Δ3_init, k4_init, Δ4_init, κ_init, data_m, 100, 1000)
results1_m = bootstrap_estimate(k1_init, Δ1_init, k2_init, Δ2_init, κ_init, μ, α, σ, data_m, 100, 1000)

# Extract estimated parameters into DataFrame
df_dist_m = DataFrame(Ψ_A_m = results1_m)

# Write the DataFrame to a CSV file
#CSV.write("input1_m.csv", df_dist_m)
CSV.write("input_m.csv", df_dist_m)

# Replace 'file_path' with the actual path to your CSV file
file_path = "/Users/clairekim/Desktop/Project/Child Development/julia/input_m.csv"

# Read the CSV file into a DataFrame
df = CSV.read(file_path, DataFrame)

# Assuming the dictionary is in the first row and the first column
dict_cell = df[1, 1]  # Adjust the row and column indices if needed

# Parse the dictionary from the cell (assuming the dictionary is stored as a JSON string)
parsed_dict = eval(Meta.parse(dict_cell))

# Extract the "Final Estimates" from the parsed dictionary
final_estimates = parsed_dict["Final Estimates"]

# Extract the final estimates from results1_f
final_estimates = results1_m["Final Estimates"]

# Assign the extracted values to the corresponding variables
k1_fin = final_estimates[1:6]
k2_fin = final_estimates[7:12]
#k3_fin = final_estimates[5:6]
#k4_fin = final_estimates[7:8]
Δ1_fin = reshape(final_estimates[13:48], 6, 6)
Δ2_fin = reshape(final_estimates[49:84], 6, 6)
#Δ3_fin = reshape(final_estimates[17:20], 2, 2)
#Δ4_fin = reshape(final_estimates[21:24], 2, 2)
κ_fin = final_estimates[85:86]
μ = final_estimates[87:107]
α = final_estimates[108:128]
σ = final_estimates[129:149]

# Use the final estimates in subsequent computations
#results2_f = bootstrap_estimate(k1_fin, Δ1_fin, k2_fin, Δ2_fin, k3_fin, Δ3_fin, k4_fin, Δ4_fin, κ_fin, data_f_1, 100, 5000)
results2_m = bootstrap_estimate(k1_init, Δ1_init, k2_init, Δ2_init, κ_init, μ, α, σ, data_m, 100, 5000)

# Extract estimated parameters into DataFrame
df_dist_m = DataFrame(Ψ_A_m = results2_m)

# Write the DataFrame to a CSV file
CSV.write("input_fin_m.csv", df_dist_m)

file_path = "/Users/clairekim/Desktop/Project/Child Development/julia/input_fin_m.csv"

# Read the CSV file into a DataFrame
df = CSV.read(file_path, DataFrame)

# Assuming the dictionary is in the first row and the first column
dict_cell = df[1, 1]  # Adjust the row and column indices if needed

# Parse the dictionary from the cell (assuming the dictionary is stored as a JSON string)
parsed_dict = eval(Meta.parse(dict_cell))

# Extract the "Final Estimates" from the parsed dictionary
final_estimates = parsed_dict["Final Estimates"]

# Assign the extracted values to the corresponding variables
k1 = final_estimates[1:6]
k2 = final_estimates[7:12]
#k3_fin = final_estimates[5:6]
#k4_fin = final_estimates[7:8]
Δ1 = reshape(final_estimates[13:48], 6, 6)
Δ2 = reshape(final_estimates[49:84], 6, 6)
#Δ3_fin = reshape(final_estimates[17:20], 2, 2)
#Δ4_fin = reshape(final_estimates[21:24], 2, 2)
κ = final_estimates[85:86]

θ_m = draw_samples_positive_one(k1, Δ1, k2, Δ2, κ, data_m)
snr_m = noise(α, σ, θ_m)

