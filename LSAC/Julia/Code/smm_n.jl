using DataFrames, Statistics, LinearAlgebra
using Statistics, Random, Optim
using Base.Threads

function data_moments(df, n_age)

    # Helper: Get mean and std for a variable conditional on CPM and child age
    function mean_std_by_cpm_age(df, var_col::Int)
        means = Float64[]
        stds = Float64[]
        for cpm in (0, 1), age in 1:n_age
            idx = (df[:, var_col] .!= -999) .& (df[:, 4] .== cpm) .& (df[:, 2] .== age)
            push!(means, mean(df[idx, var_col]))
            push!(stds, std(df[idx, var_col], corrected=true))
        end
        return means, stds
    end

    # 1. τ (col 8): Study time
    τ_means, τ_stds = mean_std_by_cpm_age(df, 7)

    # 2. R (col 10)
    R_means, R_stds = mean_std_by_cpm_age(df, 9)

    # 3. M (col 11)
    #M_means, M_stds = mean_std_by_cpm_age(df, 10)

    # 4. Mean CPM (col 5) overall and by age
    CPM_mean_all = mean(df[df[:, 4] .!= -999, 4])
    CPM_mean_age = [
        mean(df[(df[:, 4] .!= -999) .& (df[:, 2] .== age), 4]) for age in 1:n_age
    ]

    # 5. Correlations
    corr1 = cor(df[(df[:, 4] .!= -999) .& (df[:, 3] .!= -999), 4], df[(df[:, 4] .!= -999) .& (df[:, 3] .!= -999), 3])
    corr2 = cor(df[(df[:, 4] .!= -999) .& (df[:, 8] .!= -999), 4], df[(df[:, 4] .!= -999) .& (df[:, 8] .!= -999), 8])
    corr3 = cor(df[(df[:, 4] .!= -999) .& (df[:, 7] .!= -999), 4], df[(df[:, 4] .!= -999) .& (df[:, 7] .!= -999), 7])

    # Combine all moments into one vector (43 total)
    moments = vcat(
        τ_means..., τ_stds...,       # 1–12
        R_means..., R_stds...,       # 13–24
        #M_means..., M_stds...,       # 25–36
        CPM_mean_all, CPM_mean_age...,  # 37–40
        corr1, corr2, corr3          # 41–43
    )

    return moments
end

function simulated_moments(CPM_opt, τ_opt, R_opt, M_opt, df)
    # Flatten the arrays to vectors for easy indexing
    CPM_vec = vec(CPM_opt)
    τ_vec = vec(τ_opt)
    R_vec = vec(R_opt)
    M_vec = vec(M_opt)

    n_c, n_age = size(CPM_opt)
    time_vec = repeat(1:n_age, n_c)

    # Extract exogenous variables from original data
    cage_vec = df[:, 3]  # Child's actual age
    medu_vec = df[:, 8]  # Mother's education

    # 1. τ (study time): mean and std by CPM and child age
    τ_means = Float64[]
    τ_stds = Float64[]
    for cpm in (0, 1), age in 1:n_age
        idx = (CPM_vec .== cpm) .& (time_vec .== age)
        push!(τ_means, mean(τ_vec[idx]))
        push!(τ_stds, std(τ_vec[idx], corrected=true))
    end

    # 2. R: mean and std by CPM and child age
    R_means = Float64[]
    R_stds = Float64[]
    for cpm in (0, 1), age in 1:n_age
        idx = (CPM_vec .== cpm) .& (time_vec .== age)
        push!(R_means, mean(R_vec[idx]))
        push!(R_stds, std(R_vec[idx], corrected=true))
    end

    # 3. M: mean and std by CPM and child age
    #M_means = Float64[]
    #M_stds = Float64[]
    #for cpm in (0, 1), age in 1:n_age
    #    idx = (CPM_vec .== cpm) .& (time_vec .== age)
    #    push!(M_means, mean(M_vec[idx]))
    #    push!(M_stds, std(M_vec[idx], corrected=true))
    #end

    # 4. Mean CPM overall and by child age
    CPM_mean_all = mean(CPM_vec)
    CPM_mean_age = [
        mean(CPM_vec[time_vec .== age]) for age in 1:n_age
    ]

    # 5. Correlations with exogenous variables
    valid1 = (CPM_vec .!= -999) .& (cage_vec .!= -999)
    valid2 = (CPM_vec .!= -999) .& (medu_vec .!= -999)
    valid3 = (CPM_vec .!= -999) .& (τ_vec .!= -999)

    corr1 = cor(CPM_vec[valid1], cage_vec[valid1])
    corr2 = cor(CPM_vec[valid2], medu_vec[valid2])
    corr3 = cor(CPM_vec[valid3], τ_vec[valid3])

    # Combine all 43 moments in correct order
    moments = vcat(
        τ_means..., τ_stds...,       # 1–12
        R_means..., R_stds...,       # 13–24
        #M_means..., M_stds...,       # 25–36
        CPM_mean_all, CPM_mean_age...,  # 37–40
        corr1, corr2, corr3          # 41–43
    )

    return moments
end

function index_by_child(df, n_c)
    child_data = Vector{Matrix{eltype(df)}}(undef, n_c)
    for cid in 1:n_c
        child_data[cid] = df[df[:, 1] .== cid, :]
    end
    return child_data
end

function bootstrap_moment_variances_fast(child_data, n_c, n_age, n_boot=100)
    n_moments = length(data_moments(vcat(child_data...), n_age))
    moment_samples = zeros(n_boot, n_moments)

    Threads.@threads for i in 1:n_boot
        sample_cids = rand(1:n_c, n_c)
        df_sample_mat = vcat(child_data[sample_cids]...)
        moment_samples[i, :] = data_moments(df_sample_mat, n_age)
    end

    moment_vars = var(moment_samples; dims=1, corrected=true)
    return vec(moment_vars)
end

function distance_function(params1c, params1n, params, rng, df, W, n_c, n_age, s, Time, Y, CT)
    println("Starting optimization process...")

    results = value_func(rng, params1c, params1n, params, n_c, n_age, s, Time, Y, CT)
    CPM_opt, τ_opt, M_opt, R_opt = results[1], results[2], results[3], results[4]

    sim_moms = simulated_moments(CPM_opt, τ_opt, R_opt, M_opt, df)
    println("Simulated Moments: ", sim_moms)
    emp_moms = data_moments(df, n_age)
    #println("Data Moments: ", emp_moms)

    # Early return if moments contain NaNs or Infs
    if any(isnan, sim_moms) || any(isinf, sim_moms) ||
       any(isnan, emp_moms) || any(isinf, emp_moms)
       #println("Error: Moments contain NaN or Inf values.")
        return Inf
    end

    diff = sim_moms .- emp_moms

    # Early return if differences or weight matrix have NaN/Inf
    if any(isnan, diff) || any(isinf, diff) ||
       any(isnan, W.diag) || any(isinf, W.diag)
       #println("Error: Differences or weight matrix contain NaN or Inf values.")
        return Inf
    end

    dist = diff' * W * diff

    println("Updated Error: ", dist)

    return isnan(dist) || isinf(dist) ? Inf : dist
end

function bootstrap_smm(child_data, params1c, params1n, initial_params,
                       n_boot, n_c, n_age, W,
                       s, Time, Y, CT, opt_options_boot)

    n_params = length(initial_params)
    n_derived = 14  # mean/var for α and λ

    all_params_boot = Matrix{Float64}(undef, n_boot, n_params)
    all_derived_boot = Matrix{Float64}(undef, n_boot, n_derived)

    for b in 1:n_boot
        # Use a fixed seed for the RNG to ensure consistent random draws for each bootstrap iteration
        rng = MersenneTwister(1234)  # Fix the seed for reproducibility

        boot_ids = rand(rng, 1:n_c, n_c)
        df_boot = vcat(child_data[boot_ids]...)

        obj(params) = distance_function(params1c, params1n, params, rng, df_boot, W, n_c, n_age, s, Time, Y, CT)

        result = optimize(obj, initial_params, NelderMead(), opt_options_boot)
        boot_params = Optim.minimizer(result)
        all_params_boot[b, :] .= boot_params

        # Derived parameters from value_func
        results = value_func(rng, params1c, params1n, boot_params, n_c, n_age, s, Time, Y, CT)

        all_derived_boot[b, :] .= [
            results[5], results[6],
            results[7], results[8],
            results[9], results[10],
            results[11], results[12],
            results[13], results[14],
            results[15], results[16],
            results[17], results[18]
        ]
    end

    return all_params_boot, all_derived_boot
end
