using DataFrames, Statistics
using Statistics, Random, Optim
using Base.Threads

function data_moments(df, n_age)

    # Helper: Get mean and std for a variable conditional on CPM and child age
    function mean_std_by_age(df, var_col::Int)
        means = Float64[]
        stds = Float64[]
        for age in 1:n_age            
            idx = (df[:, var_col] .!= -999) .& (df[:, 2] .== age)
            if any(idx)
                values = df[idx, var_col]
                push!(means, isempty(values) ? 0.0 : mean(values))
                push!(stds, length(values) <= 1 ? 0.0 : std(values, corrected=true))
            else
                push!(means, 0.0)
                push!(stds, 0.0)
            end
        end
        return means, stds
    end

    read_means, read_stds = mean_std_by_age(df, 10)
    hyp_means, hyp_stds = mean_std_by_age(df, 11)
    CPM_mean_age = mean_std_by_age(df, 4)[1]

    # Time investment by CPM × age × school indicator (mean + std)
    time_by_cpm_age = Float64[]
    time_sd_by_cpm_age = Float64[]
    for cpm in (0, 1)
        for age in 1:n_age
            idx = (df[:, 7] .> 0.0) .& (df[:, 4] .== cpm) .& (df[:, 2] .== age)
            if any(idx)
                values = df[idx, 7]
                push!(time_by_cpm_age, mean(values))
                push!(time_sd_by_cpm_age, length(values) <= 1 ? 0.0 : std(values, corrected=true))
            else
                push!(time_by_cpm_age, 0.0)
                push!(time_sd_by_cpm_age, 0.0)
            end
        end
    end

    # Pocket Money by CPM × age (mean + std)
    R_by_cpm_age = Float64[]
    R_sd_by_cpm_age = Float64[]
    for cpm in (0, 1)
        for age in 1:n_age
            #idx = (df[:, 8] .!= -999) .& (df[:, 4] .== cpm) .& (df[:, 2] .== age)
            idx = (df[:, 8] .> 0) .& (df[:, 4] .== cpm) .& (df[:, 2] .== age)
            if any(idx)
                values = df[idx, 8]  # pocket money / income
                push!(R_by_cpm_age, mean(values))
                push!(R_sd_by_cpm_age, length(values) <= 1 ? 0.0 : std(values, corrected=true))
            else
                push!(R_by_cpm_age, 0.0)
                push!(R_sd_by_cpm_age, 0.0)
            end
        end
    end

    # Time Trend
    τ_diff_no_cpm = diff(time_by_cpm_age[1:3])
    τ_diff_cpm = diff(time_by_cpm_age[4:6])
    R_diff_no_cpm = diff(R_by_cpm_age[1:3])
    R_diff_cpm = diff(R_by_cpm_age[4:6])

    # 5. Correlations
    #valid_idx1 = (df[:, 4] .!= -999) .& (df[:, 3] .!= -999) # Mother's education
    #valid_idx2 = (df[:, 4] .!= -999) .& (df[:, 2] .!= -999) # Child's age
    #valid_idx3 = (df[:, 4] .!= -999) .& (df[:, 7] .!= -999) # Study Time
    #valid_idx4 = (df[:, 4] .!= -999) .& (df[:, 10] .!= -999) # Test Scores

    #corr1 = any(valid_idx1) ? cor(df[valid_idx1, 4], df[valid_idx1, 3]) : 0.0
    #corr2 = any(valid_idx2) ? cor(df[valid_idx2, 4], df[valid_idx2, 2]) : 0.0
    #corr3 = any(valid_idx3) ? cor(df[valid_idx3, 4], df[valid_idx3, 7]) : 0.0
    #corr4 = any(valid_idx4) ? cor(df[valid_idx4, 4], df[valid_idx4, 10]) : 0.0

    # Combine all moments into one vector (56 total)
    moments = vcat(
        #τ_means..., τ_stds...,       # 25–36
        time_by_cpm_age..., time_sd_by_cpm_age...,       # 1–12
        #R_means..., R_stds...,       # 37 - 48
        R_by_cpm_age..., R_sd_by_cpm_age...,       # 13–24
        read_means..., read_stds...,       # 25–30
        hyp_means..., hyp_stds...,       # 31 - 36
        CPM_mean_age...,  # 37 – 39
        τ_diff_no_cpm, τ_diff_cpm,
        R_diff_no_cpm, R_diff_cpm
        #corr1, corr2          # 40–41
    )

    return moments
end

function simulated_moments(params1c, params1n, CPM_opt, τ_opt, R_opt, M_opt, df, n_c, n_age)
    # --- Safe helpers ---
    safe_mean(x) = isempty(skipmissing(x)) ? NaN : mean(skipmissing(x))
    safe_std(x)  = length(collect(skipmissing(x))) <= 1 ? 0.0 : std(collect(skipmissing(x)), corrected=true)
    
    # Initialize matrices
    Z_C = zeros(n_c, n_age)
    Z_N = zeros(n_c, n_age)

    # --- Period 1: match data moments ---
    Z_C[:, 1] .= df[df[:, 2] .== 1, 10]
    Z_N[:, 1] .= df[df[:, 2] .== 1, 11]

    # Flatten arrays for CPM, τ, R, M (no filtering, assume R_vec has no zeros)
    CPM_vec  = vec(CPM_opt)
    τ_vec    = vec(τ_opt)
    R_vec    = vec(R_opt)
    M_vec    = vec(M_opt)
    time_vec = repeat(1:n_age, inner=n_c)  # still required for CPM/τ moments

    # Extract medu and income per child and broadcast over time efficiently
    #medu_vec   = repeat(df[1:n_c, 3], inner=n_age)
    income_vec = repeat(df[1:n_c, 5], inner=n_age)

    # Valid mask for filtering
    #valid_mask = repeat(child_valid, inner=n_age)
    #CPM_vec     = CPM_vec[valid_mask]
    #τ_vec       = τ_vec[valid_mask]
    #R_vec       = R_vec[valid_mask]
    #M_vec       = M_vec[valid_mask]
    #time_vec    = time_vec[valid_mask]
    #medu_vec    = medu_vec[valid_mask]
    #income_vec  = income_vec[valid_mask]

    # Simulate skill evolution
    @inbounds for t in 2:n_age
        Z_C[:, t] .= exp.(params1c[(t-1)*5-4] .+ params1c[(t-1)*5-3]*log.(Z_C[:, t-1]) .+ 
                        params1c[(t-1)*5-2]*log.(τ_opt[:, t-1]) .+ 
                        params1c[(t-1)*5-1]*log.(M_opt[:, t-1]) .+ 
                        params1c[(t-1)*5]*log.(Z_N[:, t-1]))
        Z_N[:, t] .= exp.(params1n[(t-1)*5-4] .+ params1n[(t-1)*5-3]*log.(Z_N[:, t-1]) .+
                        params1n[(t-1)*5-2]*log.(τ_opt[:, t-1]) .+ 
                        params1n[(t-1)*5-1]*log.(M_opt[:, t-1]) .+
                        params1n[(t-1)*5]*log.(Z_C[:, t-1]))
    end

    # --- Compute Z_C/Z_N means per age ---
    Z_C_means = [mean(Z_C[:, age]) for age in 1:n_age]
    Z_C_stds  = [std(Z_C[:, age])  for age in 1:n_age]
    Z_N_means = [mean(Z_N[:, age]) for age in 1:n_age]
    Z_N_stds  = [std(Z_N[:, age])  for age in 1:n_age]

    # --- Compute CPM/τ/R moments ---
    CPM_mean_age = [mean(CPM_vec[time_vec .== age]) for age in 1:n_age]

    # --- Vectorized τ moments ---
    τ_by_cpm_age     = [mean(τ_vec[(CPM_vec .== c) .& (time_vec .== a)]) for c in (0,1) for a in 1:n_age]
    τ_sd_by_cpm_age  = [std(τ_vec[(CPM_vec .== c) .& (time_vec .== a)]) for c in (0,1) for a in 1:n_age]
    R_by_cpm_age     = [mean(R_vec[(CPM_vec .== c) .& (time_vec .== a) .& (R_vec .> 0)]) for c in (0,1) for a in 1:n_age]
    R_sd_by_cpm_age  = [std(R_vec[(CPM_vec .== c) .& (time_vec .== a) .& (R_vec .> 0)]) for c in (0,1) for a in 1:n_age]
    
    # --- Vectorized R moments ---
    #R_vals_mat = 100 .* R_vec ./ income_vec

    #R_by_cpm_age = [
    #    mean(R_vals_mat[(CPM_vec .== c) .& (time_vec .== a) .& (R_vec .>= 0)])
    #    for c in (0, 1), a in 1:n_age
    #]

    #R_sd_by_cpm_age = [
    #    std(R_vals_mat[(CPM_vec .== c) .& (time_vec .== a) .& (R_vec .>= 0)])
    #    for c in (0, 1), a in 1:n_age
    #]

    # Time Trend
    τ_diff_no_cpm = diff(τ_by_cpm_age[1:3])
    τ_diff_cpm = diff(τ_by_cpm_age[4:6])
    R_diff_no_cpm = diff(R_by_cpm_age[1:3])
    R_diff_cpm = diff(R_by_cpm_age[4:6])

    # --- Correlations ---
    #valid1 = (medu_vec .!= -999)
    #corr1 = cor(CPM_vec[valid1], medu_vec[valid1])
    #corr2 = cor(CPM_vec, time_vec)

    # --- Combine moments ---
    moments = vcat(
        τ_by_cpm_age..., τ_sd_by_cpm_age...,
        R_by_cpm_age..., R_sd_by_cpm_age...,
        Z_C_means..., Z_C_stds...,
        Z_N_means..., Z_N_stds...,
        CPM_mean_age...,  # 37 – 39
        τ_diff_no_cpm, τ_diff_cpm,
        R_diff_no_cpm, R_diff_cpm
        #corr1, corr2
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

# Compute bootstrapped standard deviations for moments
function bootstrap_moment_std(child_data, n_c, n_age, n_boot=100)
    n_moments = length(data_moments(vcat(child_data...), n_age))
    moment_samples = zeros(n_boot, n_moments)

    Threads.@threads for b in 1:n_boot
        sample_cids = rand(1:n_c, n_c)              # resample children with replacement
        df_sample = vcat(child_data[sample_cids]...)  # concatenate matrices
        try
            moment_samples[b, :] = data_moments(df_sample, n_age)
        catch e
            println("Bootstrap iteration $b failed, filling with zeros.")
            moment_samples[b, :] .= 0.0
        end
    end

    # Compute standard deviations of each moment
    moment_std = std(moment_samples; dims=1, corrected=true)
    return vec(moment_std)
end

function distance_function_optimized(
    params1c, params1n, params, W, n_c, n_age,
    s, Time, Y, CT, Z, emp_moms, df
)
    BIG_PENALTY = 1e10  # large but finite

    try
        # Single call to value function
        results = value_func(Z, params1c, params1n, params, n_c, n_age, s, Time, Y, CT, R_bar)
        CPM_opt, τ_opt, M_opt, R_opt = results[1:4]

        # Compute simulated moments
        sim_moms = simulated_moments(params1c, params1n, CPM_opt, τ_opt, R_opt, M_opt, df, n_c, n_age)

        # Check for NaN/Inf early
        if any(isnan, sim_moms) || any(isinf, sim_moms)
            return BIG_PENALTY
        end

        # Compute weighted squared difference
        # Use @. to avoid repeated allocations
        diff = sim_moms .- emp_moms
        if any(isnan, diff) || any(isinf, diff)
            return BIG_PENALTY
        end

        loss = sum(W.diag .* diff.^2)
        return isnan(loss) || isinf(loss) ? BIG_PENALTY : loss

    catch e
        println("‼️ Exception in distance_function: ", e)
        return BIG_PENALTY
    end
end

function bootstrap_smm(child_data, params1c, params1n, initial_params,
                       n_boot, n_c, n_age, W,
                       s, Time, Y, CT, Z, U)

    n_params = length(initial_params)
    n_derived = 14  # mean/var for α and λ

    all_params_boot = Matrix{Float64}(undef, n_boot, n_params)
    all_derived_boot = Matrix{Float64}(undef, n_boot, n_derived)

    # Identify unique child units (assuming :child_id exists)
    child_ids = unique(child_data[:, 1])
    n_childs = length(child_ids)

    for b in 1:n_boot
        println("Bootstrap iteration $b / $n_boot")

        # --- resample households with replacement (reproducible per b)
        rng_boot = MersenneTwister(4321 + b)
        boot_ids = rand(rng_boot, child_ids, n_childs)
        df_boot  = vcat([child_data[child_data[:, 1] .== id, :] for id in boot_ids]...)

        # --- Step 2: Objective function for this bootstrap
        function bootstrap_obj(params)
            error = distance_function_optimized(
                params1c, params1n, params, W, n_c, n_age,
                s, Time, Y, CT, Z, emp_moms, df_boot
            )
                        
            return isnan(error) || isinf(error) ? 1e10 : error
        end

        # objective for this bootstrap
        bootstrap_obj = params -> begin
            err = distance_function_optimized(params1c, params1n, params,
                                              df_boot, W_boot, n_c, n_age, s, Time, Y, CT, Z, U, emp_moms)
            (isnan(err) || isinf(err)) ? 1e10 : err
        end

        # --- Differential Evolution (trimmed) to avoid anchoring
        de_res = bboptimize(bootstrap_obj;
            SearchRange = bounds,
            Method = :adaptive_de_rand_1_bin,
            NumDimensions = length(initial_params),
            PopulationSize = 150,
            MaxSteps = 10000,
            MaxStepsWithoutProgress = 500,
            FitnessTolerance = 1e-4,
            MinDeltaFitnessTolerance = 1e-4,
            TraceMode = :verbose
        )

        # seed NM with DE minimizer
        start_nm = best_candidate(de_res)  # or de_res.archive_best_candidate.x

        nm_res = Optim.optimize(
            bootstrap_obj,
            start_nm,
            NelderMead(),
            Optim.Options(
                iterations = 50000,
                f_tol = 1e-6,
                x_tol = 1e-6,
                show_trace = true
        ))

        boot_params = Optim.minimizer(nm_res)
        all_params_boot[b, :] .= boot_params

        # derived quantities
        vr = value_func(Z, U, params1c, params1n, boot_params, n_c, n_age, s, Time, Y, CT)
        all_derived_boot[b, :] .= (vr[5:18])  # adjust if indices differ
    end

    return all_params_boot, all_derived_boot
end

function key_data_moms(df, n_age)

    # Helper: Get mean and std for a variable conditional on CPM and child age
    function mean_std_by_age(df, var_col::Int)
        means = Float64[]
        stds = Float64[]
        for age in 1:n_age            
            idx = (df[:, var_col] .!= -999) .& (df[:, 2] .== age)
            if any(idx)
                values = df[idx, var_col]
                push!(means, isempty(values) ? 0.0 : mean(values))
                push!(stds, length(values) <= 1 ? 0.0 : std(values, corrected=true))
            else
                push!(means, 0.0)
                push!(stds, 0.0)
            end
        end
        return means, stds
    end

    τ_means, τ_stds = mean_std_by_age(df, 7)
    read_means, read_stds = mean_std_by_age(df, 10)
    hyp_means, hyp_stds = mean_std_by_age(df, 11)
    CPM_mean_age = mean_std_by_age(df, 4)[1] * 100

    # Pocket Money by CPM × age (mean + std)
    R_share_by_age = Float64[]
    R_share_sd_by_age = Float64[]
    for age in 1:n_age
        idx = (df[:, 8] .> 0.0) .& (df[:, 2] .== age)
        if any(idx)
            values = 100 * df[idx, 8] ./ df[idx, 5]  # pocket money / income
            push!(R_share_by_age, mean(values))
            push!(R_share_sd_by_age, length(values) <= 1 ? 0.0 : std(values, corrected=true))
        else
            push!(R_share_by_age, 0.0)
            push!(R_share_sd_by_age, 0.0)
        end
    end

    # 5. Correlations
    #valid_idx1 = (df[:, 4] .!= -999) .& (df[:, 3] .!= -999) # Mother's education
    #valid_idx2 = (df[:, 4] .!= -999) .& (df[:, 2] .!= -999) # Child's age
    #valid_idx3 = (df[:, 4] .!= -999) .& (df[:, 7] .!= -999) # Study Time
    #valid_idx4 = (df[:, 4] .!= -999) .& (df[:, 10] .!= -999) # Test Scores

    #corr1 = any(valid_idx1) ? cor(df[valid_idx1, 4], df[valid_idx1, 3]) : 0.0
    #corr2 = any(valid_idx2) ? cor(df[valid_idx2, 4], df[valid_idx2, 2]) : 0.0
    #corr3 = any(valid_idx3) ? cor(df[valid_idx3, 4], df[valid_idx3, 7]) : 0.0
    #corr4 = any(valid_idx4) ? cor(df[valid_idx4, 4], df[valid_idx4, 10]) : 0.0

    # Combine all moments into one vector (56 total)
    moments = vcat(
        τ_means..., τ_stds...,       # 25–36
        R_share_by_age..., R_share_sd_by_age...,       # 13–24
        read_means..., read_stds...,       # 25–30
        hyp_means..., hyp_stds...,       # 31 - 36
        CPM_mean_age...
    )

    return moments
end

function key_sim_moms(params1c, params1n, CPM_opt, τ_opt, R_opt, M_opt, df, n_c, n_age)
    # --- Safe helpers ---
    safe_mean(x) = isempty(skipmissing(x)) ? NaN : mean(skipmissing(x))
    safe_std(x)  = length(collect(skipmissing(x))) <= 1 ? 0.0 : std(collect(skipmissing(x)), corrected=true)
    
    # Initialize matrices
    Z_C = zeros(n_c, n_age)
    Z_N = zeros(n_c, n_age)

    # --- Period 1: match data moments ---
    Z_C[:, 1] .= df[df[:, 2] .== 1, 10]
    Z_N[:, 1] .= df[df[:, 2] .== 1, 11]

    # Extract medu and income per child and broadcast over time efficiently
    CPM_vec  = vec(CPM_opt)
    τ_vec    = vec(τ_opt)
    R_vec    = vec(R_opt)
    M_vec    = vec(M_opt)
    time_vec = repeat(1:n_age, inner=n_c)  # still required for CPM/τ moments
    medu_vec   = repeat(df[1:n_c, 3], inner=n_age)
    income_vec = repeat(df[1:n_c, 5], inner=n_age)

    # Simulate skill evolution
    @inbounds for t in 2:n_age
        Z_C[:, t] .= exp.(params1c[(t-1)*5-4] .+ params1c[(t-1)*5-3]*log.(Z_C[:, t-1]) .+ 
                        params1c[(t-1)*5-2]*log.(τ_opt[:, t-1]) .+ 
                        params1c[(t-1)*5-1]*log.(M_opt[:, t-1]) .+ 
                        params1c[(t-1)*5]*log.(Z_N[:, t-1]))
        Z_N[:, t] .= exp.(params1n[(t-1)*5-4] .+ params1n[(t-1)*5-3]*log.(Z_N[:, t-1]) .+
                        params1n[(t-1)*5-2]*log.(τ_opt[:, t-1]) .+ 
                        params1n[(t-1)*5-1]*log.(M_opt[:, t-1]) .+
                        params1n[(t-1)*5]*log.(Z_C[:, t-1]))
    end
    #println("Z_C: ", Z_C, "Z_N: ", Z_N)

    # --- Compute Z_C/Z_N means per age ---
    Z_C_means = [mean(Z_C[:, age]) for age in 1:n_age]
    Z_C_stds  = [std(Z_C[:, age])  for age in 1:n_age]
    Z_N_means = [mean(Z_N[:, age]) for age in 1:n_age]
    Z_N_stds  = [std(Z_N[:, age])  for age in 1:n_age]

    # --- Compute CPM/τ/R moments ---
    CPM_mean_age = [mean(CPM_vec[time_vec .== age]) for age in 1:n_age]*100

    # --- Vectorized τ moments ---
    τ_means = [mean(τ_vec[(time_vec .== age)]) for age in 1:n_age]
    τ_stds = [std(τ_vec[(time_vec .== age)]) for age in 1:n_age]

    # --- Vectorized R moments ---
    R_vals_mat = 100 .* R_vec ./ income_vec
    R_share_by_age     = [mean(R_vals_mat[(time_vec .== a) .& (R_vec .> 0)]) for a in 1:n_age]
    R_share_sd_by_age  = [std(R_vals_mat[(time_vec .== a) .& (R_vec .> 0)]) for a in 1:n_age]

    # --- Correlations ---
    #valid1 = (medu_vec .!= -999)
    #corr1 = cor(CPM_vec[valid1], medu_vec[valid1])
    #corr2 = cor(CPM_vec, time_vec)

    # --- Combine moments ---
    moments = vcat(
        τ_means..., τ_stds...,
        R_share_by_age..., R_share_sd_by_age...,
        Z_C_means..., Z_C_stds...,
        Z_N_means..., Z_N_stds...,
        CPM_mean_age...
    )

    return moments
end
