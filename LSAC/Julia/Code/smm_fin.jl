function data_moments(df, n_age; core_only=false)

    # Study time by (age, SR type, CPM status) - 12 moments (3 ages × 2 SR × 2 CPM)
    time_moments = Float64[]

    low_sr = [1, 2, 3, 4, 5]
    high_sr = [10, 11]
    sr_categories = [high_sr, low_sr]
    
    for sr_vals in sr_categories
        for cpm in 0:1
            for age in 1:n_age
                idx = (df[:, 7] .> 0) .& (df[:, 4] .== cpm) .& (df[:, 2] .== age) .& 
                    in.(df[:, 13], Ref(sr_vals))
                
                if any(idx)
                    values = df[idx, 7]
                    push!(time_moments, mean(values))
                else
                    push!(time_moments, 0.0)
                end
            end
        end
    end

    # ===== CPM ADOPTION MOMENTS =====
    
    # 1. CPM adoption by (Age × SR) - 6 moments (3 ages × 2 SR)
    cpm_by_age_sr = Float64[]
    for sr_vals in sr_categories
        for age in 1:n_age
            idx = (df[:, 4] .!= -999) .& (df[:, 2] .== age) .& 
                  in.(df[:, 13], Ref(sr_vals))
            if any(idx)
                push!(cpm_by_age_sr, mean(df[idx, 4]))
            else
                push!(cpm_by_age_sr, 0.0)
            end
        end
    end

    # 2. CPM adoption by income quartile × age (12 moments: 4 quartiles × 3 ages)
    inc_q1 = quantile(df[:, 5], 0.25)
    inc_q2 = quantile(df[:, 5], 0.50)
    inc_q3 = quantile(df[:, 5], 0.75)

    cpm_by_income_age = Float64[]
    for q in 1:4
        if q == 1
            q_idx = df[:, 5] .<= inc_q1
        elseif q == 2
            q_idx = (df[:, 5] .> inc_q1) .& (df[:, 5] .<= inc_q2)
        elseif q == 3
            q_idx = (df[:, 5] .> inc_q2) .& (df[:, 5] .<= inc_q3)
        else
            q_idx = df[:, 5] .> inc_q3
        end
        
        for age in 1:n_age
            idx = q_idx .& (df[:, 2] .== age) .& (df[:, 4] .!= -999)
            if any(idx)
                push!(cpm_by_income_age, mean(df[idx, 4]))
            else
                push!(cpm_by_income_age, 0.0)
            end
        end
    end

    # ===== POCKET MONEY MOMENTS =====
    
    # 1. Pocket money share by income quartile (4 moments)
    pm_share_by_income = Float64[]
    for q in 1:4
        if q == 1
            q_idx = df[:, 5] .<= inc_q1
        elseif q == 2
            q_idx = (df[:, 5] .> inc_q1) .& (df[:, 5] .<= inc_q2)
        elseif q == 3
            q_idx = (df[:, 5] .> inc_q2) .& (df[:, 5] .<= inc_q3)
        else
            q_idx = df[:, 5] .> inc_q3
        end
        
        idx = q_idx .& (df[:, 8] .!= -999)
        if any(idx)
            push!(pm_share_by_income, mean(100 .* df[idx, 8] ./ df[idx, 5]))
        else
            push!(pm_share_by_income, 0.0)
        end
    end

    # 2. Mean pocket money share by (Age × CPM status) - 6 moments
    pm_by_age_cpm = Float64[]
    for cpm in 0:1
        for age in 1:n_age
            idx = (df[:, 8] .!= -999) .& (df[:, 4] .== cpm) .& (df[:, 2] .== age)
            if any(idx)
                push!(pm_by_age_cpm, mean(100 .* df[idx, 8] ./ df[idx, 5]))
            else
                push!(pm_by_age_cpm, 0.0)
            end
        end
    end

    # ===== CORRELATION MOMENTS (3 total - among non-CPM only) =====
    
    # 1. Correlation(initial study time, study time growth) among non-CPM children
    initial_time = Float64[]
    time_growth = Float64[]
    
    for child_id in unique(df[:, 1])
        child_idx = df[:, 1] .== child_id
        child_data = df[child_idx, :]
        
        time_order = sortperm(child_data[:, 2])
        child_data = child_data[time_order, :]
        
        if all(child_data[:, 4] .== 0)
            if size(child_data, 1) >= 2 && child_data[1, 7] > 0
                initial = child_data[1, 7]
                final = child_data[end, 7]
                if final > 0
                    growth = final - initial
                    push!(initial_time, initial)
                    push!(time_growth, growth)
                end
            end
        end
    end
    
    corr_initial_growth = length(initial_time) >= 2 ? cor(initial_time, time_growth) : 0.0
    
    # 2. Correlation(avg pocket money, study time growth) among non-CPM children
    avg_pm = Float64[]
    time_growth2 = Float64[]
    
    for child_id in unique(df[:, 1])
        child_idx = df[:, 1] .== child_id
        child_data = df[child_idx, :]
        
        time_order = sortperm(child_data[:, 2])
        child_data = child_data[time_order, :]
        
        if all(child_data[:, 4] .== 0)
            if size(child_data, 1) >= 2 && child_data[1, 7] > 0
                pm_vals = child_data[:, 8] ./ child_data[:, 5]
                valid_pm = pm_vals[pm_vals .!= -999]
                
                if length(valid_pm) > 0
                    avg_pm_child = mean(valid_pm)
                    
                    initial = child_data[1, 7]
                    final = child_data[end, 7]
                    if final > 0 && !isnan(avg_pm_child)
                        growth = final - initial
                        push!(avg_pm, avg_pm_child)
                        push!(time_growth2, growth)
                    end
                end
            end
        end
    end
    
    corr_pm_growth = length(avg_pm) >= 2 ? cor(avg_pm, time_growth2) : 0.0

    # 3. Correlation(avg pocket money, initial study time) among non-CPM children
    avg_pm2 = Float64[]
    initial_time2 = Float64[]
    
    for child_id in unique(df[:, 1])
        child_idx = df[:, 1] .== child_id
        child_data = df[child_idx, :]
        
        time_order = sortperm(child_data[:, 2])
        child_data = child_data[time_order, :]
        
        if all(child_data[:, 4] .== 0)
            if size(child_data, 1) >= 2 && child_data[1, 7] > 0
                pm_vals = child_data[:, 8] ./ child_data[:, 5]
                valid_pm = pm_vals[pm_vals .!= -999]
                
                if length(valid_pm) > 0
                    avg_pm_child = mean(valid_pm)
                    initial = child_data[1, 7]
                    
                    if !isnan(avg_pm_child)
                        push!(avg_pm2, avg_pm_child)
                        push!(initial_time2, initial)
                    end
                end
            end
        end
    end
    
    corr_pm_initial = length(avg_pm2) >= 2 ? cor(avg_pm2, initial_time2) : 0.0

    # ===== CPM TRANSITIONS (3 total) =====
    n_persist_to_1 = 0
    n_from_1 = 0
    n_enter = 0
    n_from_0 = 0

    for child_id in unique(df[:, 1])
        child_idx = df[:, 1] .== child_id
        child_data = df[child_idx, :]
        
        time_order = sortperm(child_data[:, 2])
        child_data = child_data[time_order, :]
        
        for t in 1:(size(child_data, 1)-1)
            cpm_t = child_data[t, 4]
            cpm_t1 = child_data[t+1, 4]
            
            if cpm_t == 1
                n_from_1 += 1
                if cpm_t1 == 1
                    n_persist_to_1 += 1
                end
            elseif cpm_t == 0
                n_from_0 += 1
                if cpm_t1 == 1
                    n_enter += 1
                end
            end
        end
    end

    persistence_rate = n_from_1 > 0 ? n_persist_to_1 / n_from_1 : 0.0
    entry_rate = n_from_0 > 0 ? n_enter / n_from_0 : 0.0
    exit_rate = 1.0 - persistence_rate

    transition_moments = [persistence_rate, entry_rate, exit_rate]

    # ===== COMBINE ALL MOMENTS (Total: 46) =====
    moments = vcat(
        time_moments,                            # 12 moments (study time means)
        cpm_by_age_sr,                          # 6 moments (CPM by age × SR)
        cpm_by_income_age,                      # 12 moments (CPM by income × age)
        pm_share_by_income,                     # 4 moments (PM share by income quartile)
        pm_by_age_cpm,                          # 6 moments (PM share by age × CPM)
        corr_initial_growth,                    # 1 moment
        corr_pm_growth,                         # 1 moment
        corr_pm_initial,                        # 1 moment (NEW)
        transition_moments                       # 3 moments
    )

    return moments
end

function simulated_moments(CPM_opt, τ_opt, R_opt, M_opt, df, n_c, n_age; core_only=false)
    
    # Flatten arrays
    CPM_vec = vec(CPM_opt')
    τ_vec = vec(τ_opt')
    R_vec = vec(R_opt')
    M_vec = vec(M_opt')
    
    # ===== STUDY TIME MOMENTS =====
    time_moments = Float64[]
    
    low_sr = [1, 2, 3, 4, 5]
    high_sr = [10, 11]
    sr_categories = [high_sr, low_sr]
    
    for sr_vals in sr_categories
        for cpm in 0:1
            for age in 1:n_age
                idx = (CPM_vec .== cpm) .& (df[:, 2] .== age) .& 
                      in.(df[:, 13], Ref(sr_vals))
                
                if any(idx)
                    values = τ_vec[idx]
                    push!(time_moments, mean(values))
                else
                    push!(time_moments, 0.0)
                end
            end
        end
    end
    
    # ===== CPM ADOPTION MOMENTS =====
    
    # 1. CPM by (Age × SR) - 6 moments
    cpm_by_age_sr = Float64[]
    for sr_vals in sr_categories
        for age in 1:n_age
            idx = (df[:, 2] .== age) .& in.(df[:, 13], Ref(sr_vals))
            if any(idx)
                push!(cpm_by_age_sr, mean(CPM_vec[idx]))
            else
                push!(cpm_by_age_sr, 0.0)
            end
        end
    end
    
    # 2. CPM by income quartile × age (12 moments)
    inc_q1 = quantile(df[:, 5], 0.25)
    inc_q2 = quantile(df[:, 5], 0.50)
    inc_q3 = quantile(df[:, 5], 0.75)
    
    cpm_by_income_age = Float64[]
    for q in 1:4
        if q == 1
            q_idx = df[:, 5] .<= inc_q1
        elseif q == 2
            q_idx = (df[:, 5] .> inc_q1) .& (df[:, 5] .<= inc_q2)
        elseif q == 3
            q_idx = (df[:, 5] .> inc_q2) .& (df[:, 5] .<= inc_q3)
        else
            q_idx = df[:, 5] .> inc_q3
        end
        
        for age in 1:n_age
            idx = q_idx .& (df[:, 2] .== age)
            if any(idx)
                push!(cpm_by_income_age, mean(CPM_vec[idx]))
            else
                push!(cpm_by_income_age, 0.0)
            end
        end
    end
    
    # ===== POCKET MONEY MOMENTS =====
    
    # 1. PM share by income quartile (4 moments)
    pm_share_by_income = Float64[]
    for q in 1:4
        if q == 1
            q_idx = df[:, 5] .<= inc_q1
        elseif q == 2
            q_idx = (df[:, 5] .> inc_q1) .& (df[:, 5] .<= inc_q2)
        elseif q == 3
            q_idx = (df[:, 5] .> inc_q2) .& (df[:, 5] .<= inc_q3)
        else
            q_idx = df[:, 5] .> inc_q3
        end
        
        idx = q_idx .& (df[:, 5] .> 0)
        if any(idx)
            push!(pm_share_by_income, mean(100 .* R_vec[idx] ./ df[idx, 5]))
        else
            push!(pm_share_by_income, 0.0)
        end
    end
    
    # 2. PM share by (Age × CPM status) - 6 moments
    pm_by_age_cpm = Float64[]
    for cpm in 0:1
        for age in 1:n_age
            idx = (CPM_vec .== cpm) .& (df[:, 2] .== age)
            if any(idx)
                push!(pm_by_age_cpm, mean(100 .* R_vec[idx] ./ df[idx, 5]))
            else
                push!(pm_by_age_cpm, 0.0)
            end
        end
    end
    
    # ===== CORRELATION MOMENTS (3 total - among non-CPM only) =====
    
    # 1. Correlation(initial study time, study time growth) among non-CPM children
    initial_time = Float64[]
    time_growth = Float64[]
    
    for i in 1:n_c
        if all(CPM_opt[i, :] .== 0)
            initial = τ_opt[i, 1]
            final = τ_opt[i, end]
            
            if initial > 0 && final > 0
                growth = final - initial
                push!(initial_time, initial)
                push!(time_growth, growth)
            end
        end
    end
    
    corr_initial_growth = length(initial_time) >= 2 ? cor(initial_time, time_growth) : 0.0
    
    # 2. Correlation(avg pocket money, study time growth) among non-CPM children
    avg_pm = Float64[]
    time_growth2 = Float64[]
    
    for i in 1:n_c
        if all(CPM_opt[i, :] .== 0)
            pm_normalized = R_opt[i, :] ./ df[((i-1)*n_age + 1):(i*n_age), 5]
            avg_pm_child = mean(pm_normalized)
            
            initial = τ_opt[i, 1]
            final = τ_opt[i, end]
            
            if initial > 0 && final > 0
                growth = final - initial
                push!(avg_pm, avg_pm_child)
                push!(time_growth2, growth)
            end
        end
    end
    
    if length(avg_pm) >= 2 && std(avg_pm) > 0 && std(time_growth2) > 0
        corr_pm_growth = cor(avg_pm, time_growth2)
    else
        corr_pm_growth = 0.0
    end

    # 3. Correlation(avg pocket money, initial study time) among non-CPM children
    avg_pm2 = Float64[]
    initial_time2 = Float64[]
    
    for i in 1:n_c
        if all(CPM_opt[i, :] .== 0)
            pm_normalized = R_opt[i, :] ./ df[((i-1)*n_age + 1):(i*n_age), 5]
            avg_pm_child = mean(pm_normalized)
            
            initial = τ_opt[i, 1]
            
            if initial > 0
                push!(avg_pm2, avg_pm_child)
                push!(initial_time2, initial)
            end
        end
    end
    
    if length(avg_pm2) >= 2 && std(avg_pm2) > 0 && std(initial_time2) > 0
        corr_pm_initial = cor(avg_pm2, initial_time2)
    else
        corr_pm_initial = 0.0
    end

    # ===== CPM TRANSITION MOMENTS (3 total) =====
    n_persist_to_1 = 0
    n_from_1 = 0
    n_enter = 0
    n_from_0 = 0
    
    for i in 1:n_c
        for t in 1:(n_age-1)
            cpm_t = CPM_opt[i, t]
            cpm_t1 = CPM_opt[i, t+1]
            
            if cpm_t == 1
                n_from_1 += 1
                if cpm_t1 == 1
                    n_persist_to_1 += 1
                end
            elseif cpm_t == 0
                n_from_0 += 1
                if cpm_t1 == 1
                    n_enter += 1
                end
            end
        end
    end
    
    persistence_rate = n_from_1 > 0 ? n_persist_to_1 / n_from_1 : 0.0
    entry_rate = n_from_0 > 0 ? n_enter / n_from_0 : 0.0
    exit_rate = 1.0 - persistence_rate
    
    transition_moments = [persistence_rate, entry_rate, exit_rate]
    
    # ===== COMBINE ALL MOMENTS (Total: 46) =====
    moments = vcat(
        time_moments,                            # 12 moments
        cpm_by_age_sr,                          # 6 moments
        cpm_by_income_age,                      # 12 moments
        pm_share_by_income,                     # 4 moments
        pm_by_age_cpm,                          # 6 moments
        corr_initial_growth,                    # 1 moment
        corr_pm_growth,                         # 1 moment
        corr_pm_initial,                        # 1 moment (NEW)
        transition_moments                       # 3 moments
    )
    
    return moments
end

# Keep other functions unchanged
function index_by_child(df, n_c)
    child_data = Vector{Matrix{eltype(df)}}(undef, n_c)
    for cid in 1:n_c
        child_data[cid] = df[df[:, 1] .== cid, :]
    end
    return child_data
end

# Update bootstrap function to accept core_only flag
function bootstrap_moment_std(child_data, n_c, n_age, n_boot=100; core_only=false)
    n_moments = length(data_moments(vcat(child_data...), n_age, core_only=core_only))
    moment_samples = zeros(n_boot, n_moments)

    Threads.@threads for b in 1:n_boot
        sample_cids = rand(1:n_c, n_c)
        df_sample = vcat(child_data[sample_cids]...)
        try
            moment_samples[b, :] = data_moments(df_sample, n_age, core_only=core_only)
        catch e
            println("Bootstrap iteration $b failed, filling with zeros.")
            moment_samples[b, :] .= 0.0
        end
    end

    moment_std = std(moment_samples; dims=1, corrected=true)
    return vec(moment_std)
end

# Update distance function to accept core_only flag
function distance_function_optimized(
    params1c, params, W, n_c, n_age,
    s, Time, Y, CT, Z, emp_moms, df, hyp; core_only=false)
    BIG_PENALTY = 1e10

    try
        results = value_func(Z, params1c, params, n_c, n_age, s, Time, Y, CT, R_bar, hyp)
        CPM_opt, τ_opt, M_opt, R_opt = results[1:4]

        sim_moms = simulated_moments(CPM_opt, τ_opt, R_opt, M_opt, df, n_c, n_age, core_only=core_only)

        if any(isnan, sim_moms) || any(isinf, sim_moms)
            return BIG_PENALTY
        end

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
