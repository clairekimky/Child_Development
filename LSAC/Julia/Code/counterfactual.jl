
# ==============================================================================
# Project Title: The Dynamics of Parent-Child Interactions in Shaping Cognitive and
# Non-cognitive Development during Adolescence
# Author: Claire Kim
# Institution: University of Wisconsin-Madison
# Start Date: 04/28/2025
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
include("backward_induction.jl")
include("theta.jl")
include("smm.jl")

# ==============================================================================
# Counterfactual Experiments
# ------------------------------------------------------------------------------

# (1) Quantifying overall impact of CPM

# Counterfactual Moments
function cf_est(df_matrix)
    #println("Generating moments...")

    #(1) Child Time Inputs by Child Age
    mom1 = zeros(3, 1)
    for j = 1:n_age
        mom1[j, 1] = mean(df_matrix[(df_matrix[:, 2] .== j) .& (df_matrix[:, 10] .!= -999), 10])
    end

    #(2) CPM Choice
    mom2 = zeros(3, 1)

    for j = 1:n_age
        mom2[j, 1] = mean(df_matrix[(df_matrix[:, 2] .== j) .& (df_matrix[:, 7] .!= -999), 7])
    end

    moment = hcat(mom1', mom2')
    #println("Moments generated.")
    return moment
end

function simulate_skills_dynamics(params1, params2, df_matrix)

    # Initialize
    avgcog = zeros(1, 4)
    avgncog = zeros(1, 4)

    # Helper function to filter and extract relevant values
    function extract_vars(df, t, col_τ, col_hyp, col_M_or_cpm)
        rows = filter(row -> row[2] == t - 1 &&
                             row[col_τ] > 0.0 &&
                             row[col_hyp] != -999 &&
                             row[col_M_or_cpm] != -999, eachrow(df))
        log_τ = log.([row[col_τ] for row in rows])
        log_hyp = log.([row[col_hyp] for row in rows])
        M_or_cpm = [row[col_M_or_cpm] for row in rows]
        return log_τ, log_hyp, M_or_cpm
    end

    # --- Cognitive skill dynamics ---
    for t in 2:4
        # Filter rows for this specific t
        log_τ, log_hyp, log_M = extract_vars(df_matrix, t, 10, 14, 13)
        interaction = log_τ .* log_hyp

        # Compute average moments
        mean_log_τ = mean(log_τ) # E(ln τ_{c, t-1})
        mean_log_M = mean(log_M) # E(ln M_{t-1})
        mean_interaction = mean(interaction)

        # Compute avgcog recursively
        if t == 2
            # E(ln θ_2^C)
            avgcog[1, t] = params1[1] +
                        params1[2] * avgcog[1, t - 1] +
                        params1[3] * mean_log_τ +
                        params1[4] * mean_log_M +
                        params1[5] * mean_interaction
        elseif t == 3
            # E(ln θ_3^C)
            avgcog[1, t] = params1[6] +
                        params1[7] * avgcog[1, t - 1] +
                        params1[8] * mean_log_τ +
                        params1[9] * mean_log_M +
                        params1[10] * mean_interaction
        elseif t == 4
            # E(ln θ_4^C)
            avgcog[1, t] = params1[11] +
                        params1[12] * avgcog[1, t - 1] +
                        params1[13] * mean_log_τ +
                        params1[14] * mean_log_M +
                        params1[15] * mean_interaction
        end
    end

    # --- Non-cognitive skill dynamics ---
    for t in 2:4
        log_τ, log_hyp, cpm = extract_vars(df_matrix, t, 10, 14, 7)
        mean_log_hyp = mean(log_hyp)
        mean_cpm = mean(cpm)
        mean_log_τ = mean(log_τ)

        # Compute avgcog recursively
        if t == 2
            avgncog[1, 1] = mean_log_hyp
            avgncog[1, t] = params2[12] +
                        params2[13] * avgncog[1, 1] +
                        params2[14] * mean_cpm + params2[15]*mean_log_τ
        elseif t == 3
            avgncog[1, t] = params2[16] +
                        params2[17] * avgncog[1, t-1] +
                        params2[18] * mean_cpm + params2[19]*mean_log_τ
        elseif t == 4
            avgncog[1, t] = params2[20] +
                        params2[21] * avgncog[1, t-1] +
                        params2[22] * mean_cpm + params2[23]*mean_log_τ
        end
    end

    return avgcog, avgncog
end

function GenAvgMoments_cf(params1, params2, df_matrix)
    moms_b = zeros(1,6) # Study Time and CPM choice
    moms_cf1 = zeros(1,6)
    moms_cf2 = zeros(1,6)
    moms_cf3 = zeros(1,6)
    moms_cf4 = zeros(1,6)
    avgcog_b = zeros(1,4)
    avgcog_cf1 = zeros(1,4)
    avgcog_cf2 = zeros(1,4)
    avgcog_cf3 = zeros(1,4)
    avgcog_cf4 = zeros(1,4)
    avgncog_b = zeros(1,4)
    avgncog_cf1 = zeros(1,4)
    avgncog_cf2 = zeros(1,4)
    avgncog_cf3 = zeros(1,4)
    avgncog_cf4 = zeros(1,4)
    Vc = zeros(1, 5)
    Vp = zeros(1, 5)

    for q = 1:10
        data_b, data_cf1, data_cf2, data_cf3, data_cf4 = Simulate_all(params1, params2, n_c, n_age, s, Time, df_matrix, Y)
        
        moms_b = vcat(moms_b, cf_est(data_b))
        avgcog_tmp, avgncog_tmp = simulate_skills_dynamics(params1, params2, data_b)
        avgcog_b = vcat(avgcog_b, avgcog_tmp)
        avgncog_b = vcat(avgncog_b, avgncog_tmp)
    
        moms_cf1 = vcat(moms_cf1, cf_est(data_cf1))
        avgcog_tmp, avgncog_tmp = simulate_skills_dynamics(params1, params2, data_cf1)
        avgcog_cf1 = vcat(avgcog_cf1, avgcog_tmp)
        avgncog_cf1 = vcat(avgncog_cf1, avgncog_tmp)
    
        moms_cf2 = vcat(moms_cf2, cf_est(data_cf2))
        avgcog_tmp, avgncog_tmp = simulate_skills_dynamics(params1, params2, data_cf2)
        avgcog_cf2 = vcat(avgcog_cf2, avgcog_tmp)
        avgncog_cf2 = vcat(avgncog_cf2, avgncog_tmp)

        moms_cf3 = vcat(moms_cf3, cf_est(data_cf3))
        avgcog_tmp, avgncog_tmp = simulate_skills_dynamics(params1, params2, data_cf3)
        avgcog_cf3 = vcat(avgcog_cf3, avgcog_tmp)
        avgncog_cf3 = vcat(avgncog_cf3, avgncog_tmp)
    
        moms_cf4 = vcat(moms_cf4, cf_est(data_cf4))
        avgcog_tmp, avgncog_tmp = simulate_skills_dynamics(params1, params2, data_cf4)
        avgcog_cf4 = vcat(avgcog_cf4, avgcog_tmp)
        avgncog_cf4 = vcat(avgncog_cf4, avgncog_tmp)
    end    

    moms_b = moms_b[2:10+1, :]
    moms_cf1 = moms_cf1[2:10+1, :]
    moms_cf2 = moms_cf2[2:10+1, :]
    moms_cf3 = moms_cf3[2:10+1, :]
    moms_cf4 = moms_cf4[2:10+1, :]

    avgmom_b = mean(moms_b, dims=1)
    avgmom_cf1 = mean(moms_cf1, dims=1)
    avgmom_cf2 = mean(moms_cf2, dims=1)
    avgmom_cf3 = mean(moms_cf3, dims=1)
    avgmom_cf4 = mean(moms_cf4, dims=1)

    avgcog_b = avgcog_b[2:10+1, :]
    avgcog_b = mean(avgcog_b, dims=1)
    avgcog_cf1 = avgcog_cf1[2:10+1, :]
    avgcog_cf1 = mean(avgcog_cf1, dims=1)
    avgcog_cf2 = avgcog_cf2[2:10+1, :]
    avgcog_cf2 = mean(avgcog_cf2, dims=1)
    avgcog_cf3 = avgcog_cf3[2:10+1, :]
    avgcog_cf3 = mean(avgcog_cf3, dims=1)
    avgcog_cf4 = avgcog_cf4[2:10+1, :]
    avgcog_cf4 = mean(avgcog_cf4, dims=1)
    
    avgncog_b = avgncog_b[2:10+1, :]
    avgncog_b = mean(avgncog_b, dims=1)
    avgncog_cf1 = avgncog_cf1[2:10+1, :]
    avgncog_cf1 = mean(avgncog_cf1, dims=1)
    avgncog_cf2 = avgncog_cf2[2:10+1, :]
    avgncog_cf2 = mean(avgncog_cf2, dims=1)
    avgncog_cf3 = avgncog_cf3[2:10+1, :]
    avgncog_cf3 = mean(avgncog_cf3, dims=1)
    avgncog_cf4 = avgncog_cf4[2:10+1, :]
    avgncog_cf4 = mean(avgncog_cf4, dims=1)

    res = value_func(params1, params2, n_c, n_age, ncog, s, Time, Y)
    Vc[1] = mean(res[31][:, 3]) + res[23]*avgcog_b[3]+β_c*params1[12]*avgcog_b[3]*mean(res[41][:, 3])
    Vc[2] = mean(res[32][:, 3]) + res[23]*avgcog_cf1[3]+β_c*params1[12]*avgcog_cf1[3]*mean(res[41][:, 3])
    Vc[3] = mean(res[33][:, 3]) + res[23]*avgcog_cf2[3]+β_c*params1[12]*avgcog_cf2[3]*mean(res[41][:, 3])
    Vc[4] = mean(res[34][:, 3]) + res[23]*avgcog_cf3[3]+β_c*params1[12]*avgcog_cf3[3]*mean(res[41][:, 3])
    Vc[5] = mean(res[35][:, 3]) + res[23]*avgcog_cf4[3]+β_c*params1[12]*avgcog_cf4[3]*mean(res[41][:, 3])
    Vp[1] = mean(res[36][:, 3]) + ((1-φ)*res[17]+φ*res[23])*avgcog_b[3]+β_p*params1[12]*avgcog_b[3]*mean(res[43][:, 3])
    Vp[2] = mean(res[37][:, 3]) + ((1-φ)*res[17]+φ*res[23])*avgcog_cf1[3]+β_p*params1[12]*avgcog_cf1[3]*mean(res[43][:, 3])
    Vp[3] = mean(res[38][:, 3]) + ((1-φ)*res[17]+φ*res[23])*avgcog_cf2[3]+β_p*params1[12]*avgcog_cf2[3]*mean(res[43][:, 3])
    Vp[4] = mean(res[39][:, 3]) + ((1-φ)*res[17]+φ*res[23])*avgcog_cf3[3]+β_p*params1[12]*avgcog_cf3[3]*mean(res[43][:, 3])
    Vp[5] = mean(res[40][:, 3]) + ((1-φ)*res[17]+φ*res[23])*avgcog_cf4[3]+β_p*params1[12]*avgcog_cf4[3]*mean(res[43][:, 3])

    return avgmom_b, avgmom_cf1, avgmom_cf2, avgmom_cf3, avgmom_cf4,
     avgcog_b, avgcog_cf1, avgcog_cf2, avgcog_cf3, avgcog_cf4,
      avgncog_b, avgncog_cf1, avgncog_cf2, avgncog_cf3, avgncog_cf4, Vc, Vp
end

# (2) Quantifying heterogeneous impact of CPM by SES

function cf_gap(df_matrix)
    #println("Generating moments...")

    #(1) Child Time Inputs by Child Age
    mom1_h = zeros(3, 1)
    for j = 1:n_age
        mom1_h[j, 1] = mean(df_matrix[(df_matrix[:, 2] .== j) .& (df_matrix[:, 10] .!= -999) .& (df_matrix[:, 11] .== 1), 10])
    end

    #(2) CPM Choice
    mom2_h= zeros(3, 1)

    for j = 1:n_age
        mom2_h[j, 1] = mean(df_matrix[(df_matrix[:, 2] .== j) .& (df_matrix[:, 7] .!= -999) .& (df_matrix[:, 11] .== 1), 7])
    end

    mom1_l = zeros(3, 1)
    for j = 1:n_age
        mom1_l[j, 1] = mean(df_matrix[(df_matrix[:, 2] .== j) .& (df_matrix[:, 10] .!= -999) .& (df_matrix[:, 11] .== 0), 10])
    end

    #(2) CPM Choice
    mom2_l= zeros(3, 1)

    for j = 1:n_age
        mom2_l[j, 1] = mean(df_matrix[(df_matrix[:, 2] .== j) .& (df_matrix[:, 7] .!= -999) .& (df_matrix[:, 11] .== 0), 7])
    end

    moment = hcat(mom1_h', mom2_h', mom1_l', mom2_l')
    #println("Moments generated.")
    return moment
end

function simulate_gap(params1, params2, df_matrix)
    avgcog_h = zeros(1, 4)
    avgncog_h = zeros(1, 4)
    avgcog_l = zeros(1, 4)
    avgncog_l = zeros(1, 4)

    # Helper: filter and extract cognitive variables
    function get_cog_vars(df, t, group)
        rows = filter(row -> row[2] == t - 1 &&
                              row[11] == group &&
                              row[14] != -999 &&
                              row[10] > 0 &&
                              row[13] != -999, eachrow(df))
        log_τ = log.([row[10] for row in rows])
        log_M = log.([row[13] for row in rows])
        log_hyp = log.([row[14] for row in rows])
        interaction = log_τ .* log_hyp
        return log_τ, log_M, interaction
    end

    # Helper: filter and extract non-cognitive variables
    function get_noncog_vars(df, t, group)
        rows = filter(row -> row[2] == t - 1 &&
                              row[11] == group &&
                              row[14] != -999 &&
                              row[10] > 0 &&
                              row[7] != -999, eachrow(df))
        log_hyp = log.([row[14] for row in rows])
        cpm = [row[7] for row in rows]
        log_τ = log.([row[10] for row in rows])
        return log_hyp, cpm, log_τ
    end

    for t in 2:4
        # Cognitive variables
        log_τ_h, log_M_h, int_h = get_cog_vars(df_matrix, t, 1)
        log_τ_l, log_M_l, int_l = get_cog_vars(df_matrix, t, 0)

        mean_log_τ_h, mean_log_M_h, mean_int_h = mean(log_τ_h), mean(log_M_h), mean(int_h)
        mean_log_τ_l, mean_log_M_l, mean_int_l = mean(log_τ_l), mean(log_M_l), mean(int_l)

        p = params1
        if t == 2
            avgcog_h[1, t] = p[1] + p[2]*avgcog_h[1, t-1] + p[3]*mean_log_τ_h + p[4]*mean_log_M_h + p[5]*mean_int_h
            avgcog_l[1, t] = p[1] + p[2]*avgcog_l[1, t-1] + p[3]*mean_log_τ_l + p[4]*mean_log_M_l + p[5]*mean_int_l
        elseif t == 3
            avgcog_h[1, t] = p[6] + p[7]*avgcog_h[1, t-1] + p[8]*mean_log_τ_h + p[9]*mean_log_M_h + p[10]*mean_int_h
            avgcog_l[1, t] = p[6] + p[7]*avgcog_l[1, t-1] + p[8]*mean_log_τ_l + p[9]*mean_log_M_l + p[10]*mean_int_l
        elseif t == 4
            avgcog_h[1, t] = p[11] + p[12]*avgcog_h[1, t-1] + p[13]*mean_log_τ_h + p[14]*mean_log_M_h + p[15]*mean_int_h
            avgcog_l[1, t] = p[11] + p[12]*avgcog_l[1, t-1] + p[13]*mean_log_τ_l + p[14]*mean_log_M_l + p[15]*mean_int_l
        end

        # Non-cognitive variables
        log_hyp_h, cpm_h, log_τ_h = get_noncog_vars(df_matrix, t, 1)
        log_hyp_l, cpm_l, log_τ_l = get_noncog_vars(df_matrix, t, 0)

        mean_log_hyp_h, mean_cpm_h, mean_log_τ_h = mean(log_hyp_h), mean(cpm_h), mean(log_τ_h)
        mean_log_hyp_l, mean_cpm_l, mean_log_τ_l = mean(log_hyp_l), mean(cpm_l), mean(log_τ_l)

        q = params2
        if t == 2
            avgncog_h[1, 1], avgncog_l[1, 1] = mean_log_hyp_h, mean_log_hyp_l
            avgncog_h[1, t] = q[12] + q[13]*avgncog_h[1, 1] + q[14]*mean_cpm_h + q[15]*mean_log_τ_h
            avgncog_l[1, t] = q[12] + q[13]*avgncog_l[1, 1] + q[14]*mean_cpm_l + q[15]*mean_log_τ_l
        elseif t == 3
            avgncog_h[1, t] = q[16] + q[17]*avgncog_h[1, t-1] + q[18]*mean_cpm_h + q[19]*mean_log_τ_h
            avgncog_l[1, t] = q[16] + q[17]*avgncog_l[1, t-1] + q[18]*mean_cpm_l + q[19]*mean_log_τ_l
        elseif t == 4
            avgncog_h[1, t] = q[20] + q[21]*avgncog_h[1, t-1] + q[22]*mean_cpm_h + q[23]*mean_log_τ_h
            avgncog_l[1, t] = q[20] + q[21]*avgncog_l[1, t-1] + q[22]*mean_cpm_l + q[23]*mean_log_τ_l
        end
    end

    return avgcog_h, avgncog_h, avgcog_l, avgncog_l
end

function GenAvgMoments_gap(params1, params2, df_matrix, Y)
    mom_b_h = zeros(1, 6)
    mom_cf1_h = zeros(1,6)
    mom_cf2_h = zeros(1,6)
    mom_cf3_h = zeros(1,6)
    mom_cf4_h = zeros(1,6)
    mom_b_l = zeros(1, 6)
    mom_cf1_l = zeros(1,6)
    mom_cf2_l = zeros(1,6)
    mom_cf3_l = zeros(1,6)
    mom_cf4_l = zeros(1,6)
    avgcog_b_h = zeros(1,4)
    avgcog_cf1_h = zeros(1,4)
    avgcog_cf2_h = zeros(1,4)
    avgcog_cf3_h = zeros(1,4)
    avgcog_cf4_h = zeros(1,4)
    avgncog_b_h = zeros(1,4)
    avgncog_cf1_h = zeros(1,4)
    avgncog_cf2_h = zeros(1,4)
    avgncog_cf3_h = zeros(1,4)
    avgncog_cf4_h = zeros(1,4)
    avgcog_b_l = zeros(1,4)
    avgcog_cf1_l = zeros(1,4)
    avgcog_cf2_l = zeros(1,4)
    avgcog_cf3_l = zeros(1,4)
    avgcog_cf4_l = zeros(1,4)
    avgncog_b_l = zeros(1,4)
    avgncog_cf1_l = zeros(1,4)
    avgncog_cf2_l = zeros(1,4)
    avgncog_cf3_l = zeros(1,4)
    avgncog_cf4_l = zeros(1,4)
    Vc_h = zeros(1, 5)
    Vp_h = zeros(1, 5)
    Vc_l = zeros(1, 5)
    Vp_l = zeros(1, 5)

    for q = 1:10
        data_b, data_cf1, data_cf2, data_cf3, data_cf4 = Simulate_all(params1, params2, n_c, n_age, s, Time, df_matrix, Y)
        
        mom_b_h = vcat(mom_b_h, reshape(cf_gap(data_b)[1, 1:6], 1, 6))
        mom_b_l = vcat(mom_b_l, reshape(cf_gap(data_b)[1, 7:12], 1, 6))
        avgcog_tmp_h, avgncog_tmp_h, avgcog_tmp_l, avgncog_tmp_l = simulate_gap(params1, params2, data_b)
        avgcog_b_h = vcat(avgcog_b_h, avgcog_tmp_h)
        avgncog_b_h = vcat(avgncog_b_h, avgncog_tmp_h)
        avgcog_b_l = vcat(avgcog_b_l, avgcog_tmp_l)
        avgncog_b_l = vcat(avgncog_b_l, avgncog_tmp_l)
    
        mom_cf1_h = vcat(mom_cf1_h, reshape(cf_gap(data_cf1)[1, 1:6], 1, 6))
        mom_cf1_l = vcat(mom_cf1_l, reshape(cf_gap(data_cf1)[1, 7:12], 1, 6))
        avgcog_tmp_h, avgncog_tmp_h, avgcog_tmp_l, avgncog_tmp_l = simulate_gap(params1, params2, data_cf1)
        avgcog_cf1_h = vcat(avgcog_cf1_h, avgcog_tmp_h)
        avgncog_cf1_h = vcat(avgncog_cf1_h, avgncog_tmp_h)
        avgcog_cf1_l = vcat(avgcog_cf1_l, avgcog_tmp_l)
        avgncog_cf1_l = vcat(avgncog_cf1_l, avgncog_tmp_l)
    
        mom_cf2_h = vcat(mom_cf2_h, reshape(cf_gap(data_cf2)[1, 1:6], 1, 6))
        mom_cf2_l = vcat(mom_cf2_l, reshape(cf_gap(data_cf2)[1, 7:12], 1, 6))
        avgcog_tmp_h, avgncog_tmp_h, avgcog_tmp_l, avgncog_tmp_l = simulate_gap(params1, params2, data_cf2)
        avgcog_cf2_h = vcat(avgcog_cf2_h, avgcog_tmp_h)
        avgncog_cf2_h = vcat(avgncog_cf2_h, avgncog_tmp_h)
        avgcog_cf2_l = vcat(avgcog_cf2_l, avgcog_tmp_l)
        avgncog_cf2_l = vcat(avgncog_cf2_l, avgncog_tmp_l)

        mom_cf3_h = vcat(mom_cf3_h, reshape(cf_gap(data_cf3)[1, 1:6], 1, 6))
        mom_cf3_l = vcat(mom_cf3_l, reshape(cf_gap(data_cf3)[1, 7:12], 1, 6))
        avgcog_tmp_h, avgncog_tmp_h, avgcog_tmp_l, avgncog_tmp_l = simulate_gap(params1, params2, data_cf3)
        avgcog_cf3_h = vcat(avgcog_cf3_h, avgcog_tmp_h)
        avgncog_cf3_h = vcat(avgncog_cf3_h, avgncog_tmp_h)
        avgcog_cf3_l = vcat(avgcog_cf3_l, avgcog_tmp_l)
        avgncog_cf3_l = vcat(avgncog_cf3_l, avgncog_tmp_l)
    
        mom_cf4_h = vcat(mom_cf4_h, reshape(cf_gap(data_cf4)[1, 1:6], 1, 6))
        mom_cf4_l = vcat(mom_cf4_l, reshape(cf_gap(data_cf4)[1, 7:12], 1, 6))
        avgcog_tmp_h, avgncog_tmp_h, avgcog_tmp_l, avgncog_tmp_l = simulate_gap(params1, params2, data_cf4)
        avgcog_cf4_h = vcat(avgcog_cf4_h, avgcog_tmp_h)
        avgncog_cf4_h = vcat(avgncog_cf4_h, avgncog_tmp_h)
        avgcog_cf4_l = vcat(avgcog_cf4_l, avgcog_tmp_l)
        avgncog_cf4_l = vcat(avgncog_cf4_l, avgncog_tmp_l)
    end    

    avgmom_b_h = mom_b_h[2:10+1, :]
    avgmom_b_h = mean(avgmom_b_h, dims=1)
    avgcog_b_h = avgcog_b_h[2:10+1, :]
    avgcog_b_h = mean(avgcog_b_h, dims=1)
    avgncog_b_h = avgncog_b_h[2:10+1, :]
    avgncog_b_h = mean(avgncog_b_h, dims=1)
    
    avgmom_cf1_h = mom_cf1_h[2:10+1, :]
    avgmom_cf1_h = mean(avgmom_cf1_h, dims=1)
    avgcog_cf1_h = avgcog_cf1_h[2:10+1, :]
    avgcog_cf1_h = mean(avgcog_cf1_h, dims=1)
    avgncog_cf1_h = avgncog_cf1_h[2:10+1, :]
    avgncog_cf1_h = mean(avgncog_cf1_h, dims=1)

    avgmom_cf2_h = mom_cf2_h[2:10+1, :]
    avgmom_cf2_h = mean(avgmom_cf2_h, dims=1)
    avgcog_cf2_h = avgcog_cf2_h[2:10+1, :]
    avgcog_cf2_h = mean(avgcog_cf2_h, dims=1)
    avgncog_cf2_h = avgncog_cf2_h[2:10+1, :]
    avgncog_cf2_h = mean(avgncog_cf2_h, dims=1)

    avgmom_cf3_h = mom_cf3_h[2:10+1, :]
    avgmom_cf3_h = mean(avgmom_cf3_h, dims=1)
    avgcog_cf3_h = avgcog_cf3_h[2:10+1, :]
    avgcog_cf3_h = mean(avgcog_cf3_h, dims=1)
    avgncog_cf3_h = avgncog_cf3_h[2:10+1, :]
    avgncog_cf3_h = mean(avgncog_cf3_h, dims=1)

    avgmom_cf4_h = mom_cf4_h[2:10+1, :]
    avgmom_cf4_h = mean(avgmom_cf4_h, dims=1)
    avgcog_cf4_h = avgcog_cf4_h[2:10+1, :]
    avgcog_cf4_h = mean(avgcog_cf4_h, dims=1)
    avgncog_cf4_h = avgncog_cf4_h[2:10+1, :]
    avgncog_cf4_h = mean(avgncog_cf4_h, dims=1)

    avgmom_b_l = mom_b_l[2:10+1, :]
    avgmom_b_l = mean(avgmom_b_l, dims=1)
    avgcog_b_l = avgcog_b_l[2:10+1, :]
    avgcog_b_l = mean(avgcog_b_l, dims=1)
    avgncog_b_l = avgncog_b_l[2:10+1, :]
    avgncog_b_l = mean(avgncog_b_l, dims=1)

    avgmom_cf1_l = mom_cf1_l[2:10+1, :]
    avgmom_cf1_l = mean(avgmom_cf1_l, dims=1)
    avgcog_cf1_l = avgcog_cf1_l[2:10+1, :]
    avgcog_cf1_l = mean(avgcog_cf1_l, dims=1)
    avgncog_cf1_l = avgncog_cf1_l[2:10+1, :]
    avgncog_cf1_l = mean(avgncog_cf1_l, dims=1)

    avgmom_cf2_l = mom_cf2_l[2:10+1, :]
    avgmom_cf2_l = mean(avgmom_cf2_l, dims=1)
    avgcog_cf2_l = avgcog_cf2_l[2:10+1, :]
    avgcog_cf2_l = mean(avgcog_cf2_l, dims=1)
    avgncog_cf2_l = avgncog_cf2_l[2:10+1, :]
    avgncog_cf2_l = mean(avgncog_cf2_l, dims=1)

    avgmom_cf3_l = mom_cf3_l[2:10+1, :]
    avgmom_cf3_l = mean(avgmom_cf3_l, dims=1)
    avgcog_cf3_l = avgcog_cf3_l[2:10+1, :]
    avgcog_cf3_l = mean(avgcog_cf3_l, dims=1)
    avgncog_cf3_l = avgncog_cf3_l[2:10+1, :]
    avgncog_cf3_l = mean(avgncog_cf3_l, dims=1)

    avgmom_cf4_l = mom_cf4_l[2:10+1, :]
    avgmom_cf4_l = mean(avgmom_cf4_l, dims=1)
    avgcog_cf4_l = avgcog_cf4_l[2:10+1, :]
    avgcog_cf4_l = mean(avgcog_cf4_l, dims=1)
    avgncog_cf4_l = avgncog_cf4_l[2:10+1, :]
    avgncog_cf4_l = mean(avgncog_cf4_l, dims=1)

    res = value_func(params1, params2, n_c, n_age, ncog, s, Time, Y)
    Vc_h[1] = mean(res[31][:, 3]) + res[23]*avgcog_b_h[3]+β_c*params1[12]*avgcog_b_h[3]*mean(res[41][:, 3])
    Vc_h[2] = mean(res[32][:, 3]) + res[23]*avgcog_cf1_h[3]+β_c*params1[12]*avgcog_cf1_h[3]*mean(res[41][:, 3])
    Vc_h[3] = mean(res[33][:, 3]) + res[23]*avgcog_cf2_h[3]+β_c*params1[12]*avgcog_cf2_h[3]*mean(res[41][:, 3])
    Vc_h[4] = mean(res[34][:, 3]) + res[23]*avgcog_cf3_h[3]+β_c*params1[12]*avgcog_cf3_h[3]*mean(res[41][:, 3])
    Vc_h[5] = mean(res[35][:, 3]) + res[23]*avgcog_cf4_h[3]+β_c*params1[12]*avgcog_cf4_h[3]*mean(res[41][:, 3])
    Vp_h[1] = mean(res[36][:, 3]) + ((1-φ)*res[17]+φ*res[23])*avgcog_b_h[3]+β_p*params1[12]*avgcog_b_h[3]*mean(res[43][:, 3])
    Vp_h[2] = mean(res[37][:, 3]) + ((1-φ)*res[17]+φ*res[23])*avgcog_cf1_h[3]+β_p*params1[12]*avgcog_cf1_h[3]*mean(res[43][:, 3])
    Vp_h[3] = mean(res[38][:, 3]) + ((1-φ)*res[17]+φ*res[23])*avgcog_cf2_h[3]+β_p*params1[12]*avgcog_cf2_h[3]*mean(res[43][:, 3])
    Vp_h[4] = mean(res[39][:, 3]) + ((1-φ)*res[17]+φ*res[23])*avgcog_cf3_h[3]+β_p*params1[12]*avgcog_cf3_h[3]*mean(res[43][:, 3])
    Vp_h[5] = mean(res[40][:, 3]) + ((1-φ)*res[17]+φ*res[23])*avgcog_cf4_h[3]+β_p*params1[12]*avgcog_cf4_h[3]*mean(res[43][:, 3])
    
    Vc_l[1] = mean(res[31][:, 3]) + res[23]*avgcog_b_l[3]+β_c*params1[12]*avgcog_b_l[3]*mean(res[41][:, 3])
    Vc_l[2] = mean(res[32][:, 3]) + res[23]*avgcog_cf1_l[3]+β_c*params1[12]*avgcog_cf1_l[3]*mean(res[41][:, 3])
    Vc_l[3] = mean(res[33][:, 3]) + res[23]*avgcog_cf2_l[3]+β_c*params1[12]*avgcog_cf2_l[3]*mean(res[41][:, 3])
    Vc_l[4] = mean(res[34][:, 3]) + res[23]*avgcog_cf3_l[3]+β_c*params1[12]*avgcog_cf3_l[3]*mean(res[41][:, 3])
    Vc_l[5] = mean(res[35][:, 3]) + res[23]*avgcog_cf4_l[3]+β_c*params1[12]*avgcog_cf4_l[3]*mean(res[41][:, 3])
    Vp_l[1] = mean(res[36][:, 3]) + ((1-φ)*res[17]+φ*res[23])*avgcog_b_l[3]+β_p*params1[12]*avgcog_b_l[3]*mean(res[43][:, 3])
    Vp_l[2] = mean(res[37][:, 3]) + ((1-φ)*res[17]+φ*res[23])*avgcog_cf1_l[3]+β_p*params1[12]*avgcog_cf1_l[3]*mean(res[43][:, 3])
    Vp_l[3] = mean(res[38][:, 3]) + ((1-φ)*res[17]+φ*res[23])*avgcog_cf2_l[3]+β_p*params1[12]*avgcog_cf2_l[3]*mean(res[43][:, 3])
    Vp_l[4] = mean(res[39][:, 3]) + ((1-φ)*res[17]+φ*res[23])*avgcog_cf3_l[3]+β_p*params1[12]*avgcog_cf3_l[3]*mean(res[43][:, 3])
    Vp_l[5] = mean(res[40][:, 3]) + ((1-φ)*res[17]+φ*res[23])*avgcog_cf4_l[3]+β_p*params1[12]*avgcog_cf4_l[3]*mean(res[43][:, 3])

    return avgmom_b_h, avgmom_cf1_h, avgmom_cf2_h, avgmom_cf3_h, avgmom_cf4_h,
    avgmom_b_l, avgmom_cf1_l, avgmom_cf2_l, avgmom_cf3_l, avgmom_cf4_l,
     avgcog_b_h, avgcog_cf1_h, avgcog_cf2_h, avgcog_cf3_h, avgcog_cf4_h,
     avgncog_b_h, avgncog_cf1_h, avgncog_cf2_h, avgncog_cf3_h, avgncog_cf4_h,
      avgcog_b_l, avgcog_cf1_l, avgcog_cf2_l, avgcog_cf3_l, avgcog_cf4_l,
       avgncog_b_l, avgncog_cf1_l, avgncog_cf2_l, avgncog_cf3_l, avgncog_cf4_l, 
       Vc_h, Vc_l, Vp_h, Vp_l
end

# (3) Cash Transfers

function cct(n_c, n_age, Y, data)
    Y_new = zeros(n_c * n_age)

    for i in 1:n_c
        for t in 1:n_age
            # Identify the indices where data[:, 1] == i and data[:, 2] == t
            idx = (data[:, 1] .== i) .& (data[:, 2] .== t)

            if any(idx .& (data[:, 11] .== 0))  # If there's a match for value 0 in data[:, 11]
                Y_new[(i-1)*n_age + t] = Y[(i-1)*n_age + t] + 800
            elseif any(idx .& (data[:, 11] .== 1))  # If there's a match for value 1 in data[:, 11]
                Y_new[(i-1)*n_age + t] = Y[(i-1)*n_age + t]
            elseif any(idx .& (data[:, 11] .== -999))  # If there's a match for value -999 in data[:, 11]
                Y_new[(i-1)*n_age + t] = Y[(i-1)*n_age + t]
            end
        end
    end

    return Y_new
end
