
# ==============================================================================
# Project Title: The Dynamics of Parent-Child Interactions in Shaping Cognitive and
# Non-cognitive Development during Adolescence
# Author: Claire Kim
# Institution: University of Wisconsin-Madison
# Start Date: 02/27/2025
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

function Simulate_all(params1, params2, n_c, n_age, s, Time, df_matrix, Y)
    results = value_func(params1, params2, n_c, n_age, hyp, s, Time, Y)

    # Unpack results
    M_opt, CPM_opt, τ_opt, R_opt = results[3], results[1], results[2], results[4]
    CPM_cf1, τ_cf1, R_cf1 = results[5], results[9], results[13]
    CPM_cf2, τ_cf2, R_cf2 = results[6], results[10], results[14]
    CPM_cf3, τ_cf3, R_cf3 = results[7], results[11], results[15]
    CPM_cf4, τ_cf4, R_cf4 = results[8], results[12], results[16]   

    function build_dataset(CPM, τ, R, M)
        simdata = Matrix{Union{Float64, Int64}}(undef, 1, 14)
        simdata[:, 1] .= 1 # Child's ID
        simdata[:, 2] .= 1 # Child's Age Period: t = 1, 2, 3
        simdata[:, 3] .= 1 # Mother's Age
        simdata[:, 4] .= 1 # Child's Hyperactivity
        simdata[:, 5] .= 1 # Child's Actual Age
        simdata[:, 6] .= 1 # Female 
        simdata[:, 7] .= 1 # CPM choice
        simdata[:, 11] .= 1 # College Mother

        simdata[:, 8] .= 1.0 # Household income
        simdata[:, 9] .= 1.0 # School hour
        simdata[:, 10] .= 1.0 # Study time
        simdata[:, 12] .= 1.0 # Weekly pocket money
        simdata[:, 13] .= 1.0 # Weekly Educatioanl Investment
        simdata[:, 14] .= 1.0 # Error-free hyperactivity

        for i = 1:n_c
            iData = Matrix{Union{Float64, Int64}}(undef, n_age, 14)
            iData[:, 1] .= 1 # Child's ID
            iData[:, 2] .= 1 # Child's Age Period
            iData[:, 3] .= 1 # Mother's Age
            iData[:, 4] .= 1 # Child's Hyperactivity
            iData[:, 5] .= 1 # Child's Actual Age
            iData[:, 6] .= 1 # Female 
            iData[:, 7] .= 1 # CPM choice
            iData[:, 11] .= 1 # College Mother

            iData[:, 8] .= 1.0 # Household income
            iData[:, 9] .= 1.0 # School hour
            iData[:, 10] .= 1.0 # Study time
            iData[:, 12] .= 1.0 # Weekly pocket money
            iData[:, 13] .= 1.0 # Weekly 
            iData[:, 14] .= 1.0 # Error-free hyperactivity

            iData[:, 1] .= i
            iData[:, 3] = convert(Array{Int64,1}, df_matrix[df_matrix[:, 1] .== i, 3])
            #iData[:, 4] .= 1
            iData[:, 5] = convert(Array{Int64,1}, df_matrix[df_matrix[:, 1] .== i, 5])
            iData[:, 6] = convert(Array{Int64,1}, df_matrix[df_matrix[:, 1] .== i, 6])
            #iData[:, 7] .= 1
            iData[:, 8] = convert(Array{Float64,1}, df_matrix[df_matrix[:, 1] .== i, 8])
            iData[:, 9] = convert(Array{Float64,1}, df_matrix[df_matrix[:, 1] .== i, 9])
            #iData[:, 10] .= 1.0
            iData[:, 11] = convert(Array{Int64,1}, df_matrix[df_matrix[:, 1] .== i, 11])
            #iData[:, 12] .= 1.0
            #iData[:, 13] .= 1.0
            #iData[:, 14] .= 1.0

            iData[1, 4] = df_matrix[(df_matrix[:, 1] .== i) .& (df_matrix[:, 2] .== 1), 4][1]
            iData[1, 14] = df_matrix[(df_matrix[:, 1] .== i) .& (df_matrix[:, 2] .== 1), 14][1]

            # Choices per age
            for t in 1:n_age
                iData[t, 2] = t
                iData[t, 7] = CPM[i, t]
                iData[t, 10] = τ[i, t]
                iData[t, 12] = R[i, t]
                iData[t, 13] = M[i, t]
            end

            # Simulate error-free hyperactivity (based on my model)
            iData[2, 14] = exp(params2[12] + params2[13]*log(iData[1, 14]) + params2[14]*iData[1, 7] + params2[15]*log(iData[1, 10]))
            iData[3, 14] = exp(params2[16] + params2[17]*log(iData[2, 14]) + params2[18]*iData[2, 7] + params2[19]*log(iData[2, 10]))

            simdata = vcat(simdata, iData)
        end

        simdata = simdata[2:(n_c*n_age+1),:]
        return DataFrame(simdata, ["cid", "age", "mage", "hyp", "cage", "female", "CPM", "Y", "s", "τ", "medu", "pm", "M", "ncog"])
    end

    df_opt = build_dataset(CPM_opt, τ_opt, R_opt, M_opt)
    df_cf1 = build_dataset(CPM_cf1, τ_cf1, R_cf1, M_opt)
    df_cf2 = build_dataset(CPM_cf2, τ_cf2, R_cf2, M_opt)
    df_cf3 = build_dataset(CPM_cf3, τ_cf3, R_cf3, M_opt)
    df_cf4 = build_dataset(CPM_cf4, τ_cf4, R_cf4, M_opt)

    return (opt = df_opt, cf1 = df_cf1, cf2 = df_cf2, cf3 = df_cf3, cf4 = df_cf4)
end

# Generate data moments
function GenMoments(df_matrix)
    #println("Generating moments...")

    #(1) Production Parameters for non-cognitive skills
    mom1 = zeros(2, 1) # The marginal products of CPM on change in skills, δ_{2, t}^N

        # Corelations between the change in cognitive skill with lagged CPM decisions
            # Step (1) Filtering the valid rows for each time period
            t1_valid = df_matrix[(df_matrix[:, 2] .== 1) .& (df_matrix[:, 7] .!= -999), :]
            t2_valid = df_matrix[(df_matrix[:, 2] .== 2) .& (df_matrix[:, 7] .!= -999), :]
            t3_valid = df_matrix[(df_matrix[:, 2] .== 3) .& (df_matrix[:, 7] .!= -999), :]

            # Extracting the IDs and household income for each period (assuming it's in column 9)
            t1_ids = t1_valid[:, 1]  # IDs for t=1
            t2_ids = t2_valid[:, 1]  # IDs for t=2
            t3_ids = t3_valid[:, 1]  # IDs for t=3

            # Step (2) Finding common IDs across the periods
            common_ids1 = Set(t1_ids) ∩ Set(t2_ids) # Observations available for t = 1 and t = 2
            common_ids2 = Set(t2_ids) ∩ Set(t3_ids) # Observations available for t = 2 and t = 3

            # Step (3) Creating matched data based on common IDs
            t1_common1 = t1_valid[findall(x -> x in common_ids1, t1_ids), :]
            t2_common1 = t2_valid[findall(x -> x in common_ids1, t2_ids), :]

            t2_common2 = t2_valid[findall(x -> x in common_ids2, t2_ids), :]
            t3_common2 = t3_valid[findall(x -> x in common_ids2, t3_ids), :]

            # Step (4) Calculate the change in test scores (delta)
            change_in_scores_1_2 = t2_common1[:, 14] .- t1_common1[:, 14]  # Change from t1 to t2
            change_in_scores_2_3 = t3_common2[:, 14] .- t2_common2[:, 14]  # Change from t2 to t3

            # Step (6) Store results in mom3 array (if needed for further analysis)
            mom1[1, 1] = cor(change_in_scores_1_2, t1_common1[:, 7]) # corelation between Δscore (t1 to t2) and CPM at t1
            mom1[2, 1] = cor(change_in_scores_2_3, t2_common2[:, 7]) # corelation between Δscore (t2 to t3) and CPM at t2

    mom2 = zeros(2, 1) # The marginal products of lagged skills on change in skills, δ_{1, t}^N

        # Step (1) Filtering the valid rows for each time period
        t1_valid = df_matrix[(df_matrix[:, 2] .== 1), :]
        t2_valid = df_matrix[(df_matrix[:, 2] .== 2), :]
        t3_valid = df_matrix[(df_matrix[:, 2] .== 3), :]

        # Extracting the IDs and household income for each period (assuming it's in column 9)
        t1_ids = t1_valid[:, 1]  # IDs for t=1
        t2_ids = t2_valid[:, 1]  # IDs for t=2
        t3_ids = t3_valid[:, 1]  # IDs for t=3

        # Step (2) Finding common IDs across the periods
        common_ids1 = Set(t1_ids) ∩ Set(t2_ids) # Observations available for t = 1 and t = 2
        common_ids2 = Set(t2_ids) ∩ Set(t3_ids) # Observations available for t = 2 and t = 3

        # Step (3) Creating matched data based on common IDs
        t1_common1 = t1_valid[findall(x -> x in common_ids1, t1_ids), :]
        t2_common1 = t2_valid[findall(x -> x in common_ids1, t2_ids), :]

        t2_common2 = t2_valid[findall(x -> x in common_ids2, t2_ids), :]
        t3_common2 = t3_valid[findall(x -> x in common_ids2, t3_ids), :]

        # Step (4) Calculate the change in test scores (delta)
        change_in_scores_1_2 = t2_common1[:, 14] .- t1_common1[:, 14]  # Change from t1 to t2
        change_in_scores_2_3 = t3_common2[:, 14] .- t2_common2[:, 14]  # Change from t2 to t3

        # Step (6) Store results in mom3 array (if needed for further analysis)
        mom2[1, 1] = cor(change_in_scores_1_2, t1_common1[:, 14]) # corelation between Δscore (t1 to t2) and CPM at t1
        mom2[2, 1] = cor(change_in_scores_2_3, t2_common2[:, 14]) # corelation between Δscore (t2 to t3) and CPM at t2

    mom3 = zeros(3, 1) # The total factor productivity parameters, log(A_t) 
    mom3a = zeros(3, 1) # The total factor productivity parameters, log(A_t) 
        
        # Averages and standard deviation of child non-cognitive skills at each age
        for t = 1:n_age
            mom3[t, 1] = mean(df_matrix[(df_matrix[:, 2] .== t), 14])
            mom3a[t, 1] = std(df_matrix[(df_matrix[:, 2] .== t), 14])
        end

    # (3) Preference Parameters of Child
    mom4 = zeros(3, 1) # Variation in child's self-investment time conditional on CPM = 0, λ_1
    mom4a = zeros(3, 1) # Variation in child's self-investment time conditional on CPM = 0, λ_1

    for t = 1:n_age
        mom4[t, 1] = mean(df_matrix[(df_matrix[:, 2] .== t) .& (df_matrix[:, 10] .!= -999) .& (df_matrix[:, 7] .== 0), 10])
        mom4a[t, 1] = std(df_matrix[(df_matrix[:, 2] .== t) .& (df_matrix[:, 10] .!= -999) .& (df_matrix[:, 7] .== 0), 10])
    end

    mom5 = zeros(3, 1) # Within-child variation in self-investment time if CPM = 1
    mom5a = zeros(3, 1) # Within-child variation in self-investment time if CPM = 1

    for t = 1:n_age
        mom5[t, 1] = mean(df_matrix[(df_matrix[:, 2] .== t) .& (df_matrix[:, 10] .!= -999) .& (df_matrix[:, 7] .== 1), 10])
        mom5a[t, 1] = std(df_matrix[(df_matrix[:, 2] .== t) .& (df_matrix[:, 10] .!= -999) .& (df_matrix[:, 7] .== 1), 10])
    end
    
    # (4) Preference Parameters of Parents + λ_2
    mom7 = zeros(9, 1) # Use Average of R_T and R_t / Y_t by CPM choice
    mom7a = zeros(9, 1) # Use standard deviations of R_t and R_t / Y_t by CPM choice

    for t = 1:n_age
        mom7[t, 1] = mean(df_matrix[(df_matrix[:, 2] .== t) .& (df_matrix[:, 12] .!= -999), 12])
        mom7[t+3, 1] = mean(df_matrix[(df_matrix[:, 2] .== t) .& (df_matrix[:, 12] .!= -999) .& (df_matrix[:, 7] .== 1), 12])
        mom7[t+6, 1] = mean(df_matrix[(df_matrix[:, 2] .== t) .& (df_matrix[:, 12] .!= -999) .& (df_matrix[:, 7] .== 0), 12])
        mom7a[t, 1] = std(df_matrix[(df_matrix[:, 2] .== t) .& (df_matrix[:, 12] .!= -999), 12])
        mom7a[t+3, 1] = std(df_matrix[(df_matrix[:, 2] .== t) .& (df_matrix[:, 12] .!= -999) .& (df_matrix[:, 7] .== 1), 12])
        mom7a[t+6, 1] = std(df_matrix[(df_matrix[:, 2] .== t) .& (df_matrix[:, 12] .!= -999) .& (df_matrix[:, 7] .== 0), 12])
    end

    mom11 = zeros(9, 1)
    mom11a = zeros(9, 1)

    for t = 1:n_age
        mom11[t, 1] = mean((df_matrix[(df_matrix[:, 2] .== t) .& (df_matrix[:, 12] .!= -999), 12]) /
         (df_matrix[(df_matrix[:, 2] .== t) .& (df_matrix[:, 12] .!= -999), 8]))
        mom11[t+3, 1] = mean((df_matrix[(df_matrix[:, 2] .== t) .& (df_matrix[:, 12] .!= -999) .& (df_matrix[:, 7] .== 1), 12]) /
         (df_matrix[(df_matrix[:, 2] .== t) .& (df_matrix[:, 12] .!= -999) .& (df_matrix[:, 7] .== 1), 8]))
        mom11[t+6, 1] = mean((df_matrix[(df_matrix[:, 2] .== t) .& (df_matrix[:, 12] .!= -999) .& (df_matrix[:, 7] .== 0), 12]) /
         (df_matrix[(df_matrix[:, 2] .== t) .& (df_matrix[:, 12] .!= -999) .& (df_matrix[:, 7] .== 0), 8]))
        mom11a[t, 1] = std((df_matrix[(df_matrix[:, 2] .== t) .& (df_matrix[:, 12] .!= -999), 12]) /
         (df_matrix[(df_matrix[:, 2] .== t) .& (df_matrix[:, 12] .!= -999), 8]))
        mom11a[t+3, 1] = std((df_matrix[(df_matrix[:, 2] .== t) .& (df_matrix[:, 12] .!= -999) .& (df_matrix[:, 7] .== 1), 12]) /
         (df_matrix[(df_matrix[:, 2] .== t) .& (df_matrix[:, 12] .!= -999) .& (df_matrix[:, 7] .== 1), 8]))
        mom11a[t+6, 1] = std((df_matrix[(df_matrix[:, 2] .== t) .& (df_matrix[:, 12] .!= -999) .& (df_matrix[:, 7] .== 0), 12]) /
         (df_matrix[(df_matrix[:, 2] .== t) .& (df_matrix[:, 12] .!= -999) .& (df_matrix[:, 7] .== 0), 8]))
    end

    mom8 = zeros(3, 1) # CPM cost distribution
        # By Child's Age
        for j = 1:3
            # CPM Choice Frequency
            mom8[j, 1] = mean(df_matrix[(df_matrix[:, 2] .== j) .& (df_matrix[:, 7] .!= -999), 7]) # E(CPM_t | t)
        end
        
    mom9 = zeros(4, 1)    # Correlations with child's age and household characteristics
        
        mom9[1, 1] = cor(df_matrix[(df_matrix[:, 7] .!= -999) .& (df_matrix[:, 5] .!= -999), 7],
        df_matrix[(df_matrix[:, 7] .!= -999) .& (df_matrix[:, 5] .!= -999), 5]) # Child's age
        mom9[2, 1] = cor(df_matrix[(df_matrix[:, 7] .!= -999) .& (df_matrix[:, 11] .!= -999), 7],
        df_matrix[(df_matrix[:, 7] .!= -999) .& (df_matrix[:, 11] .!= -999), 11]) # Mother's education
        mom9[3, 1] = cor(df_matrix[(df_matrix[:, 7] .!= -999) .& (df_matrix[:, 10] .!= -999), 7],
        df_matrix[(df_matrix[:, 7] .!= -999) .& (df_matrix[:, 10] .!= -999), 10]) # Self-investment
        mom9[4, 1] = cor(df_matrix[(df_matrix[:, 7] .!= -999), 7],
        df_matrix[(df_matrix[:, 7] .!= -999), 14]) # Non-cognitive skills

    mom10 = zeros(2, 1) # The marginal products of τ on future non-cognitive skills, δ_{2, t}^N

        # Corelations between the change in cognitive skill with lagged CPM decisions
            # Step (1) Filtering the valid rows for each time period
            t1_valid = df_matrix[(df_matrix[:, 2] .== 1) .& (df_matrix[:, 10] .!= -999), :]
            t2_valid = df_matrix[(df_matrix[:, 2] .== 2) .& (df_matrix[:, 10] .!= -999), :]
            t3_valid = df_matrix[(df_matrix[:, 2] .== 3) .& (df_matrix[:, 10] .!= -999), :]

            # Extracting the IDs and household income for each period (assuming it's in column 9)
            t1_ids = t1_valid[:, 1]  # IDs for t=1
            t2_ids = t2_valid[:, 1]  # IDs for t=2
            t3_ids = t3_valid[:, 1]  # IDs for t=3

            # Step (2) Finding common IDs across the periods
            common_ids1 = Set(t1_ids) ∩ Set(t2_ids) # Observations available for t = 1 and t = 2
            common_ids2 = Set(t2_ids) ∩ Set(t3_ids) # Observations available for t = 2 and t = 3

            # Step (3) Creating matched data based on common IDs
            t1_common1 = t1_valid[findall(x -> x in common_ids1, t1_ids), :]
            t2_common1 = t2_valid[findall(x -> x in common_ids1, t2_ids), :]

            t2_common2 = t2_valid[findall(x -> x in common_ids2, t2_ids), :]
            t3_common2 = t3_valid[findall(x -> x in common_ids2, t3_ids), :]

            # Step (4) Calculate the change in test scores (delta)
            change_in_scores_1_2 = t2_common1[:, 14] .- t1_common1[:, 14]  # Change from t1 to t2
            change_in_scores_2_3 = t3_common2[:, 14] .- t2_common2[:, 14]  # Change from t2 to t3

            # Step (6) Store results in mom3 array (if needed for further analysis)
            mom10[1, 1] = cor(change_in_scores_1_2, t1_common1[:, 10]) # corelation between Δscore (t1 to t2) and CPM at t1
            mom10[2, 1] = cor(change_in_scores_2_3, t2_common2[:, 10]) # corelation between Δscore (t2 to t3) and CPM at t2

    #moment = hcat(mom1', mom2', mom3', mom4', mom5', mom6')
    moment = hcat(mom1', mom2', mom3', mom3a', mom4', mom4a', mom5', mom5a', mom7', mom7a', mom8', mom9', mom10', mom11', mom11a')
    #println("Moments generated.")
    return moment
end

# Take average of simulated data moments
function GenAvgMoments(params1, params2, df_matrix)
    # Pre-allocate the moms matrix for 10 simulations, each with 67 moments
    moms = zeros(10, 67)

    for q = 1:10
        # Simulate data
        data = Simulate_all(params1, params2, n_c, n_age, s, Time, df_matrix, Y).opt
        #data = Simulate(params1, params2, n_c, n_age, s, Time, df_matrix)
        data = Matrix(data)

        # Calculate moments for this simulation
        moments = GenMoments(data)

        # Store the moments in the corresponding row of the moms matrix
        moms[q, :] = moments
    end

    # Calculate the average of the moments across all 10 simulations
    avgmom = mean(moms, dims=1)

    return avgmom
end

function block_cov_weight(sim_mom, block_ranges)
    n = length(sim_mom)
    W = zeros(n, n)

    for block in block_ranges
        block_data = reshape(sim_mom[block], :, 1)
        block_len = length(block)

        # Compute covariance only if more than 1 moment
        if block_len == 1
            Σ = Matrix{Float64}(I(1)) * 1e-6
        else
            Σ = cov(block_data)

            # Make sure Σ is the right size, and regularize safely
            if size(Σ, 1) != block_len || size(Σ, 2) != block_len || any(isnan, Σ) || rank(Σ) < block_len
                Σ = Matrix{Float64}(I(block_len)) * 1e-6
            end
        end

        # Now this should always work
        W[block, block] .= (Σ)
    end

    return W
end

function Calibrate_w(params1, params2, df_matrix)
    # Simulate data using current parameters
    simdata = Simulate_all(params1, params2, n_c, n_age, s, Time, df_matrix, Y)[1]

    # Generate moments
    true_mom = GenMoments(df_matrix)
    sim_mom = GenMoments(simdata)

    # Define blocks (as in your original logic)
    block_ranges = [
        1:2, 3:4, 5:7, 8:10, 11:13, 14:16, 17:19, 20:22,
        23:31, 32:40, 41:43, 44:47, 48:49, 50:58, 59:67
    ]

    # Compute the optimal weighting matrix
    W_optimal = block_cov_weight(sim_mom, block_ranges)

    # Calculate moment distance
    diff = sim_mom .- true_mom
    err = sum(W_optimal .* (diff .^ 2))

    println("Updated moment difference (objective function): ", err)
    return err
end

function ReMatrix(B)
    blocks = [
        1:2, 3:4, 5:7, 8:10, 11:13, 14:16, 17:19, 20:22,
        23:31, 32:40, 41:43, 44:47, 48:49, 50:58, 59:67
    ]

    return [B[1, r] for r in blocks]
end

# Generate data moments with estimated parameters and simulated data
function GenMoments_est(df_matrix)
    #println("Generating moments...")

    #(1) Child Time Inputs by Child Age
    mom1 = zeros(3, 1)
    for j = 1:n_age
        mom1[j, 1] = mean(df_matrix[(df_matrix[:, 2] .== j) .& (df_matrix[:, 10] .!= -999), 10])
    end

    #(2) Child Self-Investment Time By CPM Choice
    mom2 = zeros(6, 1)

    for j = 1:n_age
        mom2[j, 1] = mean(df_matrix[(df_matrix[:, 2] .== j) .& (df_matrix[:, 7] .== 1) .& (df_matrix[:, 10] .!= -999), 10])
        mom2[j+3, 1] = mean(df_matrix[(df_matrix[:, 2] .== j) .& (df_matrix[:, 7] .== 0) .& (df_matrix[:, 10] .!= -999), 10])
    end

    #(3) CPM Choice
    mom3 = zeros(4, 1)

    for j = 1:n_age
        mom3[j, 1] = mean(df_matrix[(df_matrix[:, 2] .== j) .& (df_matrix[:, 7] .!= -999), 7])
    end
    mom3[4, 1] = mean(df_matrix[(df_matrix[:, 7] .!= -999), 7])

    #(4) Sustained attention
    mom4 = zeros(3, 1)
    for j = 1:n_age
        mom4[j, 1] = mean(df_matrix[(df_matrix[:, 2] .== j) .& (df_matrix[:, 14] .!= -999), 14])
    end

    moment = hcat(mom1', mom2', mom3', mom4')
    #println("Moments generated.")
    return moment
end

# Take average of simulated data moments
function GenAvgMoments_est(params1, params2, df_matrix)
    moms = zeros(1,16)

    for q = 1:10
        data = Simulate_all(params1, params2, n_c, n_age, s, Time, df_matrix, Y)[1]
        moms = vcat(moms, GenMoments_est(data))
    end

    moms = moms[2:10+1, :]
    avgmom = mean(moms, dims=1)

    return avgmom
end

# Construct one-row vector into multiple matrices
function ReMatrix_est(B)
    b1 = B[1,1:3]
    b2 = B[1,4:9]
    b3 = B[1,10:13]
    b4 = B[1,14:16]
    #b5 = B[1,73:92]
    #b6 = B[1,93:101]

    #A = [b1, b2, b3, b4, b5, b6]
    A = [b1, b2, b3, b4]

    return A
end
