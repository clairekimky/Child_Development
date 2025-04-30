
# ==============================================================================
# Project Title: The Dynamics of Parent-Child Interactions in Shaping Cognitive and
# Non-cognitive Development during Adolescence
# Author: Claire Kim
# Institution: University of Wisconsin-Madison
# Start Date: 03/11/2025
# Description: Compute Code to identify Latent Skill Distribution
#
# This script estimates a dynamic model of child development with a focus on
# parental incentive and self-investment on cognitive and non-cognitive traits
# using Simulated Method of Moments
# ==============================================================================

using Random
using Distributions
using Plots

# Factor Analysis

function factor(df_matrix, n_age)
    # Initialize μ_0 and μ_1 as zero matrices
    μ_0 = zeros(3, n_age)
    μ_1 = zeros(3, n_age)

    # Convert the input matrix into a standard matrix format if it's not already
    df_matrix = Matrix(df_matrix)  # Convert df_matrix to a proper 2D Matrix if it's not already
    #println("size(df_matrix): ", size(df_matrix))

    # Estimate μ_1[j, 1] or Period 1 factor loadings
    x1_raw = df_matrix[(df_matrix[:, 2] .== 1), 8] # read1
    x2_raw = df_matrix[(df_matrix[:, 2] .== 1), 9] # num1
    x3_raw = df_matrix[(df_matrix[:, 2] .== 1), 10] # write1
    # Apply log transformation
    x1_valid = log.(x1_raw)
    x2_valid = log.(x2_raw)
    x3_valid = log.(x3_raw)
    μ_1[1, 1] = sqrt(cov(x1_valid, x2_valid)*cov(x1_valid, x2_valid) / cov(x2_valid, x3_valid))
    μ_1[2, 1] = sqrt(cov(x2_valid, x1_valid)*cov(x2_valid, x3_valid) / cov(x1_valid, x3_valid))
    μ_1[3, 1] = sqrt(cov(x3_valid, x1_valid)*cov(x3_valid, x2_valid) / cov(x1_valid, x2_valid))

    # Age-invariance
    μ_1[1, 2] = μ_1[1, 1]
    μ_1[1, 3] = μ_1[1, 1]

    # Estimate μ_1[j, t] for j = 2, 3 and  t = 2, 3
    x4_raw = df_matrix[(df_matrix[:, 2] .== 2), 8] # read2
    x5_raw = df_matrix[(df_matrix[:, 2] .== 2), 9] # num2
    x6_raw = df_matrix[(df_matrix[:, 2] .== 2), 10] # write2
    x7_raw = df_matrix[(df_matrix[:, 2] .== 3), 8] # read3
    x8_raw = df_matrix[(df_matrix[:, 2] .== 3), 9] # num3
    x9_raw = df_matrix[(df_matrix[:, 2] .== 3), 10] # write4
    # Apply log transformation
    x4_valid = log.(x4_raw)
    x5_valid = log.(x5_raw)
    x6_valid = log.(x6_raw)
    x7_valid = log.(x7_raw)
    x8_valid = log.(x8_raw)
    x9_valid = log.(x9_raw)

    μ_1[2, 2] = μ_1[1, 2] * cov(x5_valid, x6_valid) / cov(x4_valid, x6_valid)
    μ_1[2, 3] = μ_1[1, 3] * cov(x8_valid, x9_valid) / cov(x7_valid, x9_valid)
    μ_1[3, 2] = μ_1[1, 2] * cov(x6_valid, x5_valid) / cov(x4_valid, x5_valid)
    μ_1[3, 3] = μ_1[1, 3] * cov(x9_valid, x8_valid) / cov(x7_valid, x8_valid)

    # Estimate μ_0[j, 1]
    for j = 1:3
        # Apply log transformation
        x1_valid = log.(df_matrix[(df_matrix[:, 2] .== 1), j+7])
    
        # Compute mean of the valid log-transformed values
        μ_0[j, 1] = mean(x1_valid)
    end    

    # Age-invariance
    μ_0[1, 2] = μ_0[1, 1]
    μ_0[1, 3] = μ_0[1, 1]
    
    avgcog2 = (mean(log.(df_matrix[(df_matrix[:, 2] .== 2), 8])) - μ_0[1, 1]) / μ_1[1, 2]
    avgcog3 = (mean(log.(df_matrix[(df_matrix[:, 2] .== 3), 8])) - μ_0[1, 1]) / μ_1[1, 3]

    μ_0[2, 2] = mean(log.(df_matrix[(df_matrix[:, 2] .== 2), 9])) - (μ_1[2, 2] * avgcog2)
    μ_0[2, 3] = mean(log.(df_matrix[(df_matrix[:, 2] .== 3), 9])) - (μ_1[2, 3] * avgcog3)
    μ_0[3, 2] = mean(log.(df_matrix[(df_matrix[:, 2] .== 2), 10])) - (μ_1[3, 2] * avgcog2)
    μ_0[3, 3] = mean(log.(df_matrix[(df_matrix[:, 2] .== 3), 10])) - (μ_1[3, 3] * avgcog3)

    return μ_0, μ_1

end

# Signal and Noise
function signal_and_noise(df_matrix, n_age)

    μ_0, μ_1 = factor(df_matrix, n_age)

    # Initialize signal grid
    logvar = zeros(n_age)
    signal = zeros(3, n_age)
    noise = zeros(3, n_age)

    for t = 1:n_age
        for j = 1:3
            x1_valid = log.(df_matrix[(df_matrix[:, 2] .== t), 8])
            x2_valid = log.(df_matrix[(df_matrix[:, 2] .== t), 9])
            logvar[t] = cov(x1_valid, x2_valid) / (μ_1[1, t]*μ_1[2, t])
            signal[j, t] = ((μ_1[j, t])^2 * logvar[t]) / 
            var(log.(df_matrix[(df_matrix[:, 2] .== t), j+7][.!(df_matrix[(df_matrix[:, 2] .== t), j+7] .== -999)]))
            noise[j, t] = (var(log.(df_matrix[(df_matrix[:, 2] .== t), j+7][.!(df_matrix[(df_matrix[:, 2] .== t), j+7] .== -999)])) -
            (μ_1[j, t])^2 * logvar[t]) / 
            var(log.(df_matrix[(df_matrix[:, 2] .== t), j+7][.!(df_matrix[(df_matrix[:, 2] .== t), j+7] .== -999)]))
        end
    end

    return signal, noise

end

function mean_theta(df_matrix, n_age)

    # Cognitive skill is a function that returns the location (μ_0) and factor loadings (μ_1)
    μ_0, μ_1 = factor(df_matrix, n_age)

    # Initialize vector to store mean cognitive skills for each age (time period)
    avgcog = zeros(n_age)

    # Loop over each time period (age)
    for t = 1:n_age
        num = 0.0   # Accumulator for the numerator
        den = 0.0   # Accumulator for the denominator

        # Loop over each measurement
        for j = 1:3
            # Extract relevant data for period t and measurement j
            valid = log.(df_matrix[(df_matrix[:, 2] .== t), j+7][df_matrix[(df_matrix[:, 2] .== t), j+7] .!= -999])
            
            # Accumulate the numerator (log(theta_{jt}^*) - μ_0)
            num += sum(valid .- μ_0[j, t])
            
            # Accumulate the denominator (μ_1)
            den += μ_1[j, t] * length(valid)  # Multiply by number of valid entries for j

        end
        #println("num: ", num)
        #println("den: ", den)
        # Calculate the mean for period t
        avgcog[t] = num / den
    end

    return avgcog
end

# Recover latent non-cognitive skills from observed SDQ scores
# Assumes fixed logistic mapping parameters: μ_0^N = -2.94, μ_1^N = 1

function recover_latent_skills_from_sdq(n_c, n_age, data; mu_0 = -2.94, mu_1 = 1.0, p_min = 0.05)
    # Initialize matrix to store recovered latent skills for each child and age
    latent_skills = zeros(n_c, n_age)

    # Loop over each child
    for h in 1:n_c
        # Loop over each age
        for t in 1:n_age
            # Get the observed SDQ score for child h at age t
            Z = data[(data[:, 1] .== h) .& (data[:, 2] .== t), 4][1]

            # Approximate the probability of success using the observed SDQ score
            p_t = Z / 10  # SDQ score ranges from 0 to 10, so Z/10 gives p_t
            
            # Normalize using mu_0 and mu_1 (for age 10-11)
            if p_t == 0.0
                # Set to low latent ability for p_t = 0 (observed Z = 0)
                latent_skills[h, t] = 1.0  # This is equivalent to a very low latent ability
            elseif p_t == 1.0
                ln_theta = (- log(p_min) - mu_0) / mu_1
                latent_skills[h, t] = exp(ln_theta)  # This is equivalent to a very low latent ability
            else
                # Use the logistic inversion to recover ln(θ) and latent ability θ
                # Applying the formula for logistic function inversion
                ln_theta = (log(p_t) - log(1 - p_t) - mu_0) / mu_1
                theta = exp(ln_theta)
                
                # Store the recovered latent non-cognitive skill for child h at age t
                latent_skills[h, t] = theta
            end
        end
    end

    return latent_skills
end

function simulate_non_cognitive_skills_with_production(n_c, n_age, data, params2)
    prior_alpha = 1.0
    prior_beta = 1.0
    ncog = zeros(n_c, n_age)

    for h in 1:n_c
        ncog[h, 1] = data[(data[:, 1] .== h) .& (data[:, 2] .== 1), 4][1]
        CPM_1 = data[(data[:, 1] .== h) .& (data[:, 2] .== 1), 7][1]
        τ_1 = data[(data[:, 1] .== h) .& (data[:, 2] .== 1), 7][1]
        CPM_2 = data[(data[:, 1] .== h) .& (data[:, 2] .== 2), 7][1]
        τ_2 = data[(data[:, 1] .== h) .& (data[:, 2] .== 2), 7][1]

        post_alpha = prior_alpha + ncog[h, 1]
        post_beta = prior_beta + (10 - ncog[h, 1])

        p = rand(Beta(post_alpha, post_beta))
        theta = p / (1 - p)

        # Skill production step
        if CPM_1 == -999 || τ_1 == -999
            ncog[h, 2] = data[(data[:, 1] .== h) .& (data[:, 2] .== 2), 4][1]
        else
            ncog[h, 2] = exp(params2[9] + params2[10] * log(ncog[h, 1]) + params2[11] * CPM_1 + params2[22] * log(τ_1))
        end

        if CPM_2 == -999 || τ_2 == -999
            ncog[h, 3] = data[(data[:, 1] .== h) .& (data[:, 2] .== 3), 4][1]
        else
            ncog[h, 3] = exp(params2[12] + params2[13] * log(ncog[h, 2]) + params2[14] * CPM_2 + params2[23] * log(τ_2))
        end
    end

    return ncog
end
