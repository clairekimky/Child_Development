
# ==============================================================================
# Project Title: The Dynamics of Parent-Child Interactions in Shaping Cognitive and
# Non-cognitive Development during Adolescence
# Author: Claire Kim
# Institution: University of Wisconsin-Madison
# Start Date: 02/04/2025
# Description:
#
# This script implements a dynamic model of child development with a focus on
# parental incentive and self-investment on cognitive and non-cognitive traits
# using backward induction to estimate parameters
# ==============================================================================

using Random
using LinearAlgebra
using Statistics
using DataFrames
using CSV  # Added this for loading the CSV
using Distributions

# Set "include" path
cd("/Users/clairekim/Desktop/Project/Child Development/LSAC/code/")

# Backward Induction starting from period t = T
function value_func(params1, params2, n_c, n_age, ncog, s, Time, Y)

    Random.seed!(1234)

    # Initialize value functions for parents and children at period t
    CPM_opt = zeros(Int, n_c, n_age)
    CPM_cf1 = zeros(Int, n_c, n_age)
    CPM_cf2 = zeros(Int, n_c, n_age)
    CPM_cf3 = zeros(Int, n_c, n_age)
    CPM_cf4 = zeros(Int, n_c, n_age)
    τ_opt = zeros(n_c, n_age) # Differ by i, t, E_c, and CPM
    τ_cf1 = zeros(n_c, n_age) # Differ by i, t, E_c, and CPM
    τ_cf2 = zeros(n_c, n_age) # Differ by i, t, E_c, and CPM
    τ_cf3 = zeros(n_c, n_age) # Differ by i, t, E_c, and CPM
    τ_cf4 = zeros(n_c, n_age) # Differ by i, t, E_c, and CPM
    M_opt = zeros(n_c, n_age) # Differ by i, t, E_c, and CPM
    R_opt = zeros(n_c, n_age) # Differ by i, t, E_c, and CPM
    R_cf1 = zeros(n_c, n_age) # Differ by i, t, E_c, and CPM
    R_cf2 = zeros(n_c, n_age) # Differ by i, t, E_c, and CPM
    R_cf3 = zeros(n_c, n_age) # Differ by i, t, E_c, and CPM
    R_cf4 = zeros(n_c, n_age) # Differ by i, t, E_c, and CPM
    ψcc_store = zeros(n_c, n_age)
    ψcn_store = zeros(n_c, n_age)
    ψpc_store = zeros(n_c, n_age)
    ψpn_store = zeros(n_c, n_age)
    Vc_opt = zeros(n_c, n_age)
    Vp_opt = zeros(n_c, n_age)
    Vc_cf1 = zeros(n_c, n_age)
    Vp_cf1 = zeros(n_c, n_age)
    Vc_cf2 = zeros(n_c, n_age)
    Vp_cf2 = zeros(n_c, n_age)
    Vc_cf3 = zeros(n_c, n_age)
    Vp_cf3 = zeros(n_c, n_age)
    Vc_cf4 = zeros(n_c, n_age)
    Vp_cf4 = zeros(n_c, n_age)

    # Lists to store the estimated alpha and lambda values
    alpha_0_vals, alpha_1_vals, alpha_2_vals = [], [], []
    lambda_0_vals, lambda_1_vals, lambda_2_vals, lambda_3_vals = [], [], [], []

    μ = params2[1:5]  # Mean vector
    σ = params2[6:10] .^ 2  # Variance for each parameter
    Σ = Diagonal(σ)  # Covariance matrix (diagonal)

    # Parents
    for i = 1:n_c
        # Random draw from preference parameters
        ν = rand(MvNormal(μ, Σ)) 
        α_0 = exp(ν[1]) / (1 + exp(ν[1]) + exp(ν[2]))
        α_1 = exp(ν[2]) / (1 + exp(ν[1]) + exp(ν[2]))
        α_2 = 1 / (1 + exp(ν[1]) + exp(ν[2]))
        λ_0 = exp(ν[3]) / (1 + exp(ν[3]) + exp(ν[4]) + exp(ν[5]))
        λ_1 = exp(ν[4]) / (1 + exp(ν[3]) + exp(ν[4]) + exp(ν[5]))
        λ_2 = exp(ν[5]) / (1 + exp(ν[3]) + exp(ν[4]) + exp(ν[5]))
        λ_3 = 1 / (1 + exp(ν[3]) + exp(ν[4]) + + exp(ν[5]))
        #println("α_0: ", α_0 , "α_1: ", α_1 , "α_2: ", α_2 , "λ_0: ", λ_0 , "λ_1: ", λ_1 , "λ_2: ", λ_2 , "λ_3: ", λ_3)

        # Store the current values of alpha and lambda parameters
        push!(alpha_0_vals, α_0)
        push!(alpha_1_vals, α_1)
        push!(alpha_2_vals, α_2)
        push!(lambda_0_vals, λ_0)
        push!(lambda_1_vals, λ_1)
        push!(lambda_2_vals, λ_2)
        push!(lambda_3_vals, λ_3)

        # CPM cost for different cases
        ζ = rand(Exponential((params2[11])^2))
        ζ_cf1, ζ_cf2, ζ_cf3, ζ_cf4 = Inf, 0, ζ, 0

        for t = n_age:n_age
            # Initialize
            χ, c, A, C, l_c, l_p, τ_c, τ_p, R_c, R_p = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            ψcc_store[i, t] = λ_0 * (1 - (β_c)^(t + 1)) / (1 - β_c) # ψ_{c, T + 1}^C
            ψcn_store[i, t] = λ_3 * (1 - (β_c)^(t + 1)) / (1 - β_c) # ψ_{c, T + 1}^N
            ψpc_store[i, t] = (α_0*(1 - φ)*(1 - (β_p)^(t + 1)) / (1 - β_p)) + φ*ψcc_store[i, t] # ψ_{p, T + 1}^C
            ψpn_store[i, t] = (α_2*(1 - φ)*(1 - (β_p)^(t + 1)) / (1 - β_p)) + φ*ψcn_store[i, t] # ψ_{p, T + 1}^N
            #println("ψcc: ", ψcc_store[i, t] , "ψcn: ", ψcn_store[i, t] , "ψpc: ", ψpc_store[i, t] ,
            # "ψpn: ", ψpn_store[i, t] , )
            χ = (1-φ)*α_1 + φ*λ_2 + β_p*ψpc_store[i, t]*params1[14]
            c = (1-φ)*α_1*Y[(i-1)*n_age+t] / χ # Parents' Consumption
            R_c = φ*λ_2*Y[(i-1)*n_age+t] / χ # Child's Allowances
            M_opt[i, t] = β_p*ψpc_store[i, t]*params1[14]*Y[(i-1)*n_age+t] / χ # Educational Investment Goods
            C = β_p*(ψpc_store[i, t]*(params1[13] + params1[15]*log(ncog[(i-1)*n_age+t])) + ψpn_store[i, t]*params2[23])
            A = β_c*(ψcc_store[i, t]*(params1[13] + params1[15]*log(ncog[(i-1)*n_age+t])) + ψcn_store[i, t]*params2[23])
            τ_c = A*(Time - s[(i-1)*n_age+t]) / (λ_1 + A) # No CPM
            τ_p = C*(Time - s[(i-1)*n_age+t]) / (φ*λ_1 + C) # CPM
            l_c = Time - s[(i-1)*n_age+t] - τ_c
            l_p = Time - s[(i-1)*n_age+t] - τ_p
            R_p = exp((λ_1*log(l_c/l_p)+λ_2*log(R_c)+log(τ_c/τ_p)*A) / λ_2)
            # Baseline
            if Y[(i-1)*n_age+t]-M_opt[i, t]-R_p <= 0
                CPM_opt[i, t] = 0
                τ_opt[i, t] = τ_c
                R_opt[i, t] = R_c
                Vc_opt[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[11]+
                 params1[13]*log(τ_c)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[23]*log(τ_c))
                Vp_opt[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[11]+params1[13]*log(τ_c)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[23]*log(τ_c))
            elseif Y[(i-1)*n_age+t]-M_opt[i, t]-R_p > 0
                if (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) >=
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)+
                        β_p*ψpn_store[i, t]*params2[22]-ζ
                    CPM_opt[i, t] = 0
                    τ_opt[i, t] = τ_c
                    R_opt[i, t] = R_c
                    Vc_opt[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[11]+
                 params1[13]*log(τ_c)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[23]*log(τ_c))
                Vp_opt[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[11]+params1[13]*log(τ_c)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[23]*log(τ_c))
                elseif (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) <
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)+
                        β_p*ψpn_store[i, t]*params2[22]-ζ
                    CPM_opt[i, t] = 1
                    τ_opt[i, t] = τ_p
                    R_opt[i, t] = R_p
                    Vc_opt[i, t] = λ_1*log(l_p)+λ_2*log(R_p)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[11]+
                 params1[13]*log(τ_p)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[22]+params2[23]*log(τ_c))
                Vp_opt[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_p)+φ*λ_2*log(R_p)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[11]+params1[13]*log(τ_p)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[22]+params2[23]*log(τ_p))-ζ
                end
            end
            # CPM is very costly
            if Y[(i-1)*n_age+t]-M_opt[i, t]-R_p <= 0
                CPM_cf1[i, t] = 0
                τ_cf1[i, t] = τ_c
                R_cf1[i, t] = R_c
                Vc_cf1[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[11]+
                 params1[13]*log(τ_c)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[23]*log(τ_c))
                Vp_cf1[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[11]+params1[13]*log(τ_c)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[23]*log(τ_c))
            elseif Y[(i-1)*n_age+t]-M_opt[i, t]-R_p > 0
                if (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) >=
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)+
                    β_p*ψpn_store[i, t]*params2[22]-ζ_cf1
                    CPM_cf1[i, t] = 0
                    τ_cf1[i, t] = τ_c
                    R_cf1[i, t] = R_c
                    Vc_cf1[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[11]+
                 params1[13]*log(τ_c)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[23]*log(τ_c))
                Vp_cf1[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[11]+params1[13]*log(τ_c)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[23]*log(τ_c))
                elseif (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) <
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)+
                        β_p*ψpn_store[i, t]*params2[22]-ζ_cf1
                    CPM_cf1[i, t] = 1
                    τ_cf1[i, t] = τ_p
                    R_cf1[i, t] = R_p
                    Vc_cf1[i, t] = λ_1*log(l_p)+λ_2*log(R_p)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[11]+
                 params1[13]*log(τ_p)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[22]+params2[23]*log(τ_c))
                Vp_cf1[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_p)+φ*λ_2*log(R_p)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[11]+params1[13]*log(τ_p)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[22]+params2[23]*log(τ_p))-ζ_cf1
                end
            end

            # CPM is affordable
            if Y[(i-1)*n_age+t]-M_opt[i, t]-R_p <= 0
                CPM_cf2[i, t] = 0
                τ_cf2[i, t] = τ_c
                R_cf2[i, t] = R_c
                Vc_cf2[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[11]+
                 params1[13]*log(τ_c)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[23]*log(τ_c))
                Vp_cf2[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[11]+params1[13]*log(τ_c)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[23]*log(τ_c))
            elseif Y[(i-1)*n_age+t]-M_opt[i, t]-R_p > 0
                if (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) >=
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)+
                    β_p*ψpn_store[i, t]*params2[22]-ζ_cf2
                    CPM_cf2[i, t] = 0
                    τ_cf2[i, t] = τ_c
                    R_cf2[i, t] = R_c
                    Vc_cf2[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[11]+
                 params1[13]*log(τ_c)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[23]*log(τ_c))
                Vp_cf2[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[11]+params1[13]*log(τ_c)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[23]*log(τ_c))
                elseif (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) <
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)+
                        β_p*ψpn_store[i, t]*params2[22]-ζ_cf2
                    CPM_cf2[i, t] = 1
                    τ_cf2[i, t] = τ_p
                    R_cf2[i, t] = R_p
                    Vc_cf2[i, t] = λ_1*log(l_p)+λ_2*log(R_p)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[11]+
                 params1[13]*log(τ_p)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[22]+params2[23]*log(τ_c))
                Vp_cf2[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_p)+φ*λ_2*log(R_p)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[11]+params1[13]*log(τ_p)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[22]+params2[23]*log(τ_p))-ζ_cf2
                end
            end
            # CPM price is the same as baseline but no effect on θ^N
            if Y[(i-1)*n_age+t]-M_opt[i, t]-R_p <= 0
                CPM_cf3[i, t] = 0
                τ_cf3[i, t] = τ_c
                R_cf3[i, t] = R_c
                Vc_cf3[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[11]+
                 params1[13]*log(τ_c)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[23]*log(τ_c))
                Vp_cf3[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[11]+params1[13]*log(τ_c)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[23]*log(τ_c))
            elseif Y[(i-1)*n_age+t]-M_opt[i, t]-R_p > 0
                if (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) >=
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)-ζ_cf3
                    CPM_cf3[i, t] = 0
                    τ_cf3[i, t] = τ_c
                    R_cf3[i, t] = R_c
                    Vc_cf3[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[11]+
                 params1[13]*log(τ_c)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[23]*log(τ_c))
                Vp_cf3[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[11]+params1[13]*log(τ_c)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[23]*log(τ_c))
                elseif (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) <
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)-ζ_cf3
                    CPM_cf3[i, t] = 1
                    τ_cf3[i, t] = τ_p
                    R_cf3[i, t] = R_p
                    Vc_cf3[i, t] = λ_1*log(l_p)+λ_2*log(R_p)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[11]+
                 params1[13]*log(τ_p)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[23]*log(τ_c))
                Vp_cf3[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_p)+φ*λ_2*log(R_p)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[11]+params1[13]*log(τ_p)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[23]*log(τ_p))-ζ_cf3
                end
            end
            # CPM is affordable and no effect on θ^N
            if Y[(i-1)*n_age+t]-M_opt[i, t]-R_p <= 0
                CPM_cf4[i, t] = 0
                τ_cf4[i, t] = τ_c
                R_cf4[i, t] = R_c
                Vc_cf4[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[11]+
                 params1[13]*log(τ_c)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[23]*log(τ_c))
                Vp_cf4[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[11]+params1[13]*log(τ_c)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[23]*log(τ_c))
            elseif Y[(i-1)*n_age+t]-M_opt[i, t]-R_p > 0
                if (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) >=
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)-ζ_cf4
                    CPM_cf4[i, t] = 0
                    τ_cf4[i, t] = τ_c
                    R_cf4[i, t] = R_c
                    Vc_cf4[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[11]+
                 params1[13]*log(τ_c)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[23]*log(τ_c))
                Vp_cf4[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[11]+params1[13]*log(τ_c)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[23]*log(τ_c))
                elseif (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) <
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)-ζ_cf4
                    CPM_cf4[i, t] = 1
                    τ_cf4[i, t] = τ_p
                    R_cf4[i, t] = R_p
                    Vc_cf4[i, t] = λ_1*log(l_p)+λ_2*log(R_p)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[11]+
                 params1[13]*log(τ_p)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[23]*log(τ_c))
                Vp_cf4[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_p)+φ*λ_2*log(R_p)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[11]+params1[13]*log(τ_p)+params1[14]*log(M_opt[i,t])+params1[15]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[20]+params2[21]*log(ncog[(i-1)*n_age+t])+params2[23]*log(τ_p))-ζ_cf4
                end
            end
        end
        # Continue with period t <= T - 1
        for t=n_age-1:n_age-1
            # Initialize
            χ, c, A, C, l_c, l_p, τ_c, τ_p, R_c, R_p = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            ψcc_store[i, t] = λ_0 + β_c*params1[12]*ψcc_store[i, t + 1] # ψ_{c, T}^C
            ψcn_store[i, t] = λ_3 + β_c*params2[21]*ψcn_store[i, t + 1] # ψ_{c, T}^N
            ψpc_store[i, t] = (1 - φ)*α_0 + φ*λ_0 + β_p*params1[12]*ψpc_store[i, t + 1] # ψ_{p, T + 1}^C
            ψpn_store[i, t] = (1 - φ)*α_2 + φ*λ_3 + β_p*params2[21]*ψpn_store[i, t + 1] # ψ_{p, T + 1}^N
            χ = (1-φ)*α_1 + φ*λ_2 + β_p*ψpc_store[i, t]*params1[9]
            c = (1-φ)*α_1*Y[(i-1)*n_age+t] / χ # Parents' Consumption
            R_c = φ*λ_2*Y[(i-1)*n_age+t] / χ # Child's Allowances
            M_opt[i, t] = β_p*ψpc_store[i, t]*params1[9]*Y[(i-1)*n_age+t] / χ # Educational Investment Goods
            C = β_p*(ψpc_store[i, t]*(params1[8] + params1[10]*log(ncog[(i-1)*n_age+t])) + ψpn_store[i, t]*params2[19])
            A = β_c*(ψcc_store[i, t]*(params1[8] + params1[10]*log(ncog[(i-1)*n_age+t])) + ψcn_store[i, t]*params2[19])
            τ_c = A*(Time - s[(i-1)*n_age+t]) / (λ_1 + A)
            τ_p = C*(Time - s[(i-1)*n_age+t]) / (φ*λ_1 + C)
            l_c = Time - s[(i-1)*n_age+t] - τ_c
            l_p = Time - s[(i-1)*n_age+t] - τ_p
            R_p = exp((λ_1*log(l_c/l_p) + λ_2*log(R_c)+log(τ_c/τ_p)*A) / λ_2)
            # Baseline
            if Y[(i-1)*n_age+t]-M_opt[i, t]-R_p <= 0
                CPM_opt[i, t] = 0
                τ_opt[i, t] = τ_c
                R_opt[i, t] = R_c
                Vc_opt[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[6]+
                 params1[8]*log(τ_c)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[19]*log(τ_c))
                Vp_opt[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[8]+params1[8]*log(τ_c)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[19]*log(τ_c))
            elseif Y[(i-1)*n_age+t]-M_opt[i, t]-R_p > 0
                if (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) >=
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)+
                        β_p*ψpn_store[i, t]*params2[18]-ζ
                    CPM_opt[i, t] = 0
                    τ_opt[i, t] = τ_c
                    R_opt[i, t] = R_c
                    Vc_opt[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[6]+
                 params1[8]*log(τ_c)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[19]*log(τ_c))
                Vp_opt[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[8]+params1[8]*log(τ_c)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[19]*log(τ_c))
                elseif (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) <
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)+
                        β_p*ψpn_store[i, t]*params2[18]-ζ
                    CPM_opt[i, t] = 1
                    τ_opt[i, t] = τ_p
                    R_opt[i, t] = R_p
                    Vc_opt[i, t] = λ_1*log(l_p)+λ_2*log(R_p)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[6]+
                 params1[8]*log(τ_p)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[18]+params2[19]*log(τ_c))
                Vp_opt[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_p)+φ*λ_2*log(R_p)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[6]+params1[8]*log(τ_p)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[18]+params2[19]*log(τ_c))-ζ
                end
            end
            # CPM is very costly
            if Y[(i-1)*n_age+t]-M_opt[i, t]-R_p <= 0
                CPM_cf1[i, t] = 0
                τ_cf1[i, t] = τ_c
                R_cf1[i, t] = R_c
                Vc_cf1[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[6]+
                 params1[8]*log(τ_c)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[19]*log(τ_c))
                Vp_cf1[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[8]+params1[8]*log(τ_c)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[19]*log(τ_c))
            elseif Y[(i-1)*n_age+t]-M_opt[i, t]-R_p > 0
                if (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) >=
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)+
                    β_p*ψpn_store[i, t]*params2[18]-ζ_cf1
                    CPM_cf1[i, t] = 0
                    τ_cf1[i, t] = τ_c
                    R_cf1[i, t] = R_c
                    Vc_cf1[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[6]+
                 params1[8]*log(τ_c)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[19]*log(τ_c))
                Vp_cf1[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[8]+params1[8]*log(τ_c)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[19]*log(τ_c))
                elseif (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) <
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)+
                        β_p*ψpn_store[i, t]*params2[18]-ζ_cf1
                    CPM_cf1[i, t] = 1
                    τ_cf1[i, t] = τ_p
                    R_cf1[i, t] = R_p
                    Vc_cf1[i, t] = λ_1*log(l_p)+λ_2*log(R_p)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[6]+
                 params1[8]*log(τ_p)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[18]+params2[19]*log(τ_c))
                Vp_cf1[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_p)+φ*λ_2*log(R_p)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[6]+params1[8]*log(τ_p)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[18]+params2[19]*log(τ_p))-ζ_cf1
                end
            end
            # CPM is affordable
            if Y[(i-1)*n_age+t]-M_opt[i, t]-R_p <= 0
                CPM_cf2[i, t] = 0
                τ_cf2[i, t] = τ_c
                R_cf2[i, t] = R_c
                Vc_cf2[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[6]+
                 params1[8]*log(τ_c)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[19]*log(τ_c))
                Vp_cf2[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[8]+params1[8]*log(τ_c)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[19]*log(τ_c))
            elseif Y[(i-1)*n_age+t]-M_opt[i, t]-R_p > 0
                if (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) >=
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)+
                     β_p*ψpn_store[i, t]*params2[18]-ζ_cf2
                    CPM_cf2[i, t] = 0
                    τ_cf2[i, t] = τ_c
                    R_cf2[i, t] = R_c
                    Vc_cf2[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[6]+
                 params1[8]*log(τ_c)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[19]*log(τ_c))
                Vp_cf2[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[8]+params1[8]*log(τ_c)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[19]*log(τ_c))
                elseif (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) <
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)+
                        β_p*ψpn_store[i, t]*params2[18]-ζ_cf2
                    CPM_cf2[i, t] = 1
                    τ_cf2[i, t] = τ_p
                    R_cf2[i, t] = R_p
                    Vc_cf2[i, t] = λ_1*log(l_p)+λ_2*log(R_p)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[6]+
                 params1[8]*log(τ_p)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[18]+params2[19]*log(τ_c))
                Vp_cf2[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_p)+φ*λ_2*log(R_p)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[6]+params1[8]*log(τ_p)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[18]+params2[19]*log(τ_p))-ζ_cf2
                end
            end
            # CPM is baseline price but no impact on θ^N
            if Y[(i-1)*n_age+t]-M_opt[i, t]-R_p <= 0
                CPM_cf3[i, t] = 0
                τ_cf3[i, t] = τ_c
                R_cf3[i, t] = R_c
                Vc_cf3[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[6]+
                 params1[8]*log(τ_c)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[19]*log(τ_c))
                Vp_cf3[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[8]+params1[8]*log(τ_c)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[19]*log(τ_c))
            elseif Y[(i-1)*n_age+t]-M_opt[i, t]-R_p > 0
                if (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) >=
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)-ζ_cf3
                    CPM_cf3[i, t] = 0
                    τ_cf3[i, t] = τ_c
                    R_cf3[i, t] = R_c
                    Vc_cf3[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[6]+
                 params1[8]*log(τ_c)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[19]*log(τ_c))
                Vp_cf3[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[8]+params1[8]*log(τ_c)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[19]*log(τ_c))
                elseif (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) <
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)-ζ_cf3
                    CPM_cf3[i, t] = 1
                    τ_cf3[i, t] = τ_p
                    R_cf3[i, t] = R_p
                    Vc_cf3[i, t] = λ_1*log(l_p)+λ_2*log(R_p)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[6]+
                 params1[8]*log(τ_p)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[19]*log(τ_c))
                Vp_cf3[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_p)+φ*λ_2*log(R_p)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[6]+params1[8]*log(τ_p)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[19]*log(τ_p))-ζ_cf3
                end
            end
            # CPM is affordable and no impact on θ^N
            if Y[(i-1)*n_age+t]-M_opt[i, t]-R_p <= 0
                CPM_cf4[i, t] = 0
                τ_cf4[i, t] = τ_c
                R_cf4[i, t] = R_c
                Vc_cf4[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[6]+
                 params1[8]*log(τ_c)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[19]*log(τ_c))
                Vp_cf4[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[8]+params1[8]*log(τ_c)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[19]*log(τ_c))
            elseif Y[(i-1)*n_age+t]-M_opt[i, t]-R_p > 0
                if (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) >=
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)-ζ_cf4
                    CPM_cf4[i, t] = 0
                    τ_cf4[i, t] = τ_c
                    R_cf4[i, t] = R_c
                    Vc_cf4[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[6]+
                 params1[8]*log(τ_c)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[19]*log(τ_c))
                Vp_cf4[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[8]+params1[8]*log(τ_c)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[19]*log(τ_c))
                elseif (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) <
                (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)-ζ_cf4
                    CPM_cf4[i, t] = 1
                    τ_cf4[i, t] = τ_p
                    R_cf4[i, t] = R_p
                    Vc_cf4[i, t] = λ_1*log(l_p)+λ_2*log(R_p)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[6]+
                 params1[8]*log(τ_p)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[19]*log(τ_c))
                Vp_cf4[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_p)+φ*λ_2*log(R_p)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[6]+params1[8]*log(τ_p)+params1[9]*log(M_opt[i,t])+params1[10]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[16]+params2[17]*log(ncog[(i-1)*n_age+t])+params2[19]*log(τ_p))-ζ_cf4
                end
            end
        end 
        for t=n_age-2:n_age-2
            # Initialize
            χ, c, A, C, l_c, l_p, τ_c, τ_p, R_c, R_p = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            ψcc_store[i, t] = λ_0 + β_c*params1[7]*ψcc_store[i, t + 1] # ψ_{c, T}^C
            ψcn_store[i, t] = λ_3 + β_c*params2[17]*ψcn_store[i, t + 1] # ψ_{c, T}^N
            ψpc_store[i, t] = (1 - φ)*α_0 + φ*λ_0 + β_p*params1[7]*ψpc_store[i, t + 1] # ψ_{p, T + 1}^C
            ψpn_store[i, t] = (1 - φ)*α_2 + φ*λ_3 + β_p*params2[17]*ψpn_store[i, t + 1] # ψ_{p, T + 1}^N
            χ = (1-φ)*α_1 + φ*λ_2 + β_p*ψpc_store[i, t]*params1[4]
            c = (1-φ)*α_1*Y[(i-1)*n_age+t] / χ # Parents' Consumption
            R_c = φ*λ_2*Y[(i-1)*n_age+t] / χ # Child's Allowances
            M_opt[i, t] = β_p*ψpc_store[i, t]*params1[4]*Y[(i-1)*n_age+t] / χ # Educational Investment Goods
            C = β_p*(ψpc_store[i, t]*(params1[3] + params1[5]*log(ncog[(i-1)*n_age+t])) + ψpn_store[i, t]*params2[15])
            A = β_c*(ψcc_store[i, t]*(params1[3] + params1[5]*log(ncog[(i-1)*n_age+t])) + ψcn_store[i, t]*params2[15])
            τ_c = A*(Time - s[(i-1)*n_age+t]) / (λ_1 + A)
            τ_p = C*(Time - s[(i-1)*n_age+t]) / (φ*λ_1 + C)
            l_c = Time - s[(i-1)*n_age+t] - τ_c
            l_p = Time - s[(i-1)*n_age+t] - τ_p
            R_p = exp((λ_1*log(l_c/l_p) + λ_2*log(R_c)+log(τ_c/τ_p)*A) / λ_2)
            if Y[(i-1)*n_age+t]-M_opt[i, t]-R_p <= 0
                CPM_opt[i, t] = 0
                τ_opt[i, t] = τ_c
                R_opt[i, t] = R_c
                Vc_opt[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[1]+
                 params1[3]*log(τ_c)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[15]*log(τ_c))
                Vp_opt[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[1]+params1[3]*log(τ_c)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[15]*log(τ_c))
            elseif Y[(i-1)*n_age+t]-M_opt[i, t]-R_p > 0
                if (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) >=
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)+
                    β_p*ψpn_store[i, t]*params2[14]-ζ
                    CPM_opt[i, t] = 0
                    τ_opt[i, t] = τ_c
                    R_opt[i, t] = R_c
                    Vc_opt[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[1]+
                 params1[3]*log(τ_c)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[15]*log(τ_c))
                Vp_opt[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[1]+params1[3]*log(τ_c)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[15]*log(τ_c))
                elseif (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) <
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)+
                    β_p*ψpn_store[i, t]*params2[14]-ζ
                    CPM_opt[i, t] = 1
                    τ_opt[i, t] = τ_p
                    R_opt[i, t] = R_p
                    Vc_opt[i, t] = λ_1*log(l_p)+λ_2*log(R_p)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[1]+
                 params1[3]*log(τ_p)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[14]+params2[15]*log(τ_c))
                Vp_opt[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_p)+φ*λ_2*log(R_p)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[1]+params1[3]*log(τ_p)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[14]+params2[15]*log(τ_p))-ζ
                end
            end
            # Scenario 1
            if Y[(i-1)*n_age+t]-M_opt[i, t]-R_p <= 0
                CPM_cf1[i, t] = 0
                τ_cf1[i, t] = τ_c
                R_cf1[i, t] = R_c
                Vc_cf1[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[1]+
                 params1[3]*log(τ_c)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[15]*log(τ_c))
                Vp_cf1[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[1]+params1[3]*log(τ_c)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[15]*log(τ_c))
            elseif Y[(i-1)*n_age+t]-M_opt[i, t]-R_p > 0
                if (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) >=
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)+
                        β_p*ψpn_store[i, t]*params2[14]-ζ_cf1
                    CPM_cf1[i, t] = 0
                    τ_cf1[i, t] = τ_c
                    R_cf1[i, t] = R_c
                    Vc_cf1[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[1]+
                 params1[3]*log(τ_c)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[15]*log(τ_c))
                Vp_cf1[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[1]+params1[3]*log(τ_c)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[15]*log(τ_c))
                elseif (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) <
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)+
                    β_p*ψpn_store[i, t]*params2[14]-ζ_cf1
                    CPM_cf1[i, t] = 1
                    τ_cf1[i, t] = τ_p
                    R_cf1[i, t] = R_p
                    Vc_cf1[i, t] = λ_1*log(l_p)+λ_2*log(R_p)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[1]+
                 params1[3]*log(τ_p)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[14]+params2[15]*log(τ_c))
                Vp_cf1[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_p)+φ*λ_2*log(R_p)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[1]+params1[3]*log(τ_p)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[14]+params2[15]*log(τ_p))-ζ_cf1
                end
            end
            # Scenario 1
            if Y[(i-1)*n_age+t]-M_opt[i, t]-R_p <= 0
                CPM_cf2[i, t] = 0
                τ_cf2[i, t] = τ_c
                R_cf2[i, t] = R_c
                Vc_cf2[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[1]+
                 params1[3]*log(τ_c)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[15]*log(τ_c))
                Vp_cf2[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[1]+params1[3]*log(τ_c)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[15]*log(τ_c))
            elseif Y[(i-1)*n_age+t]-M_opt[i, t]-R_p > 0
                if (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) >=
                        (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)+
                        β_p*ψpn_store[i, t]*params2[14]-ζ_cf2
                    CPM_cf2[i, t] = 0
                    τ_cf2[i, t] = τ_c
                    R_cf2[i, t] = R_c
                    Vc_cf2[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[1]+
                    params1[3]*log(τ_c)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                    β_c*ψcn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[15]*log(τ_c))
                   Vp_cf2[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                    β_p*ψpc_store[i,t]*(params1[1]+params1[3]*log(τ_c)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                    β_p*ψpn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[15]*log(τ_c))
                elseif (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) <
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)+
                    β_p*ψpn_store[i, t]*params2[14]-ζ_cf2
                    CPM_cf2[i, t] = 1
                    τ_cf2[i, t] = τ_p
                    R_cf2[i, t] = R_p
                    Vc_cf2[i, t] = λ_1*log(l_p)+λ_2*log(R_p)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[1]+
                 params1[3]*log(τ_p)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[14]+params2[15]*log(τ_c))
                Vp_cf2[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_p)+φ*λ_2*log(R_p)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[1]+params1[3]*log(τ_p)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[14]+params2[15]*log(τ_p))-ζ_cf2
                end
            end
            # CPM is baseline price but no impact on θ^N
            if Y[(i-1)*n_age+t]-M_opt[i, t]-R_p <= 0
                CPM_cf3[i, t] = 0
                τ_cf3[i, t] = τ_c
                R_cf3[i, t] = R_c
                Vc_cf3[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[1]+
                params1[3]*log(τ_c)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                β_c*ψcn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[15]*log(τ_c))
               Vp_cf3[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                β_p*ψpc_store[i,t]*(params1[1]+params1[3]*log(τ_c)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                β_p*ψpn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[15]*log(τ_c))
            elseif Y[(i-1)*n_age+t]-M_opt[i, t]-R_p > 0
                if (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) >=
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)-ζ_cf3
                    CPM_cf3[i, t] = 0
                    τ_cf3[i, t] = τ_c
                    R_cf3[i, t] = R_c
                    Vc_cf3[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[1]+
                params1[3]*log(τ_c)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                β_c*ψcn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[15]*log(τ_c))
               Vp_cf3[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                β_p*ψpc_store[i,t]*(params1[1]+params1[3]*log(τ_c)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                β_p*ψpn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[15]*log(τ_c))
                elseif (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) <
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)-ζ_cf3
                    CPM_cf3[i, t] = 1
                    τ_cf3[i, t] = τ_p
                    R_cf3[i, t] = R_p
                    Vc_cf3[i, t] = λ_1*log(l_p)+λ_2*log(R_p)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[1]+
                 params1[3]*log(τ_p)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[15]*log(τ_c))
                Vp_cf3[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_p)+φ*λ_2*log(R_p)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[1]+params1[3]*log(τ_p)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[15]*log(τ_p))-ζ_cf3
                end
            end
            # CPM is affordable and no impact on θ^N
            if Y[(i-1)*n_age+t]-M_opt[i, t]-R_p <= 0
                CPM_cf4[i, t] = 0
                τ_cf4[i, t] = τ_c
                R_cf4[i, t] = R_c
                Vc_cf4[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[1]+
                params1[3]*log(τ_c)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                β_c*ψcn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[15]*log(τ_c))
               Vp_cf4[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                β_p*ψpc_store[i,t]*(params1[1]+params1[3]*log(τ_c)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                β_p*ψpn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[15]*log(τ_c))
            elseif Y[(i-1)*n_age+t]-M_opt[i, t]-R_p > 0
                if (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) >=
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)-ζ_cf4
                    CPM_cf4[i, t] = 0
                    τ_cf4[i, t] = τ_c
                    R_cf4[i, t] = R_c
                    Vc_cf4[i, t] = λ_1*log(l_c)+λ_2*log(R_c)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[1]+
                    params1[3]*log(τ_c)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                    β_c*ψcn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[15]*log(τ_c))
                   Vp_cf4[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_c)+φ*λ_2*log(R_c)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                    β_p*ψpc_store[i,t]*(params1[1]+params1[3]*log(τ_c)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                    β_p*ψpn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[15]*log(τ_c))
                elseif (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_c) + φ*λ_1*log(l_c)+φ*λ_2*R_c+C*log(τ_c) <
                    (1-φ)*α_1*log(Y[(i-1)*n_age+t]-M_opt[i, t]-R_p) + φ*λ_1*log(l_p)+φ*λ_2*R_p+C*log(τ_p)-ζ_cf4
                    CPM_cf4[i, t] = 1
                    τ_cf4[i, t] = τ_p
                    R_cf4[i, t] = R_p
                    Vc_cf4[i, t] = λ_1*log(l_p)+λ_2*log(R_p)+λ_3*log(ncog[(i-1)*n_age+t])+β_c*ψcc_store[i,t]*(params1[1]+
                 params1[3]*log(τ_p)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_c*ψcn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[15]*log(τ_c))
                Vp_cf4[i, t] = (1-φ)*α_1*log(c)+φ*λ_1*log(l_p)+φ*λ_2*log(R_p)+((1-φ)*α_2+φ*λ_3)*log(ncog[(i-1)*n_age+t])+
                 β_p*ψpc_store[i,t]*(params1[1]+params1[3]*log(τ_p)+params1[4]*log(M_opt[i,t])+params1[5]*(log(τ_c)*log(ncog[(i-1)*n_age+t])))+
                 β_p*ψpn_store[i,t]*(params2[12]+params2[13]*log(ncog[(i-1)*n_age+t])+params2[15]*log(τ_p))-ζ_cf4
                end
            end
        end 
    end

    # Calculate and report first two moments for alpha and lambda parameters
    mean_α_0 = mean(alpha_0_vals)
    var_α_0 = std(alpha_0_vals)
    mean_α_1 = mean(alpha_1_vals)
    var_α_1 = std(alpha_1_vals)
    mean_α_2 = mean(alpha_2_vals)
    var_α_2 = std(alpha_2_vals)
    
    mean_λ_0 = mean(lambda_0_vals)
    var_λ_0 = std(lambda_0_vals)
    mean_λ_1 = mean(lambda_1_vals)
    var_λ_1 = std(lambda_1_vals)
    mean_λ_2 = mean(lambda_2_vals)
    var_λ_2 = std(lambda_2_vals)
    mean_λ_3 = mean(lambda_3_vals)
    var_λ_3 = std(lambda_3_vals)

    return CPM_opt, τ_opt, M_opt, R_opt, CPM_cf1, CPM_cf2, CPM_cf3, CPM_cf4, 
        τ_cf1, τ_cf2, τ_cf3, τ_cf4, R_cf1, R_cf2, R_cf3, R_cf4, mean_α_0,
        var_α_0, mean_α_1, var_α_1, mean_α_2, var_α_2, mean_λ_0, var_λ_0, mean_λ_1, 
        var_λ_1, mean_λ_2, var_λ_2, mean_λ_3, var_λ_3, Vc_opt, Vc_cf1, Vc_cf2, Vc_cf3, Vc_cf4, 
        Vp_opt, Vp_cf1, Vp_cf2, Vp_cf3, Vp_cf4, ψcc_store, ψcn_store, ψpc_store, ψpn_store

end