using NLopt, StaticArrays
using Distributions, Random, LinearAlgebra, Base.Threads

function make_pos_def(Σ::Matrix{Float64}; ε::Float64 = 1e-6, max_attempts::Int = 10)
    attempt = 0
    while attempt < max_attempts
        try
            _ = cholesky(Σ)  # Attempt factorization
            return Σ         # Success: return as is
        catch e
            if e isa PosDefException || e isa LinearAlgebra.LAPACKException
                Σ += ε * I   # Add small ridge
                ε *= 10      # Increase ε if needed
                attempt += 1
            else
                rethrow(e)   # Some other error: propagate
            end
        end
    end
    error("Failed to make matrix positive definite after $max_attempts attempts.")
end

# Extract CPM Subproblem
function solve_CPM_subproblem(
    Time, s_it, Y_it, CT_it, #ζ,
    ψcc, ψcn, ψpc, ψpn, α_1, λ_1, λ_2,
    params1c, params1n, β_p, β_c, φ, Vc_star, τ_star, R_star, M_star, Vp_star, t, Rbar_it
)

    # Bounds (match JuMP constraints)
    τ_lower = τ_star + 1e-4                     # enforce τ_p >= τ_star + epsilon
    τ_upper = Time - s_it                # as before
    R_lower = R_star + 1e-4
    M_lower = 0.1

    # If τ_upper <= τ_lower, bail out early with baseline
    if τ_upper < τ_lower
        return τ_star, R_star, M_star, Vp_star - 1e-6
    end

    # NLopt setup - try SLSQP or LD_MMA
    opt = NLopt.Opt(:LD_MMA, 3)  # 3 variables: τ_p, R_p, M_p
    #opt = NLopt.Opt(:LD_MMA, 2)  # 2 variables: R_p, M_p

    # Set bounds
    NLopt.lower_bounds!(opt, [τ_lower, R_lower, M_lower])
    # keep generous upper bounds for R and M to mimic Ipopt's unboundedness (but finite)
    NLopt.upper_bounds!(opt, [τ_upper, Y_it, Y_it])

    # Set bounds
    #NLopt.lower_bounds!(opt, [R_lower, M_lower])
    # keep generous upper bounds for R and M to mimic Ipopt's unboundedness (but finite)
    #NLopt.upper_bounds!(opt, [Y_it, Y_it])

    # Objective function (NLopt maximizes, so negate)
    function objective(x, grad)
        τ_p, R_p, M_p = x
        #R_p, M_p = x
        l_p = Time - s_it - τ_p
        c_p = Y_it + CT_it - M_p - R_p - Rbar_it
        
        if l_p <= 0.0 || c_p <= 0.0 || τ_p <= 0.0 || R_p <= 0.0 || M_p <= 0.0
            return -1e20
        end
        
        #Vp_cpm = (1-φ)*α_1*log(c_p) + φ*λ_1*log(l_p) + φ*λ_2*log(R_p) - ζ +
        #        β_p*ψpc*(params1c[5]*log(τ_p) + params1c[6]*log(M_p)) +
        #        β_p*ψpn*(params1n[5]*log(τ_p) + params1n[6]*log(M_p))
        Vp_cpm = (1-φ)*α_1*log(c_p) + φ*λ_1*log(l_p) + φ*λ_2*log(Rbar_it + R_p) +
                β_p*ψpc*(params1c[t*5-2]*log(τ_p) + params1c[t*5-1]*log(M_p)) +
                β_p*ψpn*(params1n[t*5-2]*log(τ_p) + params1n[t*5-1]*log(M_p))
        
        return Vp_cpm  # NLopt maximizes
    end
    
    # Participation Constraint: Vc_cpm - Vc_star >= 0
    function constr_vc(x, grad)
        τ_p, R_p, M_p = x
        #R_p, M_p = x
        l_p = Time - s_it - τ_p
        # protect domain
        if l_p <= 0.0 || τ_p <= 0.0 || R_p <= 0.0 || M_p <= 0.0
            return -Inf
        end

        Vc_cpm = λ_1*log(l_p) + λ_2*log(Rbar_it + R_p) +
                 β_c*ψcc*(params1c[t*5-2]*log(τ_p) + params1c[t*5-1]*log(M_p)) +
                 β_c*ψcn*(params1n[t*5-2]*log(τ_p) + params1n[t*5-1]*log(M_p))
        
        return Vc_cpm - Vc_star
    end

    # Constraint: l_p >= 1e-4  -> (l_p - 1e-4) >= 0
    function constr_l(x, grad)
        τ_p = x[1]
        l_p = Time - s_it - τ_p
        return l_p - 1e-4
    end

    # Constraint: c_p >= 1e-4  -> (c_p - 1e-4) >= 0
    function constr_c(x, grad)
        R_p = x[2]
        M_p = x[3]
        #R_p = x[1]; M_p = x[2]
        c_p = Y_it + CT_it - M_p - R_p - Rbar_it
        return c_p - 1e-4
    end

    NLopt.max_objective!(opt, objective)
    NLopt.inequality_constraint!(opt, constr_vc, 1e-8)
    NLopt.inequality_constraint!(opt, constr_l, 1e-8)
    NLopt.inequality_constraint!(opt, constr_c, 1e-8)
    
    NLopt.xtol_rel!(opt, 1e-4)
    #NLopt.maxeval!(opt, 2000)

    # Starting point: mirror JuMP start (mid τ, R_star, M_star)
    τ_start = clamp((τ_lower + τ_upper)/2, τ_lower, τ_upper)
    x0 = [τ_start, R_lower, max(M_lower, M_star)]
    #x0 = [max(R_lower, R_star), max(M_lower, M_star)]
    
    (optf, optx, ret) = NLopt.optimize(opt, x0)

    # DEBUG INFO
    #println("\n[DEBUG CPM] t=$t")
    #println("  Start guess: ", x0)
    #println("  τ_star=$τ_star, R_star=$R_star, M_star=$M_star, Vp_star=$Vp_star")
    #println("  Result: ret=$ret, optx=$optx, optf=$optf")
    #println("  Improvement? ", optf > Vp_star)
    #println("  Constraint value at optimum: ", constraint(optx, []))
    
    if ret == :SUCCESS || ret == :FTOL_REACHED || ret == :XTOL_REACHED
        return optx[1], optx[2], optx[3], optf
        #return τ_p, optx[1], optx[2], optf
    else
        return τ_star, R_star, M_star, Vp_star - 1e-6
    end

end

# Backward Induction starting from period t = T
function value_func(Z, params1c, params1n, params, n_c, n_age, s, Time, Y, CT, R_bar)

    # Initialize value functions for parents and children at period t
    CPM_opt = zeros(Int64, n_c, n_age) # Differ by i, t, E_c, and CPM
    τ_opt = zeros(n_c, n_age) 
    M_opt = zeros(n_c, n_age) # Differ by i, t, E_c, and CPM
    R_opt = zeros(n_c, n_age) 
    ψcc_store = zeros(n_c, n_age)
    ψcn_store = zeros(n_c, n_age)
    ψpc_store = zeros(n_c, n_age)
    ψpn_store = zeros(n_c, n_age)

    α_0_vec = zeros(n_c)
    α_1_vec = zeros(n_c)
    α_2_vec = zeros(n_c)
    λ_0_vec = zeros(n_c)
    λ_1_vec = zeros(n_c)
    λ_2_vec = zeros(n_c)
    λ_3_vec = zeros(n_c)

    μ = params[1:5]  # Mean vector
    Σ = [
        (params[6])^2  params[11]  params[15]  params[16]  params[17];
        params[11]  (params[7])^2  params[18]  params[19]  params[20];
        params[15]  params[18]  (params[8])^2  params[12]  params[13];
        params[16]  params[19]  params[12]  (params[9])^2  params[14];
        params[17]  params[20]  params[13]  params[14] (params[10])^2
    ] # Covariance matrix

    # Fix it only if needed
    Σ = make_pos_def(Σ)
    L = cholesky(Symmetric(Σ)).L
    #dist = MvNormal(SVector{5}(μ), Σ)

    # Parents
    Threads.@threads for i in 1:n_c
    #for i in 1:n_c
        # CRNs: fixed per-household base shocks
        z_i = @view Z[:, i]
        #u_i = U[i]

        ν = μ .+ L * z_i                  # ν uses current μ,Σ but same z_i
        #ζ = -log(u_i) * params[11]                # inverse-CDF for Exponential

        # map ν -> (α,λ)
        #threadid = Threads.threadid()
        #local_rng = MersenneTwister(1234 + threadid) # To avoid thread contention
        #ν = rand(local_rng, dist)

        # Random draw from preference parameters
        α_0 = exp(ν[1]) / (1 + exp(ν[1]) + exp(ν[2]))
        α_1 = exp(ν[2]) / (1 + exp(ν[1]) + exp(ν[2]))
        α_2 = 1 / (1 + exp(ν[1]) + exp(ν[2]))

        λ_sum = 1 + exp(ν[3]) + exp(ν[4]) + exp(ν[5])
        λ_0 = exp(ν[3]) / λ_sum
        λ_1 = exp(ν[4]) / λ_sum
        λ_2 = exp(ν[5]) / λ_sum
        λ_3 = 1 / λ_sum

        # Store the current values of alpha and lambda parameters
        α_0_vec[i] = α_0
        α_1_vec[i] = α_1
        α_2_vec[i] = α_2
        λ_0_vec[i] = λ_0
        λ_1_vec[i] = λ_1
        λ_2_vec[i] = λ_2
        λ_3_vec[i] = λ_3

        # CPM cost for different cases
        #ζ = rand(local_rng, Exponential(params[11]))
        
        for t = n_age:-1:1
            medu_i = medu[(i-1)*n_age+t]
            #ζ = exp(params[22]*t + params[23]*medu_i + U[i, t])
            # Initialize
            if t == n_age
                ψcc_store[i, t] = λ_0 * (1 - (β_c)^(t + 1)) / (1 - β_c) # ψ_{c, 4}^C
                ψcn_store[i, t] = λ_3 * (1 - (β_c)^(t + 1)) / (1 - β_c) # ψ_{c, 4}^N
                ψpc_store[i, t] = (α_0*(1 - φ)*(1 - (β_p)^(t + 1)) / (1 - β_p)) + φ*ψcc_store[i, t] # ψ_{p, 4}^C
                ψpn_store[i, t] = (α_2*(1 - φ)*(1 - (β_p)^(t + 1)) / (1 - β_p)) + φ*ψcn_store[i, t] # ψ_{p, 4}^N
            elseif t == 1 || t == 2 # ψ_{c, 3}^C, ψ_{c, 3}^N, ψ_{p, 3}^C, ψ_{p, 3}^N, ψ_{c, 2}^C, ψ_{c, 2}^N, ψ_{p, 2}^C, ψ_{p, 2}^N
                ψcc_store[i, t] = λ_0 + β_c*(params1c[t*5+2]*ψcc_store[i, t+1] + params1n[t*5+5]*ψcn_store[i, t+1]) # ψ_{c, T}^C
                ψcn_store[i, t] = λ_3 + β_c*(params1n[t*5+2]*ψcc_store[i, t+1] + params1c[t*5+5]*ψcn_store[i, t+1]) # ψ_{c, T}^N
                ψpc_store[i, t] = (1-φ)*α_0 + φ*λ_0 + β_p*(params1c[t*5+2]*ψcc_store[i, t+1] + params1n[t*5+5]*ψcn_store[i, t+1]) # ψ_{p, T + 1}^C
                ψpn_store[i, t] = (1-φ)*α_2 + φ*λ_3 + β_p*(params1n[t*5+2]*ψcc_store[i, t+1] + params1c[t*5+5]*ψcn_store[i, t+1]) # ψ_{p, T + 1}^N
            end
            
            ψcc, ψcn, ψpc, ψpn = ψcc_store[i, t], ψcn_store[i, t], ψpc_store[i, t], ψpn_store[i, t]
            Y_it = Y[(i-1)*n_age + t]
            s_it = s[(i-1)*n_age+t]
            CT_it = CT[(i-1)*n_age+t]
            Rbar_it = R_bar[(i-1)*n_age+t]

            χ = (1-φ)*α_1 + φ*λ_2 + β_p*(ψpc*params1c[t*5-1]+ψpn*params1n[t*5-1])
            c_star = (1-φ)*α_1*(Y_it+CT_it) / χ # Parents' Consumption
            M_star = β_p*(ψpc*params1c[t*5-1]+ψpn*params1n[t*5-1])*(Y_it+CT_it) / χ # Educational Investment Goods
            R_star = max(0.0, (φ*λ_2*(Y_it+CT_it) / χ) - Rbar_it)

            #C = β_p*(ψpc*params1c[5] + ψpn*params1n[5])
            A = β_c*(ψcc*params1c[t*5-2] + ψcn*params1n[t*5-2])
            τ_star = A*(Time-s_it) / (λ_1+A) # Closed-form optimal child effort (no CPM)
            l_star = Time - s_it - τ_star
            #println("τ_star: ", τ_star , "R_star: ", R_star+1e-6 , "M_star: ", M_star )
            Vc_star = λ_1*log(l_star) + λ_2*log(Rbar_it + R_star) + β_c*ψcc*(params1c[t*5-2]*log(τ_star) + params1c[t*5-1]*log(M_star)) +
             β_c*ψcn*(params1n[t*5-2]*log(τ_star) + params1n[t*5-1]*log(M_star))
            Vp_star = (1-φ)*α_1*log(c_star) + φ*λ_1*log(l_star) + φ*λ_2*log(Rbar_it + R_star) +
             β_p*ψpc*(params1c[t*5-2]*log(τ_star) + params1c[t*5-1]*log(M_star)) +
              β_p*ψpn*(params1n[t*5-2]*log(τ_star) + params1n[t*5-1]*log(M_star))
            #τ_p = C*(Time-s_it) / (φ*λ_1+C)
            
            τ_cpm_opt, R_cpm_opt, M_cpm_opt, Vp_cpm_val = solve_CPM_subproblem(
                Time, s_it, Y_it, CT_it, #ζ,
                ψcc, ψcn, ψpc, ψpn, α_1, λ_1, λ_2,
                params1c, params1n, β_p, β_c, φ, Vc_star, τ_star, R_star, M_star, Vp_star, t, Rbar_it
            )

            if Vp_cpm_val > Vp_star
                #println("\n=> Parents will implement CPM.")
                τ_opt[i, t] = τ_cpm_opt
                R_opt[i, t] = R_cpm_opt
                M_opt[i, t] = M_cpm_opt
                CPM_opt[i, t] = 1
            elseif Vp_cpm_val <= Vp_star
                #println("\n=> Parents will not implement CPM.")
                τ_opt[i, t] = τ_star
                R_opt[i, t] = R_star
                M_opt[i, t] = M_star
                CPM_opt[i, t] = 0
            end
        end 
    end

    return CPM_opt, τ_opt, M_opt, R_opt,
           mean(α_0_vec), std(α_0_vec), mean(α_1_vec), std(α_1_vec), mean(α_2_vec), std(α_2_vec),
           mean(λ_0_vec), std(λ_0_vec), mean(λ_1_vec), std(λ_1_vec), mean(λ_2_vec), std(λ_2_vec), mean(λ_3_vec), std(λ_3_vec)
end
