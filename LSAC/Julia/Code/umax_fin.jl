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
using NLopt
using LinearAlgebra

function solve_CPM_subproblem_equal_Vc(
    Time, s_it, Y_it, CT_it, params, τ_pstar,
    ψcc, ψpc, α_1, λ_1, λ_2,
    params1c, β_p, β_c, φ, Vc_star, τ_star, R_star, M_star, Vp_star, t, Rbar_it
)

    # Bounds
    τ_lower = 1e-4
    R_lower = 1e-4
    
    τ_upper = τ_pstar
    R_upper = (Y_it + CT_it - Rbar_it) / 100

    # Function to compute M_p that exactly satisfies Vc_cpm = Vc_star
    function M_for_Vc(τ_p, R_p)
        l_p = Time - s_it - τ_p
        term = Vc_star - λ_1*log(l_p) - λ_2*log(Rbar_it + R_p) - β_c*ψcc*params1c[t*4-2]*log(τ_p)
        M_p = exp(term / (β_c*ψcc*params1c[t*4-1]))
        return max(M_p, 1e-6)  # avoid nonpositive
    end

    # Define equality constraint: Vc_cpm - Vc_star = 0
    function equality_constraint(x, grad)
        τ_p, R_p = x
        M_p = M_for_Vc(τ_p, R_p)
        l_p = Time - s_it - τ_p
        Vc_cpm = λ_1*log(l_p) + λ_2*log(Rbar_it + R_p) +
                β_c*ψcc*(params1c[t*4-2]*log(τ_p) + params1c[t*4-1]*log(M_p))
        return Vc_cpm - Vc_star
    end

    # Objective for parental utility
    function objective(x, grad)
        τ_p, R_p = x
        M_p = M_for_Vc(τ_p, R_p)
        l_p = Time - s_it - τ_p
        c_p = Y_it + CT_it - M_p - R_p - Rbar_it
        
        if l_p <= 0.0 || c_p <= 0.0 || τ_p <= 0.0 || M_p <= 0.0 || (Rbar_it + R_p) <= 0.0
            return -1e20
        end

        Vp_cpm = (1-φ)*α_1*log(c_p) + φ*λ_1*log(l_p) + φ*λ_2*log(Rbar_it + R_p) +
                 β_p*ψpc*(params1c[t*4-2]*log(τ_p) + params1c[t*4-1]*log(M_p)) -abs(params[10])*t
        return Vp_cpm
    end

    # NLopt setup: 2D optimization (τ_p, R_p)
    opt = NLopt.Opt(:LN_COBYLA, 2)  # handles nonlinear constraints better
    NLopt.lower_bounds!(opt, [τ_lower, R_lower])
    NLopt.upper_bounds!(opt, [τ_upper, R_upper])
    NLopt.max_objective!(opt, objective)
    NLopt.xtol_rel!(opt, 1e-4)
    NLopt.ftol_rel!(opt, 1e-6)
    NLopt.equality_constraint!(opt, equality_constraint, 1e-6)

    # Starting guess
    τ_start = (τ_star + τ_pstar)/2
    R_start = max(R_star, 1e-4)
    x0 = [τ_start, R_start]

    (optf, optx, ret) = NLopt.optimize(opt, x0)

    # Compute M_p from equality
    τ_cpm, R_cpm = optx
    M_cpm = M_for_Vc(τ_cpm, R_cpm)

    # Compute final Vc to double-check
    l_cpm = Time - s_it - τ_cpm
    Vc_cpm = λ_1*log(l_cpm) + λ_2*log(Rbar_it + R_cpm) +
             β_c*ψcc*(params1c[4*5-2]*log(τ_cpm) + params1c[4*5-1]*log(M_cpm))

    # Safety check: if parental utility lower than baseline, fallback
    if optf <= Vp_star
        return τ_star, R_star, M_star, Vp_star
    end

    return τ_cpm, R_cpm, M_cpm, optf
end

function solve_CPM_grid(
    Time, s_it, Y_it, CT_it, params, τ_pstar,
    ψcc, ψpc, α_1, λ_1, λ_2,
    params1c, β_p, β_c, φ, Vc_star, τ_star, R_star, M_star, Vp_star, t, Rbar_it;
    ngrid=50
)
    # Precompute only the most expensive constants
    enforcement_cost = abs(params[10]) * t
    β_ψ_τ = β_c * ψcc * params1c[t*4-2]
    β_ψ_M = β_c * ψcc * params1c[t*4-1]
    βp_ψ_τ = β_p * ψpc * params1c[t*4-2]
    βp_ψ_M = β_p * ψpc * params1c[t*4-1]
    
    # Bounds
    τ_lower = τ_star
    #τ_lower = 1e-4
    τ_upper = Time - s_it - (1e-4)
    #τ_upper = τ_pstar
    R_lower = 0.0002*Y_it
    #R_upper = Y_it + CT_it - Rbar_it - (1e-4)
    R_upper = 0.03*Y_it
    
    # Early exit
    if τ_upper < τ_lower || R_upper < R_lower
        return τ_star, R_star, M_star, Vp_star
    end
    
    # Simple M computation function
    @inline function compute_M(τ_p, R_p)
        l_p = Time - s_it - τ_p
        term = (Vc_star - λ_1*log(l_p) - λ_2*log(Rbar_it + R_p) - β_ψ_τ*log(τ_p)) / β_ψ_M
        return exp(term)
    end
    
    # Initialize
    best_Vp = Vp_star
    best_τ, best_R, best_M = τ_star, R_star, M_star
    
    # Simple uniform grids (faster to create)
    τ_grid = range(τ_lower, τ_upper, length=ngrid)
    R_grid = range(R_lower, R_upper, length=ngrid)
    
    nτ = length(τ_grid)
    local_results = Vector{Tuple{Float64, Float64, Float64, Float64}}(undef, nτ)
    
    Threads.@threads for iτ in 1:nτ
        τ_p = τ_grid[iτ]
        l_p = Time - s_it - τ_p
        
        if l_p <= 0
            local_results[iτ] = (-Inf, τ_star, R_star, M_star)
            continue
        end
        
        local_best_Vp = -Inf
        local_best_R = R_star
        local_best_M = M_star
        
        for R_p in R_grid
            M_p = compute_M(τ_p, R_p)
            
            if M_p <= 0
                continue
            end
            
            c_p = Y_it + CT_it - M_p - R_p - Rbar_it
            
            if c_p <= 0
                continue
            end
            
            # Direct utility computation
            Vp_cpm = (1-φ)*α_1*log(c_p) + 
                     φ*λ_1*log(l_p) + 
                     φ*λ_2*log(Rbar_it + R_p) +
                     βp_ψ_τ*log(τ_p) + 
                     βp_ψ_M*log(M_p) - 
                     enforcement_cost
            
            if Vp_cpm > local_best_Vp
                local_best_Vp = Vp_cpm
                local_best_R = R_p
                local_best_M = M_p
            end
        end
        
        local_results[iτ] = (local_best_Vp, τ_p, local_best_R, local_best_M)
    end
    
    # Reduction
    for (Vp, τ, R, M) in local_results
        if Vp > best_Vp
            best_Vp, best_τ, best_R, best_M = Vp, τ, R, M
        end
    end
    
    return best_τ, best_R, best_M, best_Vp
end

# Backward Induction starting from period t = T
function value_func(Z, params1c, params, n_c, n_age, s, Time, Y, CT, R_bar, hyp)

    # Initialize value functions for parents and children at period t
    CPM_opt = zeros(Int64, n_c, n_age) # Differ by i, t, E_c, and CPM
    τ_opt = zeros(n_c, n_age) 
    M_opt = zeros(n_c, n_age) # Differ by i, t, E_c, and CPM
    R_opt = zeros(n_c, n_age) 
    ψcc_store = zeros(n_c, n_age)
    ψpc_store = zeros(n_c, n_age)

    α_0_vec = ones(n_c) # Consumption preference (parents)
    α_1_vec = zeros(n_c) # Consumption preference (parents)
    λ_0_vec = zeros(n_c) # Cognitive skill preference
    λ_1_vec = zeros(n_c) # Leisure preference
    λ_2_vec = zeros(n_c) # Consumption preference (child)

    μ = params[1:3]  # Mean vector
    Σ = [
        (params[4])^2  params[8]  params[9];
        params[8]  (params[5])^2  params[7];
        params[9]  params[7]  (params[6])^2
    ] # Covariance matrix

    # Fix it only if needed
    Σ = make_pos_def(Σ)
    L = cholesky(Symmetric(Σ)).L

    # Parents
    Threads.@threads for i in 1:n_c
    #for i in 1:n_c
        # CRNs: fixed per-household base shocks
        z_i = @view Z[:, i]

        ν = μ .+ L * z_i                  # ν uses current μ,Σ but same z_i

        # Random draw from preference parameters
        α_0 = 1 # Normalization
        α_1 = exp(ν[1])

        if hyp[3*(i-1)+1] >= 1.0 && hyp[3*(i-1)+1] <= 5.0 # Low Self-Regulation
            # First compute base normalized values
            g = params[11]  # or params[10], whichever is correct
            D_λ_L = 1.0 + exp(ν[2]) + (1.0 + g) * exp(ν[3])
            λ_0 = exp(ν[2]) / D_λ_L
            λ_1 = (1.0 + g) * exp(ν[3]) / D_λ_L
            λ_2 = 1.0 / D_λ_L
        elseif hyp[3*(i-1)+1] >= 6.0 && hyp[3*(i-1)+1] <= 11.0  # High Self-Regulation
            λ_sum = 1 + exp(ν[2]) + exp(ν[3])
            λ_0 = exp(ν[2]) / λ_sum
            λ_1 = exp(ν[3]) / λ_sum
            λ_2 = 1 / λ_sum
        end
        
        # Store the current values of alpha and lambda parameters
        # Parents
        α_0_vec[i] = α_0
        α_1_vec[i] = α_1
        # Child
        λ_0_vec[i] = λ_0
        λ_1_vec[i] = λ_1
        λ_2_vec[i] = λ_2

        #println("λ_1 :", λ_1,  "λ_2 :", λ_2,  "λ_3 :", λ_3,  "λ_0 :", λ_0)
        
        for t = n_age:-1:1
            # Initialize
            if t == n_age
                ψcc_store[i, t] = λ_0_vec[i] * (1 - (β_c)^(t + 1)) / (1 - β_c) # ψ_{c, 4}^C
                ψpc_store[i, t] = (α_0_vec[i]*(1 - φ)*(1 - (β_p)^(t + 1)) / (1 - β_p)) + φ*ψcc_store[i, t] # ψ_{p, 4}^C
            elseif t == 1 || t == 2 # ψ_{c, 3}^C, ψ_{c, 3}^N, ψ_{p, 3}^C, ψ_{p, 3}^N, ψ_{c, 2}^C, ψ_{c, 2}^N, ψ_{p, 2}^C, ψ_{p, 2}^N
                ψcc_store[i, t] = λ_0_vec[i] + β_c*(params1c[t*4+1]*ψcc_store[i, t+1]) # ψ_{c, T}^C
                ψpc_store[i, t] = (1-φ)*α_0_vec[i] + φ*λ_0_vec[i] + β_p*(params1c[t*4+1]*ψpc_store[i, t+1]) # ψ_{p, T + 1}^C
            end
            
            ψcc, ψpc = ψcc_store[i, t], ψpc_store[i, t]
            #println("ψcc: ", ψcc, "ψcn: ", ψcn, "ψpc: ", ψpc, "ψpn: ", ψpn)
            Y_it = Y[(i-1)*n_age+t]
            s_it = s[(i-1)*n_age+t]
            CT_it = CT[(i-1)*n_age+t]
            Rbar_it = R_bar[(i-1)*n_age+t]

            χ = (1-φ)*α_1_vec[i] + φ*λ_2_vec[i] + β_p*(ψpc*params1c[t*4-1])
            c_star = (1-φ)*α_1_vec[i]*(Y_it+CT_it) / χ # Parents' Consumption
            M_star = β_p*(ψpc*params1c[t*4-1])*(Y_it+CT_it) / χ # Educational Investment Goods
            R_star = max(0.0, ((φ*λ_2_vec[i]*(Y_it+CT_it) / χ) - Rbar_it))

            C = β_p*(ψpc*params1c[t*4-2])
            A = β_c*(ψcc*params1c[t*4-2])
            #println("A: ", A)
            τ_star = A*(Time-s_it) / (λ_1_vec[i]+A) # Closed-form optimal child effort (no CPM)
            l_star = Time - s_it - τ_star
            #println("τ_star: ", τ_star , "R_star: ", R_star , "M_star: ", M_star )
            # Just before you compute Vc_star and Vp_star
            #if l_star <= 0.0
            #    println("DEBUG: l_star <= 0! t=$t, i=$i, l_star=$l_star, τ_star=$τ_star, s_it=$s_it, Time=$Time")
            #end
            #if (Rbar_it + R_star) <= 0.0
            #    println("DEBUG: Rbar+R_star <= 0! t=$t, i=$i, Rbar_it=$Rbar_it, R_star=$R_star")
            #end
            #if τ_star <= 0.0
            #    println("DEBUG: τ_star <= 0! t=$t, i=$i, τ_star=$τ_star")
            #end
            #if M_star <= 0.0
            #    println("DEBUG: M_star <= 0! t=$t, i=$i, M_star=$M_star")
            #end
            #if c_star <= 0.0
            #    println("DEBUG: c_star <= 0! t=$t, i=$i, c_star=$c_star, Y_it=$Y_it, CT_it=$CT_it, M_star=$M_star, R_star=$R_star, Rbar_it=$Rbar_it")
            #end

            Vc_star = λ_1_vec[i]*log(l_star) + λ_2_vec[i]*log(Rbar_it + R_star) +
             β_c*ψcc*(params1c[t*4-2]*log(τ_star) + params1c[t*4-1]*log(M_star))
            Vp_star = (1-φ)*α_1_vec[i]*log(c_star) + φ*λ_1_vec[i]*log(l_star) + φ*λ_2_vec[i]*log(Rbar_it + R_star) +
             β_p*ψpc*(params1c[t*4-2]*log(τ_star) + params1c[t*4-1]*log(M_star))
            τ_pstar = C*(Time-s_it) / (φ*λ_1_vec[i]+C)
            #println("τ_p: ", τ_p )
            #println("t=$t, i=$i, τ_star=$τ_star, M_star=$M_star, l_star=$l_star, Rbar+R_star=$(Rbar_it + R_star), χ=$χ, A=$A")

            τ_cpm_opt, R_cpm_opt, M_cpm_opt, Vp_cpm_val = solve_CPM_grid(
                Time, s_it, Y_it, CT_it, params, τ_pstar,
                ψcc, ψpc, α_1, λ_1_vec[i], λ_2_vec[i],
                params1c, β_p, β_c, φ, Vc_star, τ_star, R_star, M_star, Vp_star, t, Rbar_it
            )
            #println("τ_cpm_opt, R_cpm_opt, M_cpm_opt, Vp_cpm_val: ", τ_cpm_opt, R_cpm_opt, M_cpm_opt, Vp_cpm_val)

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
             # In your CPM decision section, add:
            #if i <= 5 && t <= 3  # First 5 households, all periods
            #    println("HH $i, Period $t:")
            #    println("  Vp_star: $Vp_star")
            #    println("  Vp_cpm_val: $Vp_cpm_val")
            #    println("  CPM chosen: $(Vp_cpm_val > Vp_star)")
            #    println("Period $t ψcc: $(ψcc_store[i,t]), ψcn: $(ψcn_store[i,t])")
            #end
            # In your value_func, replace the CPM decision with:
            #CPM_opt[i, t] = 0
            #τ_opt[i, t] = τ_star
            #R_opt[i, t] = R_star
            #M_opt[i, t] = M_star
        end 
    end

    return CPM_opt, τ_opt, M_opt, R_opt,
           mean(α_1_vec), std(α_1_vec), mean(λ_0_vec), std(λ_0_vec), mean(λ_1_vec), std(λ_1_vec), mean(λ_2_vec), std(λ_2_vec)
end
