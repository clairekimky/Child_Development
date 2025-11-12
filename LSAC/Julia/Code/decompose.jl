# ---------------------------
# Step 1: Check if household would EVER adopt CPM with higher income
# ---------------------------
function would_adopt_with_income(
    i::Int, t::Int, params1c, params, 
    s, Time, Y_boost, CT, Z, R_bar, hyp, β_p, β_c, φ,
    results::Dict
)
    """
    Check if household i would adopt CPM in period t with income = Y_boost.
    Uses the same solve_CPM_grid as baseline estimation.
    
    Returns: (would_adopt, Vp_cpm_best, Vp_nash)
    """
    
    # Extract parameters
    μ = params[1:3]
    Σ = [
        (params[4])^2  params[8]  params[9];
        params[8]  (params[5])^2  params[7];
        params[9]  params[7]  (params[6])^2
    ]
    Σ = make_pos_def(Σ)
    L = cholesky(Symmetric(Σ)).L
    
    z_i = @view Z[:, i]
    ν = μ .+ L * z_i
    
    α_0 = 1.0
    α_1 = exp(ν[1])
    
    if hyp[3*(i-1)+1] >= 1.0 && hyp[3*(i-1)+1] <= 5.0
        λ_sum_base = 1 + exp(ν[2]) + exp(ν[3])
        λ_0_base = exp(ν[2]) / λ_sum_base
        λ_1_base = exp(ν[3]) / λ_sum_base
        λ_2_base = 1 / λ_sum_base
        
        λ_1_tilde = (1 + params[11]) * λ_1_base
        scale = 1.0 / (λ_0_base + λ_1_tilde + λ_2_base)
        
        λ_0 = λ_0_base * scale
        λ_1 = λ_1_tilde * scale
        λ_2 = λ_2_base * scale
    else
        λ_sum = 1 + exp(ν[2]) + exp(ν[3])
        λ_0 = exp(ν[2]) / λ_sum
        λ_1 = exp(ν[3]) / λ_sum
        λ_2 = 1 / λ_sum
    end
    
    Y_it = Y_boost
    s_it = s[(i-1)*n_age + t]
    Rbar_it = R_bar[(i-1)*n_age + t]
    CT_it = CT[(i-1)*n_age + t]
    
    # Compute continuation values (simplified for single period)
    if t == n_age
        ψcc = λ_0 * (1 - (β_c)^(t + 1)) / (1 - β_c)
        ψpc = (α_0*(1 - φ)*(1 - (β_p)^(t + 1)) / (1 - β_p)) + φ*ψcc
    else
        ψcc = λ_0 + β_c*(params1c[t*4+1]*1.0)
        ψpc = (1-φ)*α_0 + φ*λ_0 + β_p*(params1c[t*4+1]*1.0)
    end
    
    # Nash equilibrium
    χ = (1-φ)*α_1 + φ*λ_2 + β_p*(ψpc*params1c[t*4-1])
    c_star = (1-φ)*α_1*(Y_it+CT_it) / χ
    M_star = β_p*(ψpc*params1c[t*4-1])*(Y_it+CT_it) / χ
    R_star = max(0.0, ((φ*λ_2*(Y_it+CT_it) / χ) - Rbar_it))
    
    A = β_c*(ψcc*params1c[t*4-2])
    τ_star = A*(Time-s_it) / (λ_1+A)
    l_star = Time - s_it - τ_star
    
    Vc_star = λ_1*log(l_star) + λ_2*log(Rbar_it + R_star) +
              β_c*ψcc*(params1c[t*4-2]*log(τ_star) + params1c[t*4-1]*log(M_star))
    Vp_star = (1-φ)*α_1*log(c_star) + φ*λ_1*log(l_star) + φ*λ_2*log(Rbar_it + R_star) +
              β_p*ψpc*(params1c[t*4-2]*log(τ_star) + params1c[t*4-1]*log(M_star))
    
    τ_pstar = (β_p*(ψpc*params1c[t*4-2]))*(Time-s_it) / (φ*λ_1+(β_p*(ψpc*params1c[t*4-2])))
    
    # Try CPM with grid search (same as baseline estimation)
    τ_cpm_opt, R_cpm_opt, M_cpm_opt, Vp_cpm_val = solve_CPM_grid(
        Time, s_it, Y_it, CT_it, params, τ_pstar,
        ψcc, ψpc, α_1, λ_1, λ_2,
        params1c, β_p, β_c, φ, Vc_star, τ_star, R_star, M_star, Vp_star, t, Rbar_it
    )
    
    #would_adopt = (Vp_cpm_val > Vp_star)
    would_adopt = (Vp_cpm_val <= Vp_star) && τ_cpm_opt != τ_star
    
    return (would_adopt, Vp_cpm_val, Vp_star)
end


# ---------------------------
# Step 2: Classify households (CORRECTED)
# ---------------------------
function classify_non_adopter_improved(
    i::Int, results::Dict, params1c, params, n_c, n_age,
    s, Time, Y, CT, Z, R_bar, hyp, β_p, β_c, φ;
    high_income_test=5000.0  # Test with very high income
)
    """
    Classify household i as Type A or Type B:
    
    Type A (Constrained): Would adopt CPM with higher income
    Type B (Preference): Wouldn't adopt CPM even with high income
    
    Returns: (is_type_A, reason, periods_would_adopt)
    """
    
    periods_would_adopt = Int[]
    
    # Check each period with high income
    for t in 1:n_age
        would_adopt, Vp_cpm, Vp_nash = would_adopt_with_income(
            i, t, params1c, params,
            s, Time, high_income_test, CT, Z, R_bar, hyp, β_p, β_c, φ,
            results
        )
        
        if would_adopt
            push!(periods_would_adopt, t)
        end
    end
    
    if isempty(periods_would_adopt)
        # Wouldn't adopt even with high income → Type B
        return (false, "preference_against_CPM", periods_would_adopt)
    else
        # Would adopt with high income → Type A (constrained)
        return (true, "budget_constrained", periods_would_adopt)
    end
end


# ---------------------------
# Step 3: Find minimum income supplement for Type A
# ---------------------------
function find_minimum_income_typeA(
    i::Int, results::Dict, params1c, params, n_c, n_age,
    s, Time, Y, CT, Z, R_bar, hyp, β_p, β_c, φ;
    tol=10.0, max_iter=50
)
    """
    For a Type A household, find minimum ΔY such that they adopt CPM
    in at least one period.
    
    Returns: ΔY_min (per week)
    """
    
    baseline_income = mean(Y[(i-1)*n_age + 1 : i*n_age])
    
    ΔY_low, ΔY_high = 0.0, 800.0
    
    for iter in 1:max_iter
        ΔY_mid = (ΔY_low + ΔY_high) / 2
        
        # Test if household adopts with this income boost
        adopts_any_period = false
        
        for t in 1:n_age
            Y_test = Y[(i-1)*n_age + t] + ΔY_mid
            
            would_adopt, _, _ = would_adopt_with_income(
                i, t, params1c, params,
                s, Time, Y_test, CT, Z, R_bar, hyp, β_p, β_c, φ,
                results
            )
            
            if would_adopt
                adopts_any_period = true
                break
            end
        end
        
        if adopts_any_period
            ΔY_high = ΔY_mid
        else
            ΔY_low = ΔY_mid
        end
        
        if abs(ΔY_high - ΔY_low) < tol
            return ΔY_high
        end
    end
    
    @warn "Income supplement search didn't converge for household $i"
    return ΔY_high
end

# ==============================================================================
# Simulate Skills Under Income Support for Type A Families
# ==============================================================================

function simulate_skills_with_support(
    typeA_households::Vector{Int},
    income_supplements::Dict{Int, Float64},  # household_id => ΔY
    results_baseline::Dict,
    params1c, params, n_c, n_age, df, s, Time, Y, CT, Z, R_bar, hyp, β_p, β_c, φ
)
    """
    Simulate counterfactual where Type A families receive income support.
    
    Returns: Dict with skill trajectories
    """
    
    # Create modified income vector
    Y_supported = copy(Y)
    for i in typeA_households
        ΔY = income_supplements[i]
        for t in 1:n_age
            Y_supported[(i-1)*n_age + t] += ΔY
        end
    end
    
    # Re-solve value function with new incomes
    results_supported = value_func_cf_comparative(
        Z, R_bar, params1c, params, n_c, n_age, s, Time, Y_supported, CT,
        β_p, β_c, φ,
        results_baseline[:baseline][3],  # M_opt baseline
        results_baseline[:baseline][1],  # CPM_opt baseline  
        results_baseline[:baseline][2],  # τ_opt baseline
        results_baseline[:baseline][4]   # R_opt baseline
    )
    
    # Simulate skills
    skills_supported = simulate_skills_comparative(results_supported, params1c, n_c, n_age, df)
    
    return skills_supported
end

# ==============================================================================
# Analyze Skill Gains from Income Support
# ==============================================================================

# After you've identified Type A families and computed income supplements:
typeA_ids = [i for i in 1:n_c if is_typeA[i]]
typeB_ids = [i for i in 1:n_c if !is_typeA[i]]

# Simulate with income support
skills_with_support = simulate_skills_with_support(
    typeA_ids,
    income_supplements,  # Dict: household_id => ΔY needed
    results,
    params1c, params, n_c, n_age, df, s, Time, Y, CT, Z, R_bar, hyp, β_p, β_c, φ
)

# Compare outcomes at final period (t=4, ages 14-15)
final_period = 4

# Type A families WITHOUT support (baseline)
typeA_skills_baseline = mean([results[:baseline_skills][:cognitive][i, final_period] 
                               for i in typeA_ids])

# Type A families WITH support
typeA_skills_supported = mean([skills_with_support[:baseline][:cognitive][i, final_period] 
                                for i in typeA_ids])

# Type B families (no change, they don't get support)
typeB_skills = mean([results[:baseline_skills][:cognitive][i, final_period] 
                     for i in typeB_ids])

# Compute gains
typeA_gain = typeA_skills_supported - typeA_skills_baseline
baseline_gap = typeB_skills - typeA_skills_baseline
gap_after_support = typeB_skills - typeA_skills_supported
percent_gap_closed = 100 * (baseline_gap - gap_after_support) / baseline_gap

# Convert to SD units (using baseline SD from your sample)
baseline_sd = std(results[:baseline_skills][:cognitive][:, final_period])
typeA_gain_sd = typeA_gain / baseline_sd

println("=== Impact of Income Support on Type A Families ===")
println("Type A skill gain from support: $(round(typeA_gain, digits=4)) log points")
println("Type A skill gain (SD units): $(round(typeA_gain_sd, digits=3)) SD")
println("Type A-B gap without support: $(round(baseline_gap, digits=4)) log points")
println("Type A-B gap with support: $(round(gap_after_support, digits=4)) log points")
println("Percent of gap closed: $(round(percent_gap_closed, digits=1))%")
