# ==============================================================================
# Monetary Equivalence: Convert CPM Skill Gains to ΔM and ΔY
# ==============================================================================

using Printf

# ---------------------------
# Helper: ΔM-equivalence (add educational investment, keep CPM off)
# ---------------------------
function find_equiv_M(ΔZ_target::Float64, hidx, results::Dict,
                      skills_baseline::Dict, params1c, n_c, n_age, df;
                      tol=1e-3, max_iter=10000, invest_period::Int=1)
    
    low, high = 0.0, 1000.0  # Search range for ΔM (AUD/week)
    
    for iter in 1:max_iter
        delta_M = (low + high) / 2
        
        results_mod = deepcopy(results)
        M_mod = copy(results_mod[:noCPM][3])
        
        # Apply ΔM only in the investment period
        for i in hidx
            M_mod[i, invest_period] += delta_M
        end
        
        results_mod[:noCPM] = (results_mod[:noCPM][1], results_mod[:noCPM][2], M_mod, results_mod[:noCPM][4])
        
        skills_new = simulate_skills_comparative(results_mod, params1c, n_c, n_age, df)
        
        # Compute average skill gain at next period
        baseline_skills = skills_baseline[:noCPM][:cognitive][hidx, invest_period+1]
        new_skills = skills_new[:noCPM][:cognitive][hidx, invest_period+1]
        avg_gain = mean(new_skills .- baseline_skills)
        
        diff = avg_gain - ΔZ_target
        
        if abs(diff) < tol
            return delta_M
        elseif diff > 0
            high = delta_M
        else
            low = delta_M
        end
    end
    
    @warn "ΔM equivalence did not converge after $max_iter iterations"
    return (low + high) / 2
end

# ---------------------------
# Helper: ΔY-equivalence (add income in specific period)
# ---------------------------
function find_equiv_Y(ΔZ_target::Float64, hidx, results::Dict,
                      skills_baseline::Dict, params1c, params, n_c, n_age, 
                      s, Time, Y, CT, Z, R_bar, hyp, β_p, β_c, φ, df;
                      tol=1e-3, max_iter=10000, invest_period::Int=1)
    
    Y_low, Y_high = 0.0, 5000.0  # Search range for ΔY
    
    for iter in 1:max_iter
        Y_mid = (Y_low + Y_high) / 2
        
        # Modify income only for the investment period
        Y_mod = copy(Y)
        for i in hidx
            Y_mod[(i-1)*n_age + invest_period] += Y_mid
        end
        
        # Re-run value function (no CPM)
        results_new = value_func_cf_comparative(
            Z, R_bar, params1c, params, n_c, n_age, s, Time, Y_mod, CT,
            β_p, β_c, φ,
            results[:baseline][3],  # M_opt
            results[:baseline][1],  # CPM_opt
            results[:baseline][2],  # τ_opt
            results[:baseline][4]   # R_opt
        )
        
        skills_new = simulate_skills_comparative(results_new, params1c, n_c, n_age, df)
        
        # Skill gain at next period
        baseline_skills = skills_baseline[:noCPM][:cognitive][hidx, invest_period+1]
        new_skills = skills_new[:noCPM][:cognitive][hidx, invest_period+1]
        avg_gain = mean(new_skills .- baseline_skills)
        
        diff = avg_gain - ΔZ_target
        
        if abs(diff) < tol
            return Y_mid
        elseif diff > 0
            Y_high = Y_mid
        else
            Y_low = Y_mid
        end
    end
    
    @warn "ΔY equivalence did not converge after $max_iter iterations"
    return (Y_low + Y_high) / 2
end
