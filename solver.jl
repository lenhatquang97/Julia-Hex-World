struct MDP
    γ # discount factor
    𝒮 # state space
    𝒜 # action space
    T # transition function
    R # reward function
    TR # sample transition and reward
end
MDP(γ, 𝒮, 𝒜, T, R) = MDP(γ, 𝒮, 𝒜, T, R, nothing)


#Reward
function R(s, a=missing)
	if s == (5,2)
		return -10
	elseif s == (4,1)
		return -5
	elseif s == (9,0)
		return 10
	elseif s == (7,2)
		return 3
	else
		return 0
	end
end

function T(s,a,s′)
    if R(s,a) > 0
        return 1
    end
    if R(s,a) < 0
        return -0.5
    end
    return 0
end

action = [1,2,3,4,5,6]
state = [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(-1,2),
(0,2),(1,2),(2,2),(3,2),(4,2),(5,2),(6,2),(7,2),
(8,2),(4,1),(5,0),(6,0),(7,0),(7,1),(8,1),(9,0)]

discountFactor = 0.85

function lookahead(𝒫::MDP, U, s, a)
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    return R(s,a) + γ*sum(T(s,a,s′)*U(s′) for s′ in 𝒮)
end

function lookahead(𝒫::MDP, U::Vector, s, a)
    𝒮, T, R, γ = 𝒫.𝒮, 𝒫.T, 𝒫.R, 𝒫.γ
    return R(s,a) + γ*sum(T(s,a,s′)*U[i] for (i,s′) in enumerate(𝒮))
end

struct ValueFunctionPolicy
    𝒫 # problem
    U # utility function
end

function greedy(𝒫::MDP, U, s)
    u, a = _findmax(a->lookahead(𝒫, U, s, a), 𝒫.𝒜)
    return (a=a, u=u)
end

function (π::ValueFunctionPolicy)(s)
    return greedy(π.𝒫, π.U, s).a
end
# -------------------------- Value Iteration --------------------------


function backup(𝒫::MDP, U, s)
    return maximum(lookahead(𝒫, U, s, a) for a in 𝒫.𝒜)
end


struct ValueIteration
    k_max # maximum number of iterations
end


function solve(M::ValueIteration, 𝒫::MDP)
    U = [0.0 for s in 𝒫.𝒮]
    for k = 1:M.k_max
        U = [backup(𝒫, U, s) for s in 𝒫.𝒮]
    end
    return ValueFunctionPolicy(𝒫, U)
end
M = ValueIteration(5)
P = MDP(discountFactor,state,action,T,R)
print(solve(M,P).U)