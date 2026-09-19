"""R51 constructed correctness regressions; not a formal workload study."""
import copy
import math
from unittest.mock import patch

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from src.behavior import W
from src.contracts import solve_contract_p1
from src.effort_solver import solve_a_star
from src.mechanisms import get_mechanism
from src.pair_eval import evaluate_pairs
from src.robust_payment import objective_ir_reserve, objective_ir_reserve_scalar


def witness_context(k=1.3, robust=True, zeta=.65):
    vals = dict(L=1., V=.2, q_bar=1., q_min=.01, kappa=k,
        alpha=.3, beta=.05, alpha_d=.3, beta_d=.05, kappa_d=1.,
        omega_t=0., omega_d=0., zeta_t=zeta, zeta_d=zeta,
        xi_t=.1, xi_d=.1, delta_t=.1, delta_d=.1, H_t=0., H_d=0.,
        H_lower=0., U_out=0., F_i_t=10000., communication_rate=1.,
        availability=100., deadline=100., input_size=0., output_size=0.,
        last_p=0., n_interactions=0.)
    ctx={key:np.array([val]) for key,val in vals.items()}
    ctx.update(in_cultivation=np.array([False]),p_min=.05,p_max=.8,
        D_bar=50., reinforcement_margin=0.,Theta_M=.7,delta_p_max=.1,
        design_equals_true=True,robust_enabled=robust,
        response_ratio_lower=.7,response_ratio_upper=1.3,
        response_relative_bound=0.,cost_relative_bound=0.)
    return ctx


DECISION_FIELDS = ['feasible','reason_code','a_min','a_target','lambda_required',
    'p_star','D_star','a_star_design','base_payment','decision_contract_cost',
    'decision_pair_value','decision_quality','decision_ir_ok']


@pytest.mark.parametrize('robust',[False,True])
@pytest.mark.parametrize('flag',[False,True])
def test_unknown_current_response_cannot_change_offer_or_matching(robust,flag):
    left=witness_context(.7,robust)
    right=witness_context(1.3,robust)
    left['design_equals_true']=right['design_equals_true']=flag
    a=evaluate_pairs(get_mechanism('PASI',{}),left)
    b=evaluate_pairs(get_mechanism('PASI',{}),right)
    for key in DECISION_FIELDS:
        np.testing.assert_array_equal(a[key],b[key],err_msg=key)
    assert abs(a['a_star'][0]-b['a_star'][0])>1e-3
    assert abs(a['expected_contract_cost'][0]-b['expected_contract_cost'][0])>1e-3


@pytest.mark.parametrize('method',['PASI','MOI','FR'])
def test_other_current_audit_inputs_cannot_change_decision(method):
    first=witness_context(1.)
    second=copy.deepcopy(first)
    for key,val in dict(alpha=.38,beta=.1,omega_t=.4,H_t=.9,
                        zeta_t=1.2,xi_t=.3,delta_t=.7).items():
        second[key]=np.array([val])
    a=evaluate_pairs(get_mechanism(method,{}),first)
    b=evaluate_pairs(get_mechanism(method,{}),second)
    for key in DECISION_FIELDS:
        np.testing.assert_array_equal(a[key],b[key],err_msg=key)
    # Information invariance is not an IR guarantee outside declared bounds.


def test_round50_robust_ir_counterexample_fixed():
    ev=evaluate_pairs(get_mechanism('PASI',{}),witness_context())
    assert ev['feasible'][0]
    assert ev['experienced_utility'][0]>=-1e-10
    assert ev['expected_contract_cost'][0]<=ev['decision_contract_cost'][0]+1e-10
    # Response remains unconstrained and unchanged in this interior witness.
    assert ev['a_star'][0]==pytest.approx(.39291334288452184,abs=1e-10)


@pytest.mark.parametrize('zeta',[.65,.8,1.,1.2,2.])
def test_uniform_ir_continuous_box_samples(zeta):
    rng=np.random.default_rng(51001)
    n=160
    kl=rng.uniform(.2,2,n);kh=kl+rng.uniform(.1,3,n)
    ah=rng.uniform(.01,.7,n);bh=rng.uniform(.01,.4,n)
    rho=rng.uniform(0,.5,n);L=rng.uniform(.1,2,n)
    p=rng.uniform(.02,.95,n);D=rng.uniform(0,6,n);U=rng.uniform(0,.1,n)
    lam=np.array([W(float(x),zeta) for x in p])*D;t=p*D
    b,gup=objective_ir_reserve(U,lam,t,rho,kl,kh,ah,bh,L)
    # Endpoints plus independently optimized interior true responses/costs.
    for fraction in [0.,.19,.53,.81,1.]:
        k=kl+fraction*(kh-kl)
        for cost_fraction in [.3,1.]:
            aa=ah*cost_fraction;bb=bh*cost_fraction
            rr=rho+.8*fraction
            for i in range(n):
                utility=lambda a:(lam[i]+rr[i])*(-math.expm1(-k[i]*a))-L[i]*(aa[i]*a+bb[i]*a*a)
                opt=minimize_scalar(lambda a:-utility(a),bounds=(0,1),method='bounded')
                a=max([0.,1.,float(opt.x)],key=utility)
                g=-math.expm1(-k[i]*a)
                obj=b[i]+(t[i]+rr[i])*g-L[i]*(aa[i]*a+bb[i]*a*a)
                assert obj>=U[i]-1e-9
                assert b[i]+t[i]*g<=b[i]+t[i]*gup[i]+1e-10
    s,u=objective_ir_reserve_scalar(U[0],lam[0],t[0],rho[0],kl[0],kh[0],ah[0],bh[0],L[0])
    assert s==pytest.approx(b[0]);assert u==pytest.approx(gup[0])


@pytest.mark.parametrize('zeta',[1.01,1.2,2.,8.])
@pytest.mark.parametrize('bounds',[(.05,.8),(.8,.95),(.05,.3)])
def test_prelec_interior_or_clipped_optimum(zeta,bounds):
    result=solve_contract_p1(.2,*bounds,50.,zeta)
    p=result['p_star'];D=result['D_star']
    assert result['feasible'];assert W(p,zeta)*D>=.2-1e-10
    if W(bounds[0],zeta)*50<.2:
        # Use the actual feasible lower bound, not invalid grid contracts.
        from src.behavior import W_inv
        low=max(bounds[0],W_inv(.2/50,zeta))
    else:low=bounds[0]
    alt=minimize_scalar(lambda x:x*.2/W(x,zeta),bounds=(low,bounds[1]),method='bounded')
    assert p*D<=alt.fun+1e-9


def test_vector_prelec_matches_independent_scalar():
    ctx=witness_context(1.,False,1.2)
    ctx['alpha']=ctx['alpha_d']=np.array([.1])
    ctx['beta']=ctx['beta_d']=np.array([.5])
    ev=evaluate_pairs(get_mechanism('PASI',{}),ctx)
    sc=solve_contract_p1(float(ev['lambda_required'][0]),.05,.8,50.,1.2)
    assert ev['p_star'][0]==pytest.approx(.6690626526678188,abs=1e-10)
    assert ev['p_star'][0]==pytest.approx(sc['p_star'])
    assert ev['D_star'][0]==pytest.approx(sc['D_star'])


def test_zero_incentive_does_not_force_qos_effort():
    assert solve_a_star(0.,1.,.1,.5,1.,.2)==0.
    from dataclasses import replace
    mech=replace(get_mechanism('PASI',{}),contract_mode='fixed',p_fixed=.5,
                 D_fixed=0.,enforce_target=False)
    ctx=witness_context(1.,False)
    ctx['q_min']=np.array([1-math.exp(-.2)])
    ev=evaluate_pairs(mech,ctx)
    assert ev['a_min'][0]>=.2-1e-10
    assert ev['a_star'][0]==0.
    assert ev['execution_quality'][0]<ctx['q_min'][0]


@pytest.mark.parametrize('zeta',[.8,1.,1.2,2.])
def test_nominal_full_payment_against_independent_contract_grid(zeta):
    ctx=witness_context(1.,False,zeta)
    ctx['alpha']=ctx['alpha_d']=np.array([.1])
    ctx['beta']=ctx['beta_d']=np.array([.5])
    ctx['omega_t']=ctx['omega_d']=np.array([.1])
    ctx['H_t']=ctx['H_d']=np.array([.3])
    ctx['U_out']=np.array([.02])
    ev=evaluate_pairs(get_mechanism('PASI',{}),ctx)
    assert ev['feasible'][0]
    target=ev['a_target'][0];canonical=ev['expected_contract_cost'][0]
    feasible=0
    # Independent p,D enumeration: no PASI feasibility or quoted-cost input.
    for p in np.linspace(.05,.8,25):
        for D in np.linspace(0,2,25):
            gamma=W(float(p),zeta)*D+.03
            utility=lambda a:gamma*(-math.expm1(-a))-.1*a-.5*a*a
            opt=minimize_scalar(lambda a:-utility(a),bounds=(0,1),method='bounded')
            a=max([0.,1.,float(opt.x)],key=utility)
            if a<target-1e-9:continue
            feasible+=1
            g=-math.expm1(-a);cost=.1*a+.5*a*a
            b=max(0.,.02+cost-(p*D+.03)*g)
            assert canonical<=b+p*D*g+1e-7
    assert feasible>100


@pytest.mark.parametrize('objective',['coverage_first_payment_second','lagrangian'])
def test_simulator_matching_receives_only_design_fields(objective):
    from src.datasets.synthetic import generate_synthetic_episode
    from src.simulator import Simulator
    import src.matching as matching
    from tests.test_mechanisms import _cfg
    cfg=_cfg(T=1,N=4,M=2)
    cfg['matching']['objective']=objective
    cfg['uncertainty_aware']={'enabled':True,'response_ratio_lower':.7,
                             'response_ratio_upper':1.3}
    cfg['contract']['D_bar']=50.
    data=generate_synthetic_episode(cfg,seed=51002)
    data['tasks']['kappa_design']=data['tasks']['kappa'].copy()
    capture=[]
    function=('coverage_first_matching' if objective=='coverage_first_payment_second'
              else 'lagrangian_matching')
    real=getattr(matching,function)
    decision=[]
    def evaluate(mech,ctx):
        out=evaluate_pairs(mech,ctx)
        idx=out['feasible']
        decision.append((out['decision_pair_value'][idx],out['decision_contract_cost'][idx]))
        return out
    def allocate(P,T,values,costs,**kwargs):
        np.testing.assert_allclose(values,decision[-1][0])
        np.testing.assert_allclose(costs,decision[-1][1])
        capture.append((values.copy(),costs.copy()))
        return real(P,T,values,costs,**kwargs)
    for ratio in [.7,1.3]:
        run_data={key:value.copy() for key,value in data.items()}
        run_data['tasks']['kappa']=run_data['tasks']['kappa_design']*ratio
        with patch('src.simulator.evaluate_pairs',side_effect=evaluate),patch(
                'src.simulator.'+function,side_effect=allocate):
            Simulator(cfg,run_data,method='PASI',seed=51002).run()
    assert len(capture)==2
    for a,b in zip(capture[0],capture[1]):np.testing.assert_array_equal(a,b)
    assert capture[0][1].size>0
