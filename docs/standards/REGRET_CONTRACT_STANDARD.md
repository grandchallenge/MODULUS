# GCL Regret Contract Standard

Status: **Normative, version 1.0.0**  
Canonical implementation: `modulus.online`  
Machine schema: `schemas/regret_contract.schema.json`

## 1. Purpose

An adaptive subsystem is not adequately specified by saying that it changes a
learning rate, routes work, or reacts to telemetry. Every adaptive GCL subsystem
MUST declare:

1. the action space and semantics;
2. the feedback available after each action;
3. the loss or bounded surrogate used to judge the action;
4. the geometry that converts feedback into the next action;
5. the comparator class against which adaptation is evaluated;
6. the regret notion reported;
7. operational constraints and rollback conditions;
8. the telemetry required to reproduce the decision sequence.

This is the temporal counterpart to a MODULUS boundary contract. A boundary
contract controls amplification across a component. A regret contract controls
cumulative decision quality through time.

## 2. Scope

The standard applies to outer-loop controllers for:

- groupwise learning-rate, momentum, beta, decay, and residual-scale control;
- KIBO prediction-informed interventions;
- AETHER agent, tool, model, and memory routing;
- SPINDLE/SPLICE operator scheduling and mixture weights;
- Tricorder alert thresholds and intervention policies;
- curriculum, data-mixture, noise-shaping, and compute allocation;
- safety/capability or primal/dual control games.

A controller that only executes a fixed, predeclared schedule is exempt. A
controller that changes behavior from observations is in scope.

## 3. Normative requirements

### RC-1: Action declaration

The contract MUST state the action space (`box`, `simplex`, `finite_experts`,
`manifold`, or `custom`) and the physical meaning of each coordinate. Actions
MUST have explicit admissible bounds or a projection/retraction rule.

### RC-2: Feedback declaration

The contract MUST state whether feedback is full-information, bandit, delayed
full-information, or delayed bandit. Delayed feedback MUST declare a delay bound
or an explicit unbounded-delay policy.

Counterfactual losses MUST be labelled as one of:

- directly observed;
- shadow-state estimate;
- local quadratic estimate;
- importance-weighted estimate;
- hypergradient estimate;
- periodic full replay.

### RC-3: Loss declaration

The controller loss MUST be bounded for the declared deployment envelope. A
multi-objective loss SHOULD expose its terms separately:

\[
\ell_t(a)=
\alpha\,\ell_t^{\mathrm{progress}}(a)
+\beta\,\ell_t^{\mathrm{boundary}}(a)
+\gamma\,\ell_t^{\mathrm{compute}}(a)
+\rho\,\ell_t^{\mathrm{risk}}(a).
\]

Hard constraints MUST NOT be hidden as arbitrarily large soft penalties. They
must be represented by action-space restrictions, projection, or an explicit
constraint monitor.

### RC-4: Geometry declaration

The contract MUST name the norm, regularizer, and projection/retraction. The
choice MUST agree with action semantics. Examples:

- Euclidean regularizer for bounded scalar controls;
- negative entropy for simplex-valued mixtures;
- tangent-space metric plus retraction for spherical/Stiefel controls;
- semantic group norms for MODULUS parameter groups.

### RC-5: Comparator and guarantee

The contract MUST select one primary comparator class:

| Comparator | Required interpretation |
|---|---|
| `fixed` | best single action or expert over the complete run |
| `path_length_bounded` | best moving action sequence with measured total variation |
| `k_switch` | best sequence with at most `K` regime changes |
| `interval_fixed` | best fixed comparator on every inspected interval |

The reported guarantee MUST match the comparator: static, dynamic, tracking, or
strongly adaptive regret.

### RC-6: Prediction containment

Predictors such as Koopman models MAY supply optimism hints. They MUST NOT
bypass the controller's action bounds or governance constraints. Runs MUST log
hint error in the controller's dual norm. The non-optimistic controller is the
required fallback baseline.

### RC-7: Telemetry

Every run MUST log at least:

- round/index and action;
- observed feedback and delay;
- learner loss and cumulative loss;
- comparator-loss provenance;
- static regret;
- the primary contract-specific regret;
- active constraints and projection events;
- predictor hint and hint error when optimism is used;
- controller state sufficient to replay the action sequence.

### RC-8: Evaluation

Every controller MUST be evaluated against:

1. the stable base system with no adaptive governor;
2. a fixed tuned schedule or fixed best expert;
3. an ordinary non-optimistic online controller;
4. the proposed adaptive controller;
5. a stress or regime-switch fixture.

Wall-clock cost, tokens/s, memory, and extra model evaluations MUST be reported.
A lower surrogate regret is insufficient if end-task utility or stability
worsens.

### RC-9: Failure and rollback

The contract MUST define rollback triggers for bound violations, non-finite
losses, telemetry gaps, excessive feedback delay, persistent hint error, and
boundary-contract failure. Rollback returns control to the declared stable base
system, not to an undefined previous state.

## 4. Reference controller classes

Version 1.0.0 provides:

- bounded Euclidean optimistic OMD for low-dimensional continuous controls;
- Hedge for full-information finite-expert selection;
- sleeping Hedge for temporary inapplicability;
- fixed-share Hedge for switching comparators;
- sleeping Exp3 for partial-feedback routing;
- exact finite-comparator static, dynamic, tracking, and interval diagnostics;
- a conservative anytime Hoeffding confidence sequence for bounded Tricorder
  observables.

These are reference primitives, not a claim that one controller is universally
optimal.

## 5. GCL deployment profiles

### KIBO

Action: groupwise step multipliers, residual scales, or intervention intensity.  
Hint: predicted boundary observable or predicted control gradient.  
Primary metric: dynamic regret against a path-length-bounded schedule.  
Containment: clip actions; fall back to zero-hint OMD when hint error breaches its
confidence-sequence threshold.

### AETHER-POL

Action: probability mass over eligible agents/tools.  
Feedback: bandit or delayed bandit by default.  
Primary metric: tracking regret with availability masks and cost-normalized loss.  
Containment: charter, jurisdiction, permission, and provenance filters are hard
eligibility constraints applied before the regret controller.

### SPINDLE/SPLICE

Action: operator/splitting choice or simplex mixture.  
Loss: task loss plus commutator, boundary-gain, churn, and compute terms.  
Primary metric: tracking regret under controlled regime switches.  
Containment: unstable operators sleep rather than receive merely large penalties.

### Tricorder

Action: alert, throttle, checkpoint, rollback, or abstain.  
Feedback: delayed full information.  
Primary metric: interval regret and false-alarm/late-alarm cost.  
Containment: time-uniform confidence sequences for repeatedly inspected bounded
observables.

### Adaptive optimizer controls

Action: low-dimensional groupwise multipliers or a small finite set of beta
configurations.  
Feedback: counterfactual provenance MUST be explicit.  
Primary metric: dynamic regret plus end-task utility.  
Containment: do not run a large bank of shadow model weights in the first build;
use shadow moment states or periodic replay.

## 6. Acceptance gate

A work package may claim `regret_contract: conformant` only when:

- its YAML/JSON instance validates against the schema;
- the controller and stable fallback are executable;
- replay from logged state reproduces actions within declared tolerance;
- required baselines and a regime-switch fixture have run;
- comparator-loss provenance is documented;
- boundary and compute costs are included in the report;
- unresolved assumptions are entered in the claim ledger.

## 7. Explicit non-claims

Conformance does not prove global neural-network convergence, optimality of the
chosen surrogate loss, sound governance, or validity outside the declared
feedback and bounded-loss assumptions. It makes the adaptive claim inspectable,
comparable, and falsifiable.
