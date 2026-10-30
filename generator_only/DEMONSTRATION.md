# Generator-only demonstration

This package contains exactly three deliverable files: one implementation,
one test/demonstration runner and this report. The math checker is untouched.

The generator produces typed ML-algorithm hypotheses with explicit objectives,
feedback-controller expressions, parameter updates, feature maps, assumptions,
ablations and falsification plans. A single safe NumPy executor runs those specs.
It composes known primitives into research candidates. It cannot certify that a
composition is a genuinely new algorithm or passes the ten scientific gates.

## Reproduce

Python3.10+ and NumPy. For the optional real-data demonstration, scikit-learn and
threadpoolctl are also needed. Recorded versions are in the results below.

```bash
python -m pip install numpy scikit-learn threadpoolctl
OPENBLAS_NUM_THREADS=1 python -m unittest discover -v
OPENBLAS_NUM_THREADS=1 python test_hypothesis_generator.py --demonstrate --real-data --report DEMONSTRATION.md
python hypothesis_generator.py generate --kind classification --features 8 --classes 3 --goal outliers --count 3
```

The last command prints complete JSONL hypothesis records to stdout. Save them
only if wanted; execution needs no generated helper modules. Restore a record
with `Hypothesis.from_dict(record)`, then
`GeneratedModel(hypothesis).fit(train_partition)` and `model.predict(X)`.
Use `split_data(X,y,seed,groups=entity_ids)` for grouped data. `TaskSpec` declares
public requirements; it must not contain test labels or test-derived statistics.

## Observed engineering results

- 20,000 proposals across 20 seeds;
  every within-run signature is structurally unique.
- Median 2861 proposals/second on this machine.
- 1,000/1,000 sampled proposals executed and round-tripped;
  0 execution failures.
- A separate streaming session emitted 50,000 unique proposals
  at 2676/second with only256parents cached.
  Lifetime registries grow with issued proposals; see process RSS measurements below.
- Binary/multiclass classification and regression; three feature-map families,
  six adaptive coupling mechanisms, seven loss kernels, typed mutation/crossover.
- All proposals retain ten UNVERIFIED scientific gate statuses. A record claiming
  PASS, forged equations or a false signature is rejected on round-trip import.

These counts measure generator structure, reproducibility and executability.
They are not counts of new discoveries or proofs. Throughput excludes optional
model training, record export and literature review. Uniqueness is within one
live generator session. No claim of near-duplicate freedom across literature or
arbitrary equivalent mathematical expressions is made.

## Leakage controls and demonstrated workflow

Proposals receive task metadata only. The fit API accepts training partitions
only; all scaling, random features, class encoding and target normalization use
those rows. The splitter keeps exact duplicate feature rows and supplied entity
groups together and rejects overlaps. Feedback accepts development receipts
only; reserved test IDs are refused. Freeze commits task metadata, provenance,
proposal identities and individual feedback utilities before final scoring.
Prediction cannot update model parameters or generator priors.

The tests exercise held-out-role rejection, unchanged training parameters during
test prediction, training-only preprocessing, malformed schemas and unsupported
constraints. These are safeguards for the recorded workflow, not an assertion
that software can detect arbitrary external data deliberately mislabeled as
training/development or unknown correlated subjects. Supply real group IDs.

## Real-data execution smoke demonstration

Fixed protocol: seeds11/29/47; group/duplicate-disjoint approximately60/20/20
partitions; three8-proposal batches; development-only feedback and selection;
200training steps, learning_rate0.08, execution_seed0; freeze then test once.
No test-driven retuning. The conventional comparator uses a three-value
training/development grid. Budgets differ, so this is not a fair state-of-the-art
benchmark or a claim of scientific superiority. A future math/novelty checker and
full candidate evaluation remain separate work.

|Dataset|Runs|Generated mean|Reference mean|Metric|
|---|---:|---:|---:|---|
|cancer|3|0.95280|0.97345|accuracy|
|wine|3|0.95238|0.96190|accuracy|
|digits|3|0.95531|0.97486|accuracy|
|diabetes|3|-3288.14117|-2940.72963|negative_MSE|

## Scope

The invention space is supervised parametric learners built from the declared
feature/loss/aggregation/controller/update grammar. It does not yet generate
arbitrary graph neural networks, transformers, reinforcement-learning algorithms
or formal proofs. Successful execution in this engineering demonstration does not
establish unrestricted creativity, robustness, generalization or novelty.

## Complete recorded results and an emitted hypothesis

The JSON retains per-seed timings/digests, selected specifications, final scores,
failures and the complete example. No separate result files are required.

```json
{
  "engineering": {
    "version": "generator-3.0",
    "scope": "metadata-only generator engineering, not scientific novelty or predictive superiority",
    "seeds": 20,
    "total_generated": 20000,
    "all_within_run_structurally_unique": true,
    "median_per_second": 2861.3991709170614,
    "rows": [
      {
        "seed": 0,
        "count": 1000,
        "unique_structures": 1000,
        "seconds": 0.33136656600072456,
        "per_second": 3017.8059665736264,
        "mechanisms": {
          "clip_memory": 200,
          "clip_step": 157,
          "variance_penalty": 154,
          "memory_step": 166,
          "dual_memory": 165,
          "penalty_step": 158
        },
        "strategies": {
          "construct": 334,
          "mutate": 333,
          "crossover": 333
        },
        "feature_maps": {
          "diagonal_quadratic": 368,
          "linear_fourier": 344,
          "linear": 288
        },
        "digest": "626adfdbe82c43e1c715ff7f382e49e8991bd311d38dc2b0517639cad54c7b7f"
      },
      {
        "seed": 1,
        "count": 1000,
        "unique_structures": 1000,
        "seconds": 0.372023992999857,
        "per_second": 2687.998674323095,
        "mechanisms": {
          "clip_memory": 157,
          "variance_penalty": 157,
          "memory_step": 191,
          "clip_step": 185,
          "penalty_step": 155,
          "dual_memory": 155
        },
        "strategies": {
          "construct": 334,
          "mutate": 333,
          "crossover": 333
        },
        "feature_maps": {
          "linear": 307,
          "linear_fourier": 370,
          "diagonal_quadratic": 323
        },
        "digest": "db0f8503e03a882b04b4388e790a452207a1f3568a1467c6fdb9df10b0236511"
      },
      {
        "seed": 2,
        "count": 1000,
        "unique_structures": 1000,
        "seconds": 0.6267333019995931,
        "per_second": 1595.5750186075948,
        "mechanisms": {
          "clip_memory": 210,
          "variance_penalty": 150,
          "penalty_step": 166,
          "memory_step": 169,
          "clip_step": 149,
          "dual_memory": 156
        },
        "strategies": {
          "construct": 334,
          "mutate": 333,
          "crossover": 333
        },
        "feature_maps": {
          "linear": 408,
          "linear_fourier": 302,
          "diagonal_quadratic": 290
        },
        "digest": "17a808a97f3eeed77cd72580d2e8a6efdf1363534f9142e03be50c8816663730"
      },
      {
        "seed": 3,
        "count": 1000,
        "unique_structures": 1000,
        "seconds": 0.42168750599921623,
        "per_second": 2371.4243030047437,
        "mechanisms": {
          "clip_memory": 225,
          "memory_step": 139,
          "clip_step": 179,
          "variance_penalty": 135,
          "dual_memory": 158,
          "penalty_step": 164
        },
        "strategies": {
          "construct": 334,
          "mutate": 333,
          "crossover": 333
        },
        "feature_maps": {
          "linear": 382,
          "linear_fourier": 313,
          "diagonal_quadratic": 305
        },
        "digest": "b27d447018186305c7308a08d83b5f991a29fc97eb242457739fce31dae750ba"
      },
      {
        "seed": 4,
        "count": 1000,
        "unique_structures": 1000,
        "seconds": 0.31782043800012616,
        "per_second": 3146.430752825289,
        "mechanisms": {
          "clip_memory": 203,
          "dual_memory": 150,
          "variance_penalty": 149,
          "memory_step": 189,
          "clip_step": 157,
          "penalty_step": 152
        },
        "strategies": {
          "construct": 334,
          "mutate": 333,
          "crossover": 333
        },
        "feature_maps": {
          "linear": 364,
          "diagonal_quadratic": 365,
          "linear_fourier": 271
        },
        "digest": "4c64a60ded7d6ae608e63c34f605614c64d483fd13910a4640167b152ea1770f"
      },
      {
        "seed": 5,
        "count": 1000,
        "unique_structures": 1000,
        "seconds": 0.3386669190003886,
        "per_second": 2952.753705474412,
        "mechanisms": {
          "clip_memory": 172,
          "variance_penalty": 154,
          "dual_memory": 164,
          "memory_step": 154,
          "clip_step": 194,
          "penalty_step": 162
        },
        "strategies": {
          "construct": 334,
          "mutate": 333,
          "crossover": 333
        },
        "feature_maps": {
          "linear_fourier": 301,
          "diagonal_quadratic": 353,
          "linear": 346
        },
        "digest": "28eb96c0d1e5c8aaafbcbcba7fe8f7c00f79c24e3a0113380fe6c0909933a0d8"
      },
      {
        "seed": 6,
        "count": 1000,
        "unique_structures": 1000,
        "seconds": 0.3811421350001183,
        "per_second": 2623.693126973982,
        "mechanisms": {
          "clip_memory": 197,
          "clip_step": 172,
          "penalty_step": 153,
          "memory_step": 169,
          "variance_penalty": 157,
          "dual_memory": 152
        },
        "strategies": {
          "construct": 334,
          "mutate": 333,
          "crossover": 333
        },
        "feature_maps": {
          "linear_fourier": 347,
          "linear": 336,
          "diagonal_quadratic": 317
        },
        "digest": "8fd5cb829602e4a861f6e1132b0d4eb7557d9abdb40e7d3eba99a14a409ad306"
      },
      {
        "seed": 7,
        "count": 1000,
        "unique_structures": 1000,
        "seconds": 0.3191488639995441,
        "per_second": 3133.3340418890803,
        "mechanisms": {
          "clip_memory": 192,
          "dual_memory": 164,
          "memory_step": 164,
          "clip_step": 158,
          "penalty_step": 165,
          "variance_penalty": 157
        },
        "strategies": {
          "construct": 334,
          "mutate": 333,
          "crossover": 333
        },
        "feature_maps": {
          "diagonal_quadratic": 360,
          "linear_fourier": 291,
          "linear": 349
        },
        "digest": "3569236b8c9320c110d1facbd4e5802779d8e42ea74dde9b260f6e93532c9b42"
      },
      {
        "seed": 8,
        "count": 1000,
        "unique_structures": 1000,
        "seconds": 0.3371966140002769,
        "per_second": 2965.628830422297,
        "mechanisms": {
          "clip_memory": 206,
          "clip_step": 178,
          "penalty_step": 151,
          "variance_penalty": 159,
          "dual_memory": 153,
          "memory_step": 153
        },
        "strategies": {
          "construct": 334,
          "mutate": 333,
          "crossover": 333
        },
        "feature_maps": {
          "linear": 362,
          "linear_fourier": 360,
          "diagonal_quadratic": 278
        },
        "digest": "47db0bd818d277347e13eff93a7453f26e58841a674b2154cafe141a6380a7c6"
      },
      {
        "seed": 9,
        "count": 1000,
        "unique_structures": 1000,
        "seconds": 0.3418653519993313,
        "per_second": 2925.128253424044,
        "mechanisms": {
          "clip_memory": 196,
          "dual_memory": 152,
          "clip_step": 152,
          "penalty_step": 155,
          "memory_step": 195,
          "variance_penalty": 150
        },
        "strategies": {
          "construct": 334,
          "mutate": 333,
          "crossover": 333
        },
        "feature_maps": {
          "diagonal_quadratic": 310,
          "linear_fourier": 254,
          "linear": 436
        },
        "digest": "90f464444519c908b25adcd833b3f18228830b31777f27aa87060c0a3dbcc288"
      },
      {
        "seed": 10,
        "count": 1000,
        "unique_structures": 1000,
        "seconds": 0.3354962989997148,
        "per_second": 2980.6588119794733,
        "mechanisms": {
          "clip_memory": 172,
          "clip_step": 173,
          "memory_step": 161,
          "dual_memory": 169,
          "variance_penalty": 162,
          "penalty_step": 163
        },
        "strategies": {
          "construct": 334,
          "mutate": 333,
          "crossover": 333
        },
        "feature_maps": {
          "linear_fourier": 386,
          "linear": 316,
          "diagonal_quadratic": 298
        },
        "digest": "6fc6852f5dfa97d58f4f36899633496b511368aca5c0366b50e8504337d4cbcb"
      },
      {
        "seed": 11,
        "count": 1000,
        "unique_structures": 1000,
        "seconds": 0.3296419850003076,
        "per_second": 3033.594158217033,
        "mechanisms": {
          "clip_memory": 173,
          "penalty_step": 186,
          "variance_penalty": 158,
          "dual_memory": 161,
          "clip_step": 158,
          "memory_step": 164
        },
        "strategies": {
          "construct": 334,
          "mutate": 333,
          "crossover": 333
        },
        "feature_maps": {
          "diagonal_quadratic": 385,
          "linear": 295,
          "linear_fourier": 320
        },
        "digest": "6983ac171d257cd79631e734cbf3fa4c2b8fd7e309bc1411eba225014a793f33"
      },
      {
        "seed": 12,
        "count": 1000,
        "unique_structures": 1000,
        "seconds": 0.3565767189993494,
        "per_second": 2804.4455701041566,
        "mechanisms": {
          "clip_memory": 176,
          "dual_memory": 152,
          "penalty_step": 188,
          "clip_step": 152,
          "memory_step": 181,
          "variance_penalty": 151
        },
        "strategies": {
          "construct": 334,
          "mutate": 333,
          "crossover": 333
        },
        "feature_maps": {
          "diagonal_quadratic": 266,
          "linear_fourier": 311,
          "linear": 423
        },
        "digest": "267bec80555d91f9352aa9de971accda5b9a15734c4893b14895df9d934994fc"
      },
      {
        "seed": 13,
        "count": 1000,
        "unique_structures": 1000,
        "seconds": 0.9036762230007298,
        "per_second": 1106.591027347626,
        "mechanisms": {
          "clip_memory": 166,
          "variance_penalty": 155,
          "clip_step": 158,
          "memory_step": 184,
          "dual_memory": 160,
          "penalty_step": 177
        },
        "strategies": {
          "construct": 334,
          "mutate": 333,
          "crossover": 333
        },
        "feature_maps": {
          "diagonal_quadratic": 381,
          "linear_fourier": 343,
          "linear": 276
        },
        "digest": "819be54d8193a75b0fd4f11ee979d98ae665a957c022b0c2c598792a50602291"
      },
      {
        "seed": 14,
        "count": 1000,
        "unique_structures": 1000,
        "seconds": 0.3718359730000884,
        "per_second": 2689.357869093968,
        "mechanisms": {
          "clip_memory": 162,
          "clip_step": 228,
          "variance_penalty": 151,
          "penalty_step": 157,
          "memory_step": 151,
          "dual_memory": 151
        },
        "strategies": {
          "construct": 334,
          "mutate": 333,
          "crossover": 333
        },
        "feature_maps": {
          "linear": 372,
          "linear_fourier": 383,
          "diagonal_quadratic": 245
        },
        "digest": "d3e7bf8aea2553876e4ec16ec1ec4a01ab8483bc181597d66b3133eefc8886b3"
      },
      {
        "seed": 15,
        "count": 1000,
        "unique_structures": 1000,
        "seconds": 0.3335489860000962,
        "per_second": 2998.06038085126,
        "mechanisms": {
          "clip_memory": 210,
          "penalty_step": 154,
          "clip_step": 156,
          "memory_step": 164,
          "dual_memory": 158,
          "variance_penalty": 158
        },
        "strategies": {
          "construct": 334,
          "mutate": 333,
          "crossover": 333
        },
        "feature_maps": {
          "linear": 323,
          "diagonal_quadratic": 314,
          "linear_fourier": 363
        },
        "digest": "9adee0c1a0b900359d0b3f7c3064173b6acc493f32a205683f945e2a3048357c"
      },
      {
        "seed": 16,
        "count": 1000,
        "unique_structures": 1000,
        "seconds": 0.34265905400025076,
        "per_second": 2918.352771729966,
        "mechanisms": {
          "clip_memory": 165,
          "variance_penalty": 156,
          "memory_step": 168,
          "clip_step": 187,
          "dual_memory": 156,
          "penalty_step": 168
        },
        "strategies": {
          "construct": 334,
          "mutate": 333,
          "crossover": 333
        },
        "feature_maps": {
          "diagonal_quadratic": 385,
          "linear": 267,
          "linear_fourier": 348
        },
        "digest": "b76e777a34a73d43c58ad13798ebb74d4f789a75c10cf8c983602f24e45b5883"
      },
      {
        "seed": 17,
        "count": 1000,
        "unique_structures": 1000,
        "seconds": 0.4124509539997234,
        "per_second": 2424.530699474805,
        "mechanisms": {
          "clip_memory": 154,
          "clip_step": 166,
          "memory_step": 156,
          "variance_penalty": 152,
          "dual_memory": 213,
          "penalty_step": 159
        },
        "strategies": {
          "construct": 334,
          "mutate": 333,
          "crossover": 333
        },
        "feature_maps": {
          "linear_fourier": 325,
          "linear": 367,
          "diagonal_quadratic": 308
        },
        "digest": "d7bd68e535cde534b102c03ddb1ccab28d5fd4743b4d584b68b480ee27b66b33"
      },
      {
        "seed": 18,
        "count": 1000,
        "unique_structures": 1000,
        "seconds": 0.3613688949999414,
        "per_second": 2767.255327828263,
        "mechanisms": {
          "clip_memory": 174,
          "penalty_step": 159,
          "clip_step": 175,
          "dual_memory": 166,
          "variance_penalty": 154,
          "memory_step": 172
        },
        "strategies": {
          "construct": 334,
          "mutate": 333,
          "crossover": 333
        },
        "feature_maps": {
          "linear": 457,
          "linear_fourier": 295,
          "diagonal_quadratic": 248
        },
        "digest": "030dce6edde7212f39b853ac3ea3669ed96139c57edc98782064f7bba3403c64"
      },
      {
        "seed": 19,
        "count": 1000,
        "unique_structures": 1000,
        "seconds": 0.3716713330004495,
        "per_second": 2690.549179370771,
        "mechanisms": {
          "clip_memory": 215,
          "memory_step": 180,
          "clip_step": 156,
          "dual_memory": 143,
          "penalty_step": 163,
          "variance_penalty": 143
        },
        "strategies": {
          "construct": 334,
          "mutate": 333,
          "crossover": 333
        },
        "feature_maps": {
          "linear_fourier": 447,
          "linear": 272,
          "diagonal_quadratic": 281
        },
        "digest": "02f1458692fafcd9f95d0e0898bce7c97358363a8773b2e475fb0a2c4d68c6ab"
      }
    ],
    "example": {
      "schema_version": "generator-3.0",
      "hid": "H-0d7758dd8d7b64b5",
      "signature": "0d7758dd8d7b64b5c97d5132de4b2d33007cc9f946fa513aaab65ea11aaa0dfe",
      "statement": "Hypothesis: for classification with ill_conditioned, outliers, the clip_memory coupling driven by q_t=complement(product(minimum(dispersion, energy), mean(dispersion, time))) will lower held-out prediction error relative to its fixed-controller ablation at the same training budget. This is an untested empirical claim; originality is unverified.",
      "algorithm": {
        "kind": "classification",
        "feature_map": "linear_fourier",
        "scaling": "standard",
        "loss": "focal",
        "mechanism": "clip_memory",
        "aggregation": "mean",
        "preconditioner": "identity",
        "controller": {
          "op": "complement",
          "name": "",
          "children": [
            {
              "op": "product",
              "name": "",
              "children": [
                {
                  "op": "minimum",
                  "name": "",
                  "children": [
                    {
                      "op": "signal",
                      "name": "dispersion",
                      "children": []
                    },
                    {
                      "op": "signal",
                      "name": "energy",
                      "children": []
                    }
                  ]
                },
                {
                  "op": "mean",
                  "name": "",
                  "children": [
                    {
                      "op": "signal",
                      "name": "dispersion",
                      "children": []
                    },
                    {
                      "op": "signal",
                      "name": "time",
                      "children": []
                    }
                  ]
                }
              ]
            }
          ]
        },
        "coupling": "coupled"
      },
      "task": {
        "kind": "classification",
        "features": 17,
        "classes": 5,
        "goals": [
          "ill_conditioned",
          "outliers"
        ],
        "max_parameters": 100000,
        "max_state_vectors": 6
      },
      "strategy": "construct",
      "parents": [],
      "seed": 19,
      "round_no": 1,
      "equations": [
        "Initialize theta, previous gradient, momentum and optional RMS arrays to zero; full-batch updates.",
        "Defaults: steps=160, learning_rate=0.08, execution_seed=0; explicit runtime overrides allowed.",
        "standard: center=training column mean, scale=training population std; scale<=1e-12 replaced by 1.",
        "robust: center=training median, scale=1.4826*training MAD; tiny MAD falls back to training std, then 1.",
        "z=clip((x-center)/scale,-10,10); linear map=z; quadratic map=[z,z^2/sqrt(2)].",
        "Fourier map=[z,cos(zR)/sqrt(m),sin(zR)/sqrt(m)]; m=min(d,32); R_ij~N(0,1/d), seeded and frozen.",
        "Regression: y_scaled=(y-training_mean_y)/training_std_y; zero std replaced by 1; invert after prediction.",
        "Classification: class encoder learned from train only; output=argmax softmax(Z@theta).",
        "p_i=softmax(logits_i); loss_i=-(1-p_i[y_i])^2*log(p_i[y_i])",
        "feature map=linear_fourier; scaling=standard, fitted on training only; append intercept 1",
        "L(theta)=mean_i focal(Z_i@theta,y_i); per-example derivative r_i=d(loss_i)/d(logits_i)",
        "g_i=outer(Z_i,r_i); g_raw=mean_i(g_i); e_i=||g_i||_F",
        "disagreement=max(mean(e_i^2)-||g_raw||_F^2,0)/(mean(e_i^2)+1e-12)",
        "dispersion=std(||r_i||)/(mean(||r_i||)+std(||r_i||)+1e-12)",
        "alignment=(1-cos(g_raw,g_raw_previous))/2, or 0.5 if either norm is zero",
        "progress=mean(loss_i)/(mean(loss_i)+previous_mean_loss+1e-12), initially previous=current",
        "time=t/max(steps-1,1); energy=||g_raw||/(1+||g_raw||)",
        "q_t=complement(product(minimum(dispersion, energy), mean(dispersion, time))); q2_t=q_t",
        "defaults: beta_t=0.9; lambda_t=0.001; c0_t=max(quantile_0.75(e_i),1e-12); c_t=c0_t",
        "base_t=learning_rate/sqrt(1+t/steps); eta_t=base_t",
        "c_t=c0_t*(0.25+1.5*(1-q_t))",
        "beta_t=0.95*(1-q2_t)",
        "w_i=min(1,c_t/(e_i+1e-12)); clipped_g_i=w_i*g_i",
        "g_t=mean(clipped_g_i); median_of_means uses up to 5 fixed seeded shuffled blocks",
        "g_t <- g_t + lambda_t*[theta_without_intercept; zero_intercept]",
        "m_fast_t=beta_t*m_fast_previous+(1-beta_t)*g_t; for dual_memory beta_t=0.5, m_slow_t=0.95*m_slow_previous+0.05*g_t",
        "m_t=m_fast_t except for the dual_memory equation above",
        "v_t=0.95*v_previous+0.05*g_t^2 when RMS is requested",
        "D_t=ones for identity, or sqrt(v_t)+1e-6 for RMS, except the variance_penalty equation",
        "direction=m_t/D_t; theta_next=theta-eta_t*direction/max(1,||direction||_F)"
      ],
      "pseudocode": [
        "Fit feature/target transforms on training rows only.",
        "Initialize weights, optimizer state and fixed shuffled blocks.",
        "For each training step: compute losses/gradients and training-only signals.",
        "Evaluate the controller; set the coupled update coefficients.",
        "Aggregate clipped gradients; update momentum/preconditioner and bounded step.",
        "Predict with the frozen fitted representation and weights."
      ],
      "assumptions": [
        "Finite supervised data and correct task metadata; no future/test statistics.",
        "Training rows contain every declared class; group dependence requires supplied group IDs.",
        "The selected representation is suitable for the task; no universal expressivity claim.",
        "No convergence, superiority, literature novelty or robustness theorem is asserted."
      ],
      "risks": [
        "Adaptive coupling may hurt optimization or generalization.",
        "Clipping/median aggregation can bias gradients; Cauchy and focal objectives need not be convex.",
        "Fixed component libraries restrict the invention space; compositions may match prior work."
      ],
      "falsification": {
        "primary_comparison": "same generated algorithm with controller fixed to 0.5",
        "second_comparison": "same algorithm with its second coupled branch fixed to 0.5",
        "reject_empirical_improvement_if": "paired, multiplicity-corrected interval fails to support positive improvement",
        "required_evaluation": "development-only selection, new sealed test data, multiple seeds, tuned strong baselines"
      },
      "ablations": [
        {
          "kind": "classification",
          "feature_map": "linear_fourier",
          "scaling": "standard",
          "loss": "focal",
          "mechanism": "clip_memory",
          "aggregation": "mean",
          "preconditioner": "identity",
          "controller": {
            "op": "half",
            "name": "",
            "children": []
          },
          "coupling": "coupled"
        },
        {
          "kind": "classification",
          "feature_map": "linear_fourier",
          "scaling": "standard",
          "loss": "focal",
          "mechanism": "clip_memory",
          "aggregation": "mean",
          "preconditioner": "identity",
          "controller": {
            "op": "complement",
            "name": "",
            "children": [
              {
                "op": "product",
                "name": "",
                "children": [
                  {
                    "op": "minimum",
                    "name": "",
                    "children": [
                      {
                        "op": "signal",
                        "name": "dispersion",
                        "children": []
                      },
                      {
                        "op": "signal",
                        "name": "energy",
                        "children": []
                      }
                    ]
                  },
                  {
                    "op": "mean",
                    "name": "",
                    "children": [
                      {
                        "op": "signal",
                        "name": "dispersion",
                        "children": []
                      },
                      {
                        "op": "signal",
                        "name": "time",
                        "children": []
                      }
                    ]
                  }
                ]
              }
            ]
          },
          "coupling": "first_only"
        }
      ],
      "acceptance_gates": {
        "novelty": "UNVERIFIED",
        "mathematical_foundation": "UNVERIFIED",
        "working_implementation": "UNVERIFIED",
        "accuracy_generalization": "UNVERIFIED",
        "robustness": "UNVERIFIED",
        "efficiency": "UNVERIFIED",
        "fair_benchmarking": "UNVERIFIED",
        "statistical_validation": "UNVERIFIED",
        "independent_reproducibility": "UNVERIFIED",
        "real_world_validation": "UNVERIFIED"
      },
      "novelty_status": "UNVERIFIED; structural uniqueness is not literature novelty",
      "resources": {
        "stored_parameters": 588,
        "persistent_optimizer_vectors": 3,
        "controller_observed_signal_influences": [
          "dispersion",
          "energy",
          "time"
        ]
      }
    }
  },
  "environment": {
    "python": "3.12.14",
    "numpy": "2.3.5",
    "sklearn": "1.8.0"
  },
  "execution_protocol": "Fixed4steps on synthetic inputs; measures executability, not accuracy.",
  "execution": {
    "attempted": 1000,
    "successful": 1000,
    "failures": [],
    "seconds": 1.970505098999638,
    "distinct_kind_loss_mechanism_map_tuples": 122
  },
  "streaming": {
    "emitted": 50000,
    "seconds": 18.682286690000183,
    "per_second": 2676.331908917918,
    "registry_size": 50000,
    "parent_cache_size": 256,
    "process_peak_RSS_before_MiB": 39.05078125,
    "process_peak_RSS_after_MiB": 54.4140625,
    "memory_scope": "Linux high-water RSS including Python/NumPy and earlier audit; not isolated per-object allocation",
    "behavioral_aliases_rejected": 566
  },
  "real_data": [
    {
      "dataset": "cancer",
      "seed": 11,
      "kind": "classification",
      "classes": 2,
      "features": 30,
      "n_train_development_test": [
        342,
        114,
        113
      ],
      "data_digest": "25e2eb7b7a8745b1bde5ba18bc4969983d2e2f403291ecd5bde4a29ef30c3859",
      "candidate_count": 24,
      "candidate_set_digest": "f943f9e80706e7c17b35da3601f2427761cd6db97659a35200331e95b4ce161d",
      "frozen_digest": "a21ab5b796dc4bd2f8ff2ab02369584a15bbed68d088c24fe6905ef798406c55",
      "selected_id": "5742abae33d366fd5435a723efffa78ae54038571450d4dfbeb01ce36ac51fcd",
      "selected_algorithm": {
        "kind": "classification",
        "feature_map": "diagonal_quadratic",
        "scaling": "robust",
        "loss": "cross_entropy",
        "mechanism": "penalty_step",
        "aggregation": "median_of_means",
        "preconditioner": "rms",
        "controller": {
          "op": "mean",
          "name": "",
          "children": [
            {
              "op": "signal",
              "name": "dispersion",
              "children": []
            },
            {
              "op": "signal",
              "name": "progress",
              "children": []
            }
          ]
        },
        "coupling": "coupled"
      },
      "selected_development_score": 0.9912280701754386,
      "selected_test_score": 0.9469026548672567,
      "reference_test_score": 0.9734513274336283,
      "reference_parameter": 0.1,
      "metric": "accuracy",
      "fit_failures": [],
      "development_feedback_receipts": 24,
      "generated_training_seconds": 1.2563312189995486,
      "test_used_for_generation_or_selection": false
    },
    {
      "dataset": "cancer",
      "seed": 29,
      "kind": "classification",
      "classes": 2,
      "features": 30,
      "n_train_development_test": [
        342,
        114,
        113
      ],
      "data_digest": "25e2eb7b7a8745b1bde5ba18bc4969983d2e2f403291ecd5bde4a29ef30c3859",
      "candidate_count": 24,
      "candidate_set_digest": "01287c1fcf503d9fc81c43a1f18984769f5101a9f45384557d03704f130c486c",
      "frozen_digest": "576c52e5797cab2d564d7b851ed0e5298f6589242dcb5db5489d651a021cdb69",
      "selected_id": "dbc48a13bb557c28d39ebf13499cfb6618d7cc085cb6e1c369c000bcda17bd30",
      "selected_algorithm": {
        "kind": "classification",
        "feature_map": "diagonal_quadratic",
        "scaling": "robust",
        "loss": "cross_entropy",
        "mechanism": "clip_step",
        "aggregation": "mean",
        "preconditioner": "rms",
        "controller": {
          "op": "product",
          "name": "",
          "children": [
            {
              "op": "minimum",
              "name": "",
              "children": [
                {
                  "op": "signal",
                  "name": "energy",
                  "children": []
                },
                {
                  "op": "signal",
                  "name": "time",
                  "children": []
                }
              ]
            },
            {
              "op": "maximum",
              "name": "",
              "children": [
                {
                  "op": "signal",
                  "name": "progress",
                  "children": []
                },
                {
                  "op": "signal",
                  "name": "time",
                  "children": []
                }
              ]
            }
          ]
        },
        "coupling": "coupled"
      },
      "selected_development_score": 0.9649122807017544,
      "selected_test_score": 0.9646017699115044,
      "reference_test_score": 0.9823008849557522,
      "reference_parameter": 0.1,
      "metric": "accuracy",
      "fit_failures": [],
      "development_feedback_receipts": 24,
      "generated_training_seconds": 1.4140155229997617,
      "test_used_for_generation_or_selection": false
    },
    {
      "dataset": "cancer",
      "seed": 47,
      "kind": "classification",
      "classes": 2,
      "features": 30,
      "n_train_development_test": [
        342,
        114,
        113
      ],
      "data_digest": "25e2eb7b7a8745b1bde5ba18bc4969983d2e2f403291ecd5bde4a29ef30c3859",
      "candidate_count": 24,
      "candidate_set_digest": "e8792deadfe9152b3ba09ba6e94358f00e19e0cd2a6924cac3afc1d4b63b21b2",
      "frozen_digest": "6dae87a247381b6c18ef749d9054fc0bc5a6a93780d936c6cf96510e85ccd04c",
      "selected_id": "c98cea10172981ff9cce437c866b6a7b835f80dfd11435740e8e13fc956cf901",
      "selected_algorithm": {
        "kind": "classification",
        "feature_map": "linear_fourier",
        "scaling": "standard",
        "loss": "brier",
        "mechanism": "memory_step",
        "aggregation": "mean",
        "preconditioner": "rms",
        "controller": {
          "op": "maximum",
          "name": "",
          "children": [
            {
              "op": "signal",
              "name": "progress",
              "children": []
            },
            {
              "op": "complement",
              "name": "",
              "children": [
                {
                  "op": "signal",
                  "name": "disagreement",
                  "children": []
                }
              ]
            }
          ]
        },
        "coupling": "coupled"
      },
      "selected_development_score": 1.0,
      "selected_test_score": 0.9469026548672567,
      "reference_test_score": 0.9646017699115044,
      "reference_parameter": 0.1,
      "metric": "accuracy",
      "fit_failures": [],
      "development_feedback_receipts": 24,
      "generated_training_seconds": 1.2645730940002977,
      "test_used_for_generation_or_selection": false
    },
    {
      "dataset": "wine",
      "seed": 11,
      "kind": "classification",
      "classes": 3,
      "features": 13,
      "n_train_development_test": [
        107,
        36,
        35
      ],
      "data_digest": "3a28849ef2366c4b9d4e382b3fa5624a66f693c04d7fdcf43c529bbb0bb338d7",
      "candidate_count": 24,
      "candidate_set_digest": "f943f9e80706e7c17b35da3601f2427761cd6db97659a35200331e95b4ce161d",
      "frozen_digest": "e847e289521552dee67dd2ccb73570d83662dd85a974c6b979f56b3101fd934d",
      "selected_id": "80e51f140edfb095271c2d05cd42f84fd0b80999857aed2f79e7dfada0cbd9dc",
      "selected_algorithm": {
        "kind": "classification",
        "feature_map": "diagonal_quadratic",
        "scaling": "robust",
        "loss": "brier",
        "mechanism": "clip_memory",
        "aggregation": "mean",
        "preconditioner": "identity",
        "controller": {
          "op": "product",
          "name": "",
          "children": [
            {
              "op": "signal",
              "name": "disagreement",
              "children": []
            },
            {
              "op": "signal",
              "name": "time",
              "children": []
            }
          ]
        },
        "coupling": "coupled"
      },
      "selected_development_score": 0.9722222222222222,
      "selected_test_score": 0.9714285714285714,
      "reference_test_score": 0.9428571428571428,
      "reference_parameter": 1.0,
      "metric": "accuracy",
      "fit_failures": [],
      "development_feedback_receipts": 24,
      "generated_training_seconds": 0.9265193000001091,
      "test_used_for_generation_or_selection": false
    },
    {
      "dataset": "wine",
      "seed": 29,
      "kind": "classification",
      "classes": 3,
      "features": 13,
      "n_train_development_test": [
        107,
        36,
        35
      ],
      "data_digest": "3a28849ef2366c4b9d4e382b3fa5624a66f693c04d7fdcf43c529bbb0bb338d7",
      "candidate_count": 24,
      "candidate_set_digest": "01287c1fcf503d9fc81c43a1f18984769f5101a9f45384557d03704f130c486c",
      "frozen_digest": "99105d870ed7dab53e119b0880b0e5d79337f1b202e074cca05f39c901a9c177",
      "selected_id": "4a902ea311fcca75fdfd24f4c9149a7a17c94e7df57f2414f7dd03861f96baee",
      "selected_algorithm": {
        "kind": "classification",
        "feature_map": "diagonal_quadratic",
        "scaling": "standard",
        "loss": "focal",
        "mechanism": "penalty_step",
        "aggregation": "median_of_means",
        "preconditioner": "rms",
        "controller": {
          "op": "mean",
          "name": "",
          "children": [
            {
              "op": "signal",
              "name": "alignment",
              "children": []
            },
            {
              "op": "maximum",
              "name": "",
              "children": [
                {
                  "op": "signal",
                  "name": "progress",
                  "children": []
                },
                {
                  "op": "product",
                  "name": "",
                  "children": [
                    {
                      "op": "signal",
                      "name": "alignment",
                      "children": []
                    },
                    {
                      "op": "signal",
                      "name": "progress",
                      "children": []
                    }
                  ]
                }
              ]
            }
          ]
        },
        "coupling": "coupled"
      },
      "selected_development_score": 0.9722222222222222,
      "selected_test_score": 0.9428571428571428,
      "reference_test_score": 0.9428571428571428,
      "reference_parameter": 1.0,
      "metric": "accuracy",
      "fit_failures": [],
      "development_feedback_receipts": 24,
      "generated_training_seconds": 0.9852396340002088,
      "test_used_for_generation_or_selection": false
    },
    {
      "dataset": "wine",
      "seed": 47,
      "kind": "classification",
      "classes": 3,
      "features": 13,
      "n_train_development_test": [
        107,
        36,
        35
      ],
      "data_digest": "3a28849ef2366c4b9d4e382b3fa5624a66f693c04d7fdcf43c529bbb0bb338d7",
      "candidate_count": 24,
      "candidate_set_digest": "e8792deadfe9152b3ba09ba6e94358f00e19e0cd2a6924cac3afc1d4b63b21b2",
      "frozen_digest": "d86aea2c5d6f1e70b35affb8f04dfa31a9aac88fca52ecb77566c48360a78083",
      "selected_id": "0e8e2c7449ae9d2caaac6541fe50bdd80fa1ae410f813cc7eea649b7c35badef",
      "selected_algorithm": {
        "kind": "classification",
        "feature_map": "linear_fourier",
        "scaling": "standard",
        "loss": "brier",
        "mechanism": "memory_step",
        "aggregation": "median_of_means",
        "preconditioner": "rms",
        "controller": {
          "op": "mean",
          "name": "",
          "children": [
            {
              "op": "signal",
              "name": "progress",
              "children": []
            },
            {
              "op": "maximum",
              "name": "",
              "children": [
                {
                  "op": "signal",
                  "name": "disagreement",
                  "children": []
                },
                {
                  "op": "signal",
                  "name": "energy",
                  "children": []
                }
              ]
            }
          ]
        },
        "coupling": "coupled"
      },
      "selected_development_score": 0.9722222222222222,
      "selected_test_score": 0.9428571428571428,
      "reference_test_score": 1.0,
      "reference_parameter": 0.1,
      "metric": "accuracy",
      "fit_failures": [],
      "development_feedback_receipts": 24,
      "generated_training_seconds": 0.9419295119996605,
      "test_used_for_generation_or_selection": false
    },
    {
      "dataset": "digits",
      "seed": 11,
      "kind": "classification",
      "classes": 10,
      "features": 64,
      "n_train_development_test": [
        1079,
        360,
        358
      ],
      "data_digest": "f6d9e39f37dc45d327f6db33428ee58970ccceabb2535a5c179de35886b70443",
      "candidate_count": 24,
      "candidate_set_digest": "f943f9e80706e7c17b35da3601f2427761cd6db97659a35200331e95b4ce161d",
      "frozen_digest": "cd5cb08a95b2605c912299586a15b6462c2ed6aebcccce07a4621e740f7c1c63",
      "selected_id": "8f09af9230a8a2f4b3e9c107a633690e22dced84821d71580440c732eaf11ebf",
      "selected_algorithm": {
        "kind": "classification",
        "feature_map": "linear",
        "scaling": "standard",
        "loss": "focal",
        "mechanism": "memory_step",
        "aggregation": "mean",
        "preconditioner": "rms",
        "controller": {
          "op": "product",
          "name": "",
          "children": [
            {
              "op": "signal",
              "name": "energy",
              "children": []
            },
            {
              "op": "minimum",
              "name": "",
              "children": [
                {
                  "op": "signal",
                  "name": "alignment",
                  "children": []
                },
                {
                  "op": "product",
                  "name": "",
                  "children": [
                    {
                      "op": "signal",
                      "name": "alignment",
                      "children": []
                    },
                    {
                      "op": "signal",
                      "name": "progress",
                      "children": []
                    }
                  ]
                }
              ]
            }
          ]
        },
        "coupling": "coupled"
      },
      "selected_development_score": 0.9638888888888889,
      "selected_test_score": 0.9553072625698324,
      "reference_test_score": 0.9776536312849162,
      "reference_parameter": 0.1,
      "metric": "accuracy",
      "fit_failures": [],
      "development_feedback_receipts": 24,
      "generated_training_seconds": 4.560993902000519,
      "test_used_for_generation_or_selection": false
    },
    {
      "dataset": "digits",
      "seed": 29,
      "kind": "classification",
      "classes": 10,
      "features": 64,
      "n_train_development_test": [
        1079,
        360,
        358
      ],
      "data_digest": "f6d9e39f37dc45d327f6db33428ee58970ccceabb2535a5c179de35886b70443",
      "candidate_count": 24,
      "candidate_set_digest": "01287c1fcf503d9fc81c43a1f18984769f5101a9f45384557d03704f130c486c",
      "frozen_digest": "05d10f0621fb889922f3a29986dd8e426be6a374647a141c8e1ec141e95b730e",
      "selected_id": "4a902ea311fcca75fdfd24f4c9149a7a17c94e7df57f2414f7dd03861f96baee",
      "selected_algorithm": {
        "kind": "classification",
        "feature_map": "diagonal_quadratic",
        "scaling": "standard",
        "loss": "focal",
        "mechanism": "penalty_step",
        "aggregation": "median_of_means",
        "preconditioner": "rms",
        "controller": {
          "op": "mean",
          "name": "",
          "children": [
            {
              "op": "signal",
              "name": "alignment",
              "children": []
            },
            {
              "op": "maximum",
              "name": "",
              "children": [
                {
                  "op": "signal",
                  "name": "progress",
                  "children": []
                },
                {
                  "op": "product",
                  "name": "",
                  "children": [
                    {
                      "op": "signal",
                      "name": "alignment",
                      "children": []
                    },
                    {
                      "op": "signal",
                      "name": "progress",
                      "children": []
                    }
                  ]
                }
              ]
            }
          ]
        },
        "coupling": "coupled"
      },
      "selected_development_score": 0.9583333333333334,
      "selected_test_score": 0.9664804469273743,
      "reference_test_score": 0.9832402234636871,
      "reference_parameter": 1.0,
      "metric": "accuracy",
      "fit_failures": [],
      "development_feedback_receipts": 24,
      "generated_training_seconds": 5.451609042999735,
      "test_used_for_generation_or_selection": false
    },
    {
      "dataset": "digits",
      "seed": 47,
      "kind": "classification",
      "classes": 10,
      "features": 64,
      "n_train_development_test": [
        1079,
        360,
        358
      ],
      "data_digest": "f6d9e39f37dc45d327f6db33428ee58970ccceabb2535a5c179de35886b70443",
      "candidate_count": 24,
      "candidate_set_digest": "e8792deadfe9152b3ba09ba6e94358f00e19e0cd2a6924cac3afc1d4b63b21b2",
      "frozen_digest": "3311de3172f789341e7f9de3872cb71e6afdee2f23079b7f80a82ee54d9fd0a9",
      "selected_id": "b5551b226667397c86ac17085be6fae499700c048dc5cde0c21cfa6af32bd4b4",
      "selected_algorithm": {
        "kind": "classification",
        "feature_map": "linear",
        "scaling": "standard",
        "loss": "brier",
        "mechanism": "memory_step",
        "aggregation": "mean",
        "preconditioner": "rms",
        "controller": {
          "op": "minimum",
          "name": "",
          "children": [
            {
              "op": "signal",
              "name": "alignment",
              "children": []
            },
            {
              "op": "minimum",
              "name": "",
              "children": [
                {
                  "op": "signal",
                  "name": "alignment",
                  "children": []
                },
                {
                  "op": "signal",
                  "name": "progress",
                  "children": []
                }
              ]
            }
          ]
        },
        "coupling": "coupled"
      },
      "selected_development_score": 0.95,
      "selected_test_score": 0.9441340782122905,
      "reference_test_score": 0.9636871508379888,
      "reference_parameter": 1.0,
      "metric": "accuracy",
      "fit_failures": [],
      "development_feedback_receipts": 24,
      "generated_training_seconds": 5.0468825739999374,
      "test_used_for_generation_or_selection": false
    },
    {
      "dataset": "diabetes",
      "seed": 11,
      "kind": "regression",
      "classes": 1,
      "features": 10,
      "n_train_development_test": [
        266,
        89,
        87
      ],
      "data_digest": "84ec38030d0ff6dfdc32f475c5633e39e7ada9dbc328d3deb29bb757eec6a331",
      "candidate_count": 24,
      "candidate_set_digest": "e03da5d37a33f23e72264f9a9a531adf220734397d8b6a10ede1da83d6748289",
      "frozen_digest": "fd6586b2061c14fe8e0dfeb88182f73eb34f433332c4e3f1776acbc82577506a",
      "selected_id": "e28c6c86ba14b02773e96b1a352a8cdaded5c1ceb4654ceecbabe4b2ebbf353f",
      "selected_algorithm": {
        "kind": "regression",
        "feature_map": "diagonal_quadratic",
        "scaling": "robust",
        "loss": "cauchy",
        "mechanism": "penalty_step",
        "aggregation": "mean",
        "preconditioner": "identity",
        "controller": {
          "op": "minimum",
          "name": "",
          "children": [
            {
              "op": "product",
              "name": "",
              "children": [
                {
                  "op": "signal",
                  "name": "alignment",
                  "children": []
                },
                {
                  "op": "signal",
                  "name": "progress",
                  "children": []
                }
              ]
            },
            {
              "op": "product",
              "name": "",
              "children": [
                {
                  "op": "signal",
                  "name": "alignment",
                  "children": []
                },
                {
                  "op": "signal",
                  "name": "time",
                  "children": []
                }
              ]
            }
          ]
        },
        "coupling": "coupled"
      },
      "selected_development_score": -2884.669020714158,
      "selected_test_score": -3165.8062688914033,
      "reference_test_score": -2551.6527602103247,
      "reference_parameter": 0.1,
      "metric": "negative_MSE",
      "fit_failures": [],
      "development_feedback_receipts": 24,
      "generated_training_seconds": 0.806971386999976,
      "test_used_for_generation_or_selection": false,
      "test_R2": 0.46077046517473685,
      "reference_test_R2": 0.5653787964082743
    },
    {
      "dataset": "diabetes",
      "seed": 29,
      "kind": "regression",
      "classes": 1,
      "features": 10,
      "n_train_development_test": [
        266,
        89,
        87
      ],
      "data_digest": "84ec38030d0ff6dfdc32f475c5633e39e7ada9dbc328d3deb29bb757eec6a331",
      "candidate_count": 24,
      "candidate_set_digest": "4dac8df91ed17eafd8bf85f06e6f45cf3f48dac63807699fe1db99516702b57e",
      "frozen_digest": "5d21881bc1958913ee1f01797588a62ab3de8fc8cbe14f5b466ae5e7e3d941e4",
      "selected_id": "9e8fa0926b12bd653dde55193bfd39dd2def3dbe2388f4cb285ad1cd6adba04f",
      "selected_algorithm": {
        "kind": "regression",
        "feature_map": "linear_fourier",
        "scaling": "standard",
        "loss": "pseudo_huber",
        "mechanism": "clip_memory",
        "aggregation": "median_of_means",
        "preconditioner": "identity",
        "controller": {
          "op": "maximum",
          "name": "",
          "children": [
            {
              "op": "complement",
              "name": "",
              "children": [
                {
                  "op": "signal",
                  "name": "alignment",
                  "children": []
                }
              ]
            },
            {
              "op": "product",
              "name": "",
              "children": [
                {
                  "op": "signal",
                  "name": "dispersion",
                  "children": []
                },
                {
                  "op": "signal",
                  "name": "energy",
                  "children": []
                }
              ]
            }
          ]
        },
        "coupling": "coupled"
      },
      "selected_development_score": -2637.7559630720298,
      "selected_test_score": -3601.000770909615,
      "reference_test_score": -3290.959099147433,
      "reference_parameter": 10.0,
      "metric": "negative_MSE",
      "fit_failures": [],
      "development_feedback_receipts": 24,
      "generated_training_seconds": 0.9278225210000528,
      "test_used_for_generation_or_selection": false,
      "test_R2": 0.3551194413067563,
      "reference_test_R2": 0.41064285249827304
    },
    {
      "dataset": "diabetes",
      "seed": 47,
      "kind": "regression",
      "classes": 1,
      "features": 10,
      "n_train_development_test": [
        266,
        89,
        87
      ],
      "data_digest": "84ec38030d0ff6dfdc32f475c5633e39e7ada9dbc328d3deb29bb757eec6a331",
      "candidate_count": 24,
      "candidate_set_digest": "799ed7f276dabce50a3844557987549fc35ca5d462e1b92a2fc66d591546c256",
      "frozen_digest": "bce5c10001e5f2130748eee7f9225c695f07fcd3e47f2866b6a643e09f4b0927",
      "selected_id": "46a6707500bab3c6c730db409c6fccb3be56cc2a9b8a77a810c38f491f640abd",
      "selected_algorithm": {
        "kind": "regression",
        "feature_map": "linear",
        "scaling": "standard",
        "loss": "pseudo_huber",
        "mechanism": "memory_step",
        "aggregation": "mean",
        "preconditioner": "rms",
        "controller": {
          "op": "minimum",
          "name": "",
          "children": [
            {
              "op": "signal",
              "name": "alignment",
              "children": []
            },
            {
              "op": "minimum",
              "name": "",
              "children": [
                {
                  "op": "signal",
                  "name": "alignment",
                  "children": []
                },
                {
                  "op": "signal",
                  "name": "progress",
                  "children": []
                }
              ]
            }
          ]
        },
        "coupling": "coupled"
      },
      "selected_development_score": -2807.11505721345,
      "selected_test_score": -3097.616457748481,
      "reference_test_score": -2979.5770274647557,
      "reference_parameter": 1.0,
      "metric": "negative_MSE",
      "fit_failures": [],
      "development_feedback_receipts": 24,
      "generated_training_seconds": 1.464115290999871,
      "test_used_for_generation_or_selection": false,
      "test_R2": 0.5021291830988117,
      "reference_test_R2": 0.5211013148599613
    }
  ],
  "limitations": [
    "A bounded composition grammar, not unrestricted invention across every ML family.",
    "Structural and public-probe uniqueness do not prove literature novelty or full functional inequivalence.",
    "Real-data runs demonstrate executable generated learners with sealed selection; they do not clear the ten scientific gates.",
    "Partition roles and entity IDs must be truthful; unknown external group dependence/mislabeled data cannot be detected universally.",
    "No math checker was changed, imported or used. All scientific acceptance gates remain UNVERIFIED."
  ],
  "source_sha256": "8ce3010e741603c9627c85f37944586c4cde2155eeb24660adcfd58ca5931b3f",
  "test_sha256": "aeddcb03716e20d91fdcde10a39429f751d3cb25df6bf728a307db734624b650"
}
```

## Independent engineering review

A separate reviewer (same environment, not external scientific replication)
inspected the implementation and demonstration protocol without modifying files.
All22tests passed in that review. Independent checks covered all7loss derivatives
(maximum finite-difference discrepancy4.1e-10) and308five-step optimizer
configurations reconstructed using explicit per-example gradient tensors
(maximum weight discrepancy1.18e-15). Forged metadata/gates, AST arity/name
validation, revised memory budgets, duplicate/entity grouping and train-only
preprocessing were checked. An independent reconstruction of wine/seed11 matched
the selected proposal identity, freeze digest and both test scores exactly.
The reviewer inspected but did not repeat the full50000proposal streaming audit.
No remaining important issue was found within that reviewed scope. This review
does not certify novelty, scientific superiority, all possible leakage paths,
or any of the ten algorithm acceptance gates.

Reviewed implementation SHA-256:
`8ce3010e741603c9627c85f37944586c4cde2155eeb24660adcfd58ca5931b3f`

Reviewed support/test SHA-256:
`aeddcb03716e20d91fdcde10a39429f751d3cb25df6bf728a307db734624b650`
