"""Behavioral contracts; no math-checker dependency."""
import json
import unittest
import numpy as np

class GeneratorContract(unittest.TestCase):
    def test_generator_exists(self):
        import importlib.util
        self.assertIsNotNone(importlib.util.find_spec('hypothesis_generator'))

    def test_complete_distinct_executable_hypotheses(self):
        from hypothesis_generator import TaskSpec,HypothesisGenerator,AlgorithmSpec
        hs=HypothesisGenerator(42).generate(TaskSpec('classification',4,3),1000)
        self.assertEqual(len({h.signature for h in hs}),1000)
        self.assertEqual(len({h.algorithm.structural_key for h in hs}),1000)
        for h in hs:
            r=h.to_dict()
            self.assertTrue(r['equations'] and r['pseudocode'] and r['assumptions'])
            self.assertTrue(r['falsification'] and len(r['ablations'])==2)
            self.assertEqual(set(r['acceptance_gates'].values()),{'UNVERIFIED'})
            self.assertEqual(AlgorithmSpec.from_dict(r['algorithm']),h.algorithm)
            self.assertGreaterEqual(len(h.algorithm.controller.signals),2)
        self.assertEqual(len({h.algorithm.mechanism for h in hs}),6)
        self.assertEqual({h.strategy for h in hs},{'construct','mutate','crossover'})

    def test_reproducible_and_round_trip(self):
        from hypothesis_generator import TaskSpec,HypothesisGenerator,Hypothesis
        t=TaskSpec('regression',3)
        a=HypothesisGenerator(11).generate(t,50);b=HypothesisGenerator(11).generate(t,50)
        self.assertEqual([h.to_dict() for h in a],[h.to_dict() for h in b])
        for h in a:
            self.assertEqual(Hypothesis.from_dict(json.loads(json.dumps(h.to_dict()))).to_dict(),h.to_dict())

    def test_budget_and_unknown_input_fail_closed(self):
        from hypothesis_generator import TaskSpec,HypothesisGenerator
        with self.assertRaises(ValueError):
            HypothesisGenerator().generate(TaskSpec('classification',50,10,max_parameters=10),1)
        with self.assertRaises(ValueError):
            TaskSpec.from_dict({'kind':'classification','features':3,'test_labels':[0,1]})
        with self.assertRaises(ValueError):
            TaskSpec('regression',True)

    def test_feedback_boundary_and_freeze(self):
        from hypothesis_generator import TaskSpec,HypothesisGenerator,Feedback
        g=HypothesisGenerator(1);h=g.generate(TaskSpec('classification',3),1)[0]
        with self.assertRaises(ValueError):
            g.observe(Feedback(h.signature,.9,'test','final','a'*64))
        g.reserve_holdout('sealed')
        with self.assertRaises(ValueError):
            g.observe(Feedback(h.signature,.9,'development','sealed','a'*64))
        g.observe(Feedback(h.signature,.5,'development','inner','a'*64))
        self.assertEqual(g.freeze(),g.freeze())
        with self.assertRaises(RuntimeError):
            g.observe(Feedback(h.signature,.9,'development','inner','b'*64))
        with self.assertRaises(RuntimeError):
            g.generate(TaskSpec('classification',3),1)

    def test_canonical_controllers(self):
        from hypothesis_generator import Controller
        a=Controller('signal',name='disagreement');b=Controller('signal',name='dispersion')
        self.assertEqual(Controller.make('mean',a,b),Controller.make('mean',b,a))
        self.assertEqual(Controller.make('mean',a,a),a)
        self.assertEqual(Controller.make('complement',Controller.make('complement',a)),a)
        with self.assertRaises(ValueError):
            Controller.from_dict({'op':'__import__','children':[]})

class ExecutionContract(unittest.TestCase):
    def test_loss_gradients_match_objective(self):
        from hypothesis_generator import loss_and_residual
        rng=np.random.default_rng(3)
        for kind,names in [('regression',['squared','huber','pseudo_huber','cauchy']),('classification',['cross_entropy','brier','focal'])]:
            z=rng.normal(size=(5,1 if kind=='regression' else 3))
            y=rng.normal(size=5) if kind=='regression' else np.array([0,1,2,0,2])
            for name in names:
                _,r=loss_and_residual(z,y,kind,name)
                for i,j in np.ndindex(z.shape):
                    plus=z.copy();minus=z.copy();plus[i,j]+=1e-6;minus[i,j]-=1e-6
                    lp=loss_and_residual(plus,y,kind,name)[0].sum();lm=loss_and_residual(minus,y,kind,name)[0].sum()
                    self.assertAlmostEqual((lp-lm)/2e-6,r[i,j],places=5,msg=name)

    def test_programs_execute_on_unseen_dimensions(self):
        from hypothesis_generator import TaskSpec,HypothesisGenerator,GeneratedModel,Partition
        rng=np.random.default_rng(123)
        for kind,d,k in [('regression',7,1),('classification',17,5)]:
            x=rng.normal(size=(24,d));y=rng.normal(size=24) if kind=='regression' else np.arange(24)%k
            train=Partition('train',x,y,tuple(map(str,range(24))))
            for h in HypothesisGenerator(44).generate(TaskSpec(kind,d,k),100):
                m=GeneratedModel(h,steps=4).fit(train);p=m.predict(x[:3])
                self.assertEqual(p.shape,(3,));self.assertTrue(np.isfinite(p.astype(float)).all())
                self.assertTrue(np.isfinite(m.weights_).all())
                if kind=='classification':
                    np.testing.assert_allclose(m.predict_proba(x[:3]).sum(axis=1),1.)

    def test_train_only_statistics_and_no_fit_on_test(self):
        from hypothesis_generator import TaskSpec,HypothesisGenerator,GeneratedModel,Partition
        h=HypothesisGenerator(8).generate(TaskSpec('regression',2),1)[0]
        x=np.array([[0.,1.],[1.,2.],[2.,3.],[3.,4.]]);y=np.arange(4.)
        tr=Partition('train',x,y,('a','b','c','d'));te=Partition('test',x+1000,y,('e','f','g','h'))
        with self.assertRaises(ValueError):
            GeneratedModel(h).fit(te)
        a=GeneratedModel(h,steps=10).fit(tr);before=a.weights_.copy();stats=a.center_.copy()
        a.predict(te.x)
        np.testing.assert_array_equal(a.weights_,before);np.testing.assert_array_equal(a.center_,stats)
        np.testing.assert_array_equal(a.weights_,GeneratedModel(h,steps=10).fit(tr).weights_)

    def test_group_disjoint_split(self):
        from hypothesis_generator import split_data
        x=np.arange(120.).reshape(60,2);y=np.arange(60)%2;groups=np.repeat(np.arange(20),3)
        parts=split_data(x,y,seed=2,groups=groups)
        sets=[set(p.row_ids) for p in parts]
        self.assertFalse(sets[0]&sets[1] or sets[1]&sets[2] or sets[0]&sets[2])
        gs=[set(groups[[int(i) for i in p.row_ids]]) for p in parts]
        self.assertFalse(gs[0]&gs[1] or gs[1]&gs[2] or gs[0]&gs[2])

    def test_nonfinite_and_single_class_fail(self):
        from hypothesis_generator import TaskSpec,HypothesisGenerator,GeneratedModel,Partition
        h=HypothesisGenerator().generate(TaskSpec('classification',2),1)[0]
        with self.assertRaises(ValueError):
            GeneratedModel(h).fit(Partition('train',np.ones((4,2)),np.zeros(4),('a','b','c','d')))
        with self.assertRaises(ValueError):
            Partition('train',np.array([[1,np.nan],[0,1]]),np.array([0,1]),('a','b'))


class HardeningContract(unittest.TestCase):
    def test_provenance_and_gate_forgery_rejected(self):
        from hypothesis_generator import TaskSpec,HypothesisGenerator,Hypothesis
        h=HypothesisGenerator().generate(TaskSpec('classification',3),1)[0]
        for field,value in [('seed','fabricated'),('round_no',-17),('parents',['bad']),('strategy','crossover')]:
            r=h.to_dict();r[field]=value
            with self.assertRaises(ValueError,msg=field):
                Hypothesis.from_dict(r)
        r=h.to_dict();r['acceptance_gates']['novelty']='PASS'
        with self.assertRaises(ValueError):
            Hypothesis.from_dict(r)

    def test_freeze_commits_task_metadata(self):
        from hypothesis_generator import TaskSpec,HypothesisGenerator
        a,b=HypothesisGenerator(7),HypothesisGenerator(7)
        a.generate(TaskSpec('classification',3),10);b.generate(TaskSpec('classification',5),10)
        self.assertNotEqual(a.freeze(),b.freeze())

    def test_unfitted_prediction_fails_explicitly(self):
        from hypothesis_generator import TaskSpec,HypothesisGenerator,GeneratedModel
        h=HypothesisGenerator().generate(TaskSpec('classification',2),1)[0]
        with self.assertRaises(RuntimeError):
            GeneratedModel(h).predict(np.ones((2,2)))

    def test_behavioral_aliases_do_not_count_as_new(self):
        from hypothesis_generator import Controller,TaskSpec,HypothesisGenerator
        from dataclasses import replace
        a=Controller('signal',name='dispersion');b=Controller('signal',name='energy')
        left=Controller.make('product',a,Controller.make('mean',a,b))
        right=Controller.make('mean',Controller.make('product',a,a),Controller.make('product',a,b))
        base=HypothesisGenerator().generate(TaskSpec('regression',3),1)[0].algorithm
        x,y=replace(base,controller=left),replace(base,controller=right)
        self.assertNotEqual(x.structural_key,y.structural_key)
        self.assertEqual(x.behavioral_key,y.behavioral_key)


class ChangingRequirementsContract(unittest.TestCase):
    def test_tighter_requirements_do_not_trap_mutation(self):
        from hypothesis_generator import TaskSpec,HypothesisGenerator
        g=HypothesisGenerator(0);g.generate(TaskSpec('regression',5),1)
        t=TaskSpec('regression',5,max_parameters=18,max_state_vectors=3)
        h=g.generate(t,1)[0]
        self.assertTrue(h.algorithm.fits(t))
        self.assertEqual(h.algorithm.feature_map,'linear')
        self.assertEqual(h.algorithm.preconditioner,'identity')



class SignalInfluenceContract(unittest.TestCase):
    def test_absorbed_signal_not_misrepresented(self):
        from hypothesis_generator import Controller
        a=Controller('signal',name='time');b=Controller('signal',name='energy')
        masked=Controller.make('maximum',a,Controller.make('product',a,b))
        self.assertEqual(masked.influences,frozenset({'time'}))

    def test_emitted_controllers_have_multiple_observed_influences(self):
        from hypothesis_generator import TaskSpec,HypothesisGenerator
        for h in HypothesisGenerator(42).generate(TaskSpec('classification',4,3),1000):
            self.assertGreaterEqual(len(h.algorithm.controller.influences),2)


class AdditionalBoundaryContract(unittest.TestCase):
    def test_direct_composite_controller_round_trip(self):
        from hypothesis_generator import Controller
        a=Controller('signal',name='time');b=Controller('signal',name='energy')
        for c in (Controller('mean',(a,b)),Controller('mean',(a,a)),
                  Controller('complement',(Controller('complement',(a,)),))):
            self.assertEqual(Controller.from_dict(c.to_dict()),c)

    def test_test_content_cannot_affect_metadata_only_proposals(self):
        from hypothesis_generator import TaskSpec,HypothesisGenerator,Partition
        task=TaskSpec('classification',3)
        ids1=[h.signature for h in HypothesisGenerator(82).generate(task,100)]
        rng=np.random.default_rng(82)
        test=Partition('test',rng.normal(size=(10,3)),rng.integers(0,2,10),tuple(map(str,range(10))))
        poisoned=Partition('test',test.x+1e8,1-test.y,test.row_ids)
        ids2=[h.signature for h in HypothesisGenerator(82).generate(task,100)]
        self.assertEqual(ids1,ids2)
        with self.assertRaises(TypeError):
            HypothesisGenerator().generate(poisoned,1)

    def test_duplicate_features_never_cross_partitions(self):
        from hypothesis_generator import split_data,assert_disjoint
        x=np.repeat(np.arange(30.).reshape(15,2),2,axis=0);y=np.arange(30)%2
        parts=split_data(x,y,seed=17)
        assert_disjoint(parts)
        self.assertEqual(sum(len(p.x) for p in parts),len(x))


class ImmutableAstContract(unittest.TestCase):
    def test_mutable_signal_name_rejected(self):
        from hypothesis_generator import Controller
        with self.assertRaises(ValueError):
            Controller('half',name=[])
        with self.assertRaises(ValueError):
            Controller.from_dict({'op':'half','name':[],'children':[]})


def run_demonstration(seeds=20,count=1000,execute=1000,real_data=False,stream_count=50000):
    """Engineering demonstration. Final test scores never select/generate proposals."""
    import hashlib,platform
    from time import perf_counter
    import hypothesis_generator as hg
    result={'engineering':hg.demonstrate(seeds,count),
            'environment':{'python':platform.python_version(),'numpy':np.__version__},
            'execution_protocol':'Fixed4steps on synthetic inputs; measures executability, not accuracy.'}
    rng=np.random.default_rng(888)
    start=perf_counter();failures=[];coverage=set();successful=0
    groups=[('regression',7,1),('classification',17,2),('classification',31,9)]
    for index,(kind,d,k) in enumerate(groups):
        n=30;x=rng.normal(size=(n,d));x[:,0]=0;x[0,-1]=1e3
        y=rng.normal(size=n) if kind=='regression' else np.arange(n)%k
        train=hg.Partition('train',x,y,tuple(f'synthetic-{i}' for i in range(n)))
        number=execute//3+int(index<execute%3)
        if not number: continue
        for h in hg.HypothesisGenerator(888+index).iter_generate(hg.TaskSpec(kind,d,k),number):
            try:
                restored=hg.Hypothesis.from_dict(json.loads(json.dumps(h.to_dict())))
                model=hg.GeneratedModel(restored,steps=4).fit(train)
                prediction=model.predict(x[:5])
                assert prediction.shape==(5,) and np.isfinite(prediction.astype(float)).all()
                assert np.isfinite(model.weights_).all()
                coverage.add((kind,h.algorithm.loss,h.algorithm.mechanism,h.algorithm.feature_map))
                successful+=1
            except Exception as e:
                failures.append({'id':h.signature,'error':repr(e)})
    result['execution']={'attempted':execute,'successful':successful,'failures':failures,
                         'seconds':perf_counter()-start,'distinct_kind_loss_mechanism_map_tuples':len(coverage)}
    # One long session measures scaling of the lifetime uniqueness registry.
    # No full output corpus is retained by this streaming consumer.
    import resource
    import gc
    gc.collect()
    peak_before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    streaming=hg.HypothesisGenerator(2026);stream_start=perf_counter();streamed=0
    for h in streaming.iter_generate(hg.TaskSpec('classification',17,5),stream_count):
        streamed+=1
    stream_seconds=perf_counter()-stream_start
    peak_after=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['streaming']={'emitted':streamed,'seconds':stream_seconds,
                         'per_second':streamed/stream_seconds,'registry_size':streaming.summary()['emitted'],
                         'parent_cache_size':len(streaming._parents),
                         'process_peak_RSS_before_MiB':peak_before/1024,
                         'process_peak_RSS_after_MiB':peak_after/1024,
                         'memory_scope':'Linux high-water RSS including Python/NumPy and earlier audit; not isolated per-object allocation',
                         'behavioral_aliases_rejected':streaming.behavioral_aliases_rejected}
    del streaming
    result['real_data']=[]
    if real_data:
        from sklearn.datasets import load_breast_cancer,load_wine,load_digits,load_diabetes
        from sklearn.linear_model import LogisticRegression,Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        import sklearn
        result['environment']['sklearn']=sklearn.__version__
        # This protocol and its runtime settings are fixed before reading test scores.
        # Metadata(kind/dimension/classes) are the only dataset information seen by generator.
        datasets=[('cancer',load_breast_cancer,'classification',2),
                  ('wine',load_wine,'classification',3),('digits',load_digits,'classification',10),
                  ('diabetes',load_diabetes,'regression',1)]
        for name,loader,kind,k in datasets:
            raw=loader()
            for seed in (11,29,47):
                train,dev,test=hg.split_data(raw.data,raw.target,seed=seed)
                task=hg.TaskSpec(kind,train.x.shape[1],k,goals=('ill_conditioned',))
                g=hg.HypothesisGenerator(seed);g.reserve_holdout(f'{name}-test-{seed}')
                def score(truth,prediction):
                    return float(np.mean(truth==prediction)) if kind=='classification' else -float(np.mean((truth-prediction)**2))
                best=None;all_ids=[];errors=[];fit_start=perf_counter()
                for batch in range(3):
                    for h in g.generate(task,8):
                        all_ids.append(h.signature)
                        try:
                            model=hg.GeneratedModel(h,steps=200,seed=0).fit(train)
                            value=score(dev.y,model.predict(dev.x))
                            if best is None or value>best[0]: best=(value,model,h)
                            utility=value if kind=='classification' else 1/(1-value/max(float(train.y.var()),1e-12))
                            evidence=hashlib.sha256(json.dumps([h.signature,dev.row_ids,value]).encode()).hexdigest()
                            g.observe(hg.Feedback(h.signature,utility,'development',f'{name}-development-{seed}',evidence))
                        except (ValueError,FloatingPointError) as e:
                            errors.append({'id':h.signature,'error':str(e)})
                generated_fit_seconds=perf_counter()-fit_start
                if best is None: raise RuntimeError('all generated algorithms failed; no demonstration possible')
                # Conventional comparator: small development-tuned grid; smoke comparison,
                # not a fair exhaustive state-of-the-art algorithm benchmark.
                references=[]
                for parameter in (.1,1.,10.):
                    estimator=LogisticRegression(C=parameter,max_iter=2000) if kind=='classification' else Ridge(alpha=parameter)
                    ref=make_pipeline(StandardScaler(),estimator).fit(train.x,train.y)
                    references.append((score(dev.y,ref.predict(dev.x)),ref,parameter))
                reference=max(references,key=lambda item:item[0])
                frozen=g.freeze()
                before=best[1].weights_.copy()
                generated_test=score(test.y,best[1].predict(test.x))
                reference_test=score(test.y,reference[1].predict(test.x))
                np.testing.assert_array_equal(before,best[1].weights_)
                if frozen!=g.freeze(): raise AssertionError('test evaluation changed proposal commitment')
                row={'dataset':name,'seed':seed,'kind':kind,'classes':k,'features':task.features,
                     'n_train_development_test':[len(p.x) for p in (train,dev,test)],
                     'data_digest':hashlib.sha256(raw.data.tobytes()+raw.target.tobytes()).hexdigest(),
                     'candidate_count':len(all_ids),'candidate_set_digest':hg._hash(all_ids),'frozen_digest':frozen,
                     'selected_id':best[2].signature,'selected_algorithm':best[2].algorithm.to_dict(),
                     'selected_development_score':best[0],'selected_test_score':generated_test,
                     'reference_test_score':reference_test,'reference_parameter':reference[2],
                     'metric':'accuracy' if kind=='classification' else 'negative_MSE',
                     'fit_failures':errors,'development_feedback_receipts':g.summary()['development_receipts'],
                     'generated_training_seconds':generated_fit_seconds,
                     'test_used_for_generation_or_selection':False}
                if kind=='regression':
                    variance=float(np.mean((test.y-test.y.mean())**2))
                    row['test_R2']=1+generated_test/variance
                    row['reference_test_R2']=1+reference_test/variance
                result['real_data'].append(row)
                print(f'DEMO {name} seed={seed} generated={generated_test:.6f} reference={reference_test:.6f}',flush=True)
    result['limitations']=[
        'A bounded composition grammar, not unrestricted invention across every ML family.',
        'Structural and public-probe uniqueness do not prove literature novelty or full functional inequivalence.',
        'Real-data runs demonstrate executable generated learners with sealed selection; they do not clear the ten scientific gates.',
        'Partition roles and entity IDs must be truthful; unknown external group dependence/mislabeled data cannot be detected universally.',
        'No math checker was changed, imported or used. All scientific acceptance gates remain UNVERIFIED.']
    return result


def write_demonstration_report(result,path):
    import hashlib
    from pathlib import Path
    source=Path(__file__).with_name('hypothesis_generator.py')
    test_source=Path(__file__)
    result['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    result['test_sha256']=hashlib.sha256(test_source.read_bytes()).hexdigest()
    rows=result['real_data']
    table=['|Dataset|Runs|Generated mean|Reference mean|Metric|','|---|---:|---:|---:|---|']
    for name in dict.fromkeys(r['dataset'] for r in rows):
        subset=[r for r in rows if r['dataset']==name]
        table.append(f"|{name}|{len(subset)}|{np.mean([r['selected_test_score'] for r in subset]):.5f}|{np.mean([r['reference_test_score'] for r in subset]):.5f}|{subset[0]['metric']}|")
    text=f'''# Generator-only demonstration

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

- {result['engineering']['total_generated']:,} proposals across {result['engineering']['seeds']} seeds;
  every within-run signature is structurally unique.
- Median {result['engineering']['median_per_second']:.0f} proposals/second on this machine.
- {result['execution']['successful']:,}/{result['execution']['attempted']:,} sampled proposals executed and round-tripped;
  {len(result['execution']['failures'])} execution failures.
- A separate streaming session emitted {result['streaming']['emitted']:,} unique proposals
  at {result['streaming']['per_second']:.0f}/second with only256parents cached.
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

'''+ '\n'.join(table)+'''

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
'''+json.dumps(result,indent=2,allow_nan=False)+'\n```\n'
    Path(path).write_text(text)


if __name__=='__main__':
    import sys
    if '--demonstrate' in sys.argv:
        import argparse
        p=argparse.ArgumentParser()
        p.add_argument('--demonstrate',action='store_true');p.add_argument('--real-data',action='store_true')
        p.add_argument('--seeds',type=int,default=20);p.add_argument('--count',type=int,default=1000)
        p.add_argument('--execute',type=int,default=1000);p.add_argument('--stream-count',type=int,default=50000);p.add_argument('--report')
        a=p.parse_args()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            r=run_demonstration(a.seeds,a.count,a.execute,a.real_data,a.stream_count)
        if a.report: write_demonstration_report(r,a.report)
        print(json.dumps({'generated':r['engineering']['total_generated'],
                          'median_per_second':r['engineering']['median_per_second'],
                          'execution':r['execution'],'streaming':r['streaming'],'real_data_runs':len(r['real_data'])},indent=2))
    else:
        unittest.main()
