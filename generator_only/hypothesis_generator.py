"""Executable ML-algorithm hypothesis generator; no math checker.

One implementation file. Runtime dependency: NumPy. CLI examples:
  python hypothesis_generator.py generate --kind classification --features 8 --count 3
  python hypothesis_generator.py demonstrate --seeds 20 --count 1000

Outputs are research proposals, never certified novel algorithms. The grammar
composes bounded feedback controllers with model/loss/optimizer mechanisms.
Structural uniqueness does not establish mathematical or literature novelty.
The generator accepts metadata and development feedback only, not datasets.
"""
from __future__ import annotations
import argparse
from collections import Counter, deque
from dataclasses import asdict, dataclass, replace
from functools import cached_property
import hashlib
import json
import math
import random
from time import perf_counter
from typing import Iterator
import numpy as np

VERSION = 'generator-3.0'
GATES = ('novelty','mathematical_foundation','working_implementation',
         'accuracy_generalization','robustness','efficiency','fair_benchmarking',
         'statistical_validation','independent_reproducibility','real_world_validation')
SIGNALS = ('disagreement','dispersion','alignment','progress','time','energy')
MECHANISMS = ('clip_memory','clip_step','memory_step','dual_memory',
              'penalty_step','variance_penalty')
MAPS = ('linear','diagonal_quadratic','linear_fourier')
LOSSES = {'regression':('squared','huber','pseudo_huber','cauchy'),
          'classification':('cross_entropy','brier','focal')}
# Public synthetic construction probes, never observations from an experiment.
_PROBES=np.vstack((np.zeros((1,6)),np.ones((1,6)),np.eye(6),np.random.default_rng(112358).random((128,6))))
_PROBES.setflags(write=False)

GOALS = ('outliers','target_noise','ill_conditioned','class_imbalance',
         'high_dimension','distribution_shift')


def _json(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def _hash(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _strict(record, allowed):
    if not isinstance(record,dict) or set(record)-set(allowed):
        raise ValueError('unknown fields or invalid record')


@dataclass(frozen=True)
class TaskSpec:
    """Public problem requirements, not data-derived test-set diagnostics.

    max_parameters bounds stored numeric model parameters; max_state_vectors
    bounds persistent optimizer arrays per coefficient, excluding work buffers.
    """
    kind: str
    features: int
    classes: int | None = None
    goals: tuple[str,...] = ()
    max_parameters: int = 100000
    max_state_vectors: int = 6

    def __post_init__(self):
        if not isinstance(self.kind,str) or self.kind not in LOSSES:
            raise ValueError('supported tasks: regression or classification')
        if self.classes is None:
            object.__setattr__(self,'classes',1 if self.kind=='regression' else 2)
        for key,low,high in (('features',1,4096),('classes',1,256),
                             ('max_parameters',1,100000000),('max_state_vectors',3,32)):
            value=getattr(self,key)
            if type(value) is not int or not low<=value<=high:
                raise ValueError(f'{key} must be an integer in [{low},{high}]')
        if (self.kind=='regression' and self.classes!=1) or (self.kind=='classification' and self.classes<2):
            raise ValueError('class count incompatible with task')
        if not isinstance(self.goals,(tuple,list)) or any(g not in GOALS for g in self.goals):
            raise ValueError('unsupported research goal')
        if self.kind=='regression' and 'class_imbalance' in self.goals:
            raise ValueError('class imbalance requires classification')
        object.__setattr__(self,'goals',tuple(sorted(set(self.goals))))

    @classmethod
    def from_dict(cls,d):
        _strict(d,cls.__dataclass_fields__)
        try:
            return cls(**d)
        except TypeError as e:
            raise ValueError('invalid task record') from e


@dataclass(frozen=True)
class Controller:
    """Bounded [0,1] expression AST. Operators have no access to held-out data."""
    op: str
    children: tuple[Controller,...] = ()
    name: str = ''

    def __post_init__(self):
        if not isinstance(self.children,tuple) or any(not isinstance(x,Controller) for x in self.children):
            raise ValueError('controller children must be immutable AST nodes')
        arity={'signal':0,'half':0,'complement':1,'mean':2,'product':2,'minimum':2,'maximum':2}
        if not isinstance(self.op,str) or self.op not in arity or len(self.children)!=arity[self.op]:
            raise ValueError('unsupported controller operator or arity')
        if not isinstance(self.name,str): raise ValueError('immutable string signal name required')
        if (self.op=='signal' and self.name not in SIGNALS) or (self.op!='signal' and self.name):
            raise ValueError('invalid controller signal')
        # Canonical direct construction as well as factory/JSON construction.
        replacement=None
        if self.op=='complement' and self.children[0].op=='complement':
            replacement=self.children[0].children[0]
        elif self.op in ('mean','product','minimum','maximum'):
            children=tuple(sorted(self.children,key=lambda c:c.key))
            object.__setattr__(self,'children',children)
            if children[0]==children[1] and self.op in ('mean','minimum','maximum'):
                replacement=children[0]
        if replacement is not None:
            object.__setattr__(self,'op',replacement.op)
            object.__setattr__(self,'children',replacement.children)
            object.__setattr__(self,'name',replacement.name)
        if self.depth>4 or self.nodes>31:
            raise ValueError('controller exceeds depth/node budget')

    @property
    def depth(self):
        return 0 if not self.children else 1+max(c.depth for c in self.children)

    @property
    def nodes(self):
        return 1+sum(c.nodes for c in self.children)

    @cached_property
    def signals(self):
        return frozenset((self.name,)) if self.op=='signal' else frozenset().union(*(c.signals for c in self.children))

    @cached_property
    def influences(self):
        # Witness that changing one signal with the others fixed changes output.
        # Public construction inputs, not experimental test/validation examples.
        found=set()
        base=_PROBES[8:40]
        for name in self.signals:
            col=SIGNALS.index(name);low=base.copy();high=base.copy()
            low[:,col]=0.;high[:,col]=1.
            if np.max(np.abs(self.evaluate_batch(low)-self.evaluate_batch(high)))>1e-9:
                found.add(name)
        return frozenset(found)

    def to_dict(self):
        return {'op':self.op,'name':self.name,'children':[c.to_dict() for c in self.children]}

    @cached_property
    def key(self):
        return _json(self.to_dict())

    def evaluate_batch(self,values):
        if self.op=='signal': return values[:,SIGNALS.index(self.name)]
        if self.op=='half': return np.full(len(values),.5)
        v=[c.evaluate_batch(values) for c in self.children]
        if self.op=='complement': return 1-v[0]
        if self.op=='mean': return (v[0]+v[1])/2
        if self.op=='product': return v[0]*v[1]
        if self.op=='minimum': return np.minimum(*v)
        if self.op=='maximum': return np.maximum(*v)
        raise ValueError('invalid operator')

    @classmethod
    def make(cls,op,*children):
        if op=='complement' and len(children)==1 and children[0].op=='complement':
            return children[0].children[0]
        if op in ('mean','product','minimum','maximum') and len(children)==2:
            a,b=sorted(children,key=lambda c:c.key)
            if a==b and op in ('mean','minimum','maximum'):
                return a
            children=(a,b)
        return cls(op,tuple(children))

    @classmethod
    def from_dict(cls,d,_depth=0):
        if _depth>4:
            raise ValueError('controller nesting exceeds budget')
        _strict(d,('op','name','children'))
        if not isinstance(d.get('children',[]),list):
            raise ValueError('controller children must be a list')
        arity={'signal':0,'half':0,'complement':1,'mean':2,'product':2,'minimum':2,'maximum':2}
        op=d.get('op')
        if not isinstance(op,str) or op not in arity or len(d.get('children',[]))!=arity[op]:
            raise ValueError('unsupported controller operator or arity')
        ch=tuple(cls.from_dict(c,_depth+1) for c in d.get('children',[]))
        if d.get('op') in ('signal','half'):
            return cls(d['op'],ch,d.get('name',''))
        if d.get('name',''):
            raise ValueError('only signal nodes have names')
        return cls.make(d.get('op'),*ch)

    def evaluate(self,signals):
        if self.op=='signal':
            return float(signals[self.name])
        if self.op=='half':
            return .5
        v=[c.evaluate(signals) for c in self.children]
        if self.op=='complement': return 1-v[0]
        if self.op=='mean': return (v[0]+v[1])/2
        if self.op=='product': return v[0]*v[1]
        if self.op=='minimum': return min(v)
        if self.op=='maximum': return max(v)
        raise ValueError('invalid operator')

    def text(self):
        if self.op=='signal': return self.name
        if self.op=='half': return '0.5'
        return f'{self.op}('+', '.join(c.text() for c in self.children)+')'


@dataclass(frozen=True)
class AlgorithmSpec:
    kind: str
    feature_map: str
    scaling: str
    loss: str
    mechanism: str
    aggregation: str
    preconditioner: str
    controller: Controller
    coupling: str = 'coupled'

    def __post_init__(self):
        choices={'kind':tuple(LOSSES),'feature_map':MAPS,'scaling':('standard','robust'),
                 'mechanism':MECHANISMS,'aggregation':('mean','median_of_means'),
                 'preconditioner':('identity','rms'),'coupling':('coupled','first_only')}
        if any(getattr(self,key) not in values for key,values in choices.items()):
            raise ValueError('unsupported algorithm component')
        if self.loss not in LOSSES[self.kind] or not isinstance(self.controller,Controller):
            raise ValueError('loss/controller incompatible with algorithm')
        if self.mechanism=='variance_penalty' and self.preconditioner!='rms':
            raise ValueError('variance-penalty coupling requires RMS state')

    def to_dict(self):
        d={k:v for k,v in self.__dict__.items() if k in self.__dataclass_fields__}
        d['controller']=self.controller.to_dict()
        return d

    @classmethod
    def from_dict(cls,d):
        _strict(d,cls.__dataclass_fields__)
        try:
            return cls(**{**d,'controller':Controller.from_dict(d['controller'])})
        except (KeyError,TypeError) as e:
            raise ValueError('invalid algorithm record') from e

    @cached_property
    def structural_key(self):
        return _hash(self.to_dict())

    @cached_property
    def behavioral_key(self):
        # Conservative finite-probe alias rejection. This is not a semantic
        # equivalence decision procedure and makes no literature novelty claim.
        body=self.to_dict()
        values=np.round(self.controller.evaluate_batch(_PROBES),12)
        values[values==0]=0.
        body['controller']=values.tolist()
        return _hash(body)

    @property
    def state_vectors(self):
        # W, previous gradient, fast momentum; optional slow momentum and RMS.
        return 3+int(self.mechanism=='dual_memory')+int(self.preconditioner=='rms')

    def representation_size(self,d):
        return d if self.feature_map=='linear' else 2*d if self.feature_map=='diagonal_quadratic' else d+2*min(d,32)

    def stored_parameters(self,task):
        fixed=2*task.features
        if self.feature_map=='linear_fourier': fixed+=task.features*min(task.features,32)
        # Target normalization scalars or class-label metadata are counted conservatively.
        return (self.representation_size(task.features)+1)*task.classes+fixed+max(2,task.classes)

    def fits(self,task):
        return self.kind==task.kind and self.stored_parameters(task)<=task.max_parameters and self.state_vectors<=task.max_state_vectors


MECHANISM_EQUATIONS={
    'clip_memory':('c_t=c0_t*(0.25+1.5*(1-q_t))','beta_t=0.95*(1-q2_t)'),
    'clip_step':('c_t=c0_t*(0.25+1.5*(1-q_t))','eta_t=base_t*(0.25+0.75*q2_t)'),
    'memory_step':('beta_t=0.95*q_t','eta_t=base_t*(1-0.75*q2_t)'),
    'dual_memory':('m_t=(1-q_t)*m_fast_t+q_t*m_slow_t','eta_t=base_t*(1-0.5*q2_t)'),
    'penalty_step':('lambda_t=0.001*(0.25+1.5*q_t)','eta_t=base_t/(1+q2_t)'),
    'variance_penalty':('lambda_t=0.001*(0.25+1.5*q_t)',
                         'D_t=(1-q2_t)*ones+q2_t*(sqrt(v_t)+1e-6)'),
}


@dataclass(frozen=True)
class Hypothesis:
    algorithm: AlgorithmSpec
    task: TaskSpec
    strategy: str
    parents: tuple[str,...]
    seed: int
    round_no: int

    def __post_init__(self):
        if not isinstance(self.algorithm,AlgorithmSpec) or not isinstance(self.task,TaskSpec):
            raise ValueError('typed algorithm and task required')
        if type(self.seed) is not int or type(self.round_no) is not int or self.round_no<1:
            raise ValueError('integer seed and positive generation index required')
        if not self.algorithm.fits(self.task):
            raise ValueError('algorithm exceeds requested capability/budget')
        if self.strategy not in ('construct','mutate','crossover') or not isinstance(self.parents,tuple):
            raise ValueError('invalid generation provenance')
        arities={'construct':0,'mutate':1,'crossover':2}
        if len(self.parents)!=arities[self.strategy] or any(not isinstance(p,str) or len(p)!=64 or any(c not in '0123456789abcdef' for c in p) for p in self.parents):
            raise ValueError('strategy-specific SHA-256 parent IDs required')

    @property
    def signature(self): return self.algorithm.structural_key

    @property
    def hid(self): return 'H-'+self.signature[:16]

    @property
    def statement(self):
        goals=', '.join(self.task.goals) or 'ordinary supervised data'
        metric='held-out prediction error' if self.task.kind=='classification' else 'held-out mean squared error'
        return (f'Hypothesis: for {self.task.kind} with {goals}, the {self.algorithm.mechanism} '
                f'coupling driven by q_t={self.algorithm.controller.text()} will lower {metric} '
                'relative to its fixed-controller ablation at the same training budget. '
                'This is an untested empirical claim; originality is unverified.')

    @property
    def components(self):
        return tuple((k,str(v)) for k,v in self.algorithm.to_dict().items() if k!='controller')

    def equations(self):
        loss_defs={
            'squared':'loss_i=0.5*(prediction_i-y_scaled_i)^2',
            'huber':'r=prediction_i-y_scaled_i; loss_i=0.5*r^2 if |r|<=1 else |r|-0.5',
            'pseudo_huber':'r=prediction_i-y_scaled_i; loss_i=sqrt(1+r^2)-1',
            'cauchy':'r=prediction_i-y_scaled_i; loss_i=0.5*log(1+r^2)',
            'cross_entropy':'p_i=softmax(logits_i); loss_i=-log(p_i[y_i])',
            'brier':'p_i=softmax(logits_i); loss_i=0.5*sum_k(p_ik-onehot(y_i)_k)^2',
            'focal':'p_i=softmax(logits_i); loss_i=-(1-p_i[y_i])^2*log(p_i[y_i])'}
        return [
            'Initialize theta, previous gradient, momentum and optional RMS arrays to zero; full-batch updates.',
            'Defaults: steps=160, learning_rate=0.08, execution_seed=0; explicit runtime overrides allowed.',
            'standard: center=training column mean, scale=training population std; scale<=1e-12 replaced by 1.',
            'robust: center=training median, scale=1.4826*training MAD; tiny MAD falls back to training std, then 1.',
            'z=clip((x-center)/scale,-10,10); linear map=z; quadratic map=[z,z^2/sqrt(2)].',
            'Fourier map=[z,cos(zR)/sqrt(m),sin(zR)/sqrt(m)]; m=min(d,32); R_ij~N(0,1/d), seeded and frozen.',
            'Regression: y_scaled=(y-training_mean_y)/training_std_y; zero std replaced by 1; invert after prediction.',
            'Classification: class encoder learned from train only; output=argmax softmax(Z@theta).',
            loss_defs[self.algorithm.loss],
            f'feature map={self.algorithm.feature_map}; scaling={self.algorithm.scaling}, fitted on training only; append intercept 1',
            f'L(theta)=mean_i {self.algorithm.loss}(Z_i@theta,y_i); per-example derivative r_i=d(loss_i)/d(logits_i)',
            'g_i=outer(Z_i,r_i); g_raw=mean_i(g_i); e_i=||g_i||_F',
            'disagreement=max(mean(e_i^2)-||g_raw||_F^2,0)/(mean(e_i^2)+1e-12)',
            'dispersion=std(||r_i||)/(mean(||r_i||)+std(||r_i||)+1e-12)',
            'alignment=(1-cos(g_raw,g_raw_previous))/2, or 0.5 if either norm is zero',
            'progress=mean(loss_i)/(mean(loss_i)+previous_mean_loss+1e-12), initially previous=current',
            'time=t/max(steps-1,1); energy=||g_raw||/(1+||g_raw||)',
            f'q_t={self.algorithm.controller.text()}; q2_t='+('q_t' if self.algorithm.coupling=='coupled' else '0.5'),
            'defaults: beta_t=0.9; lambda_t=0.001; c0_t=max(quantile_0.75(e_i),1e-12); c_t=c0_t',
            'base_t=learning_rate/sqrt(1+t/steps); eta_t=base_t',
            *MECHANISM_EQUATIONS[self.algorithm.mechanism],
            'w_i=min(1,c_t/(e_i+1e-12)); clipped_g_i=w_i*g_i',
            f'g_t={self.algorithm.aggregation}(clipped_g_i); median_of_means uses up to 5 fixed seeded shuffled blocks',
            'g_t <- g_t + lambda_t*[theta_without_intercept; zero_intercept]',
            'm_fast_t=beta_t*m_fast_previous+(1-beta_t)*g_t; for dual_memory beta_t=0.5, m_slow_t=0.95*m_slow_previous+0.05*g_t',
            'm_t=m_fast_t except for the dual_memory equation above',
            'v_t=0.95*v_previous+0.05*g_t^2 when RMS is requested',
            'D_t=ones for identity, or sqrt(v_t)+1e-6 for RMS, except the variance_penalty equation',
            'direction=m_t/D_t; theta_next=theta-eta_t*direction/max(1,||direction||_F)',
        ]

    def to_dict(self):
        a=self.algorithm
        return {'schema_version':VERSION,'hid':self.hid,'signature':self.signature,
                'statement':self.statement,'algorithm':a.to_dict(),'task':asdict(self.task),
                'strategy':self.strategy,'parents':list(self.parents),'seed':self.seed,'round_no':self.round_no,
                'equations':self.equations(),
                'pseudocode':['Fit feature/target transforms on training rows only.',
                    'Initialize weights, optimizer state and fixed shuffled blocks.',
                    'For each training step: compute losses/gradients and training-only signals.',
                    'Evaluate the controller; set the coupled update coefficients.',
                    'Aggregate clipped gradients; update momentum/preconditioner and bounded step.',
                    'Predict with the frozen fitted representation and weights.'],
                'assumptions':['Finite supervised data and correct task metadata; no future/test statistics.',
                    'Training rows contain every declared class; group dependence requires supplied group IDs.',
                    'The selected representation is suitable for the task; no universal expressivity claim.',
                    'No convergence, superiority, literature novelty or robustness theorem is asserted.'],
                'risks':['Adaptive coupling may hurt optimization or generalization.',
                    'Clipping/median aggregation can bias gradients; Cauchy and focal objectives need not be convex.',
                    'Fixed component libraries restrict the invention space; compositions may match prior work.'],
                'falsification':{'primary_comparison':'same generated algorithm with controller fixed to 0.5',
                    'second_comparison':'same algorithm with its second coupled branch fixed to 0.5',
                    'reject_empirical_improvement_if':'paired, multiplicity-corrected interval fails to support positive improvement',
                    'required_evaluation':'development-only selection, new sealed test data, multiple seeds, tuned strong baselines'},
                'ablations':[replace(a,controller=Controller('half')).to_dict(),
                             replace(a,coupling='first_only').to_dict()],
                'acceptance_gates':{k:'UNVERIFIED' for k in GATES},
                'novelty_status':'UNVERIFIED; structural uniqueness is not literature novelty',
                'resources':{'stored_parameters':a.stored_parameters(self.task),
                             'persistent_optimizer_vectors':a.state_vectors,
                             'controller_observed_signal_influences':sorted(a.controller.influences)}}

    @classmethod
    def from_dict(cls,d):
        # Only canonical regeneration is accepted: do not trust a forged statement/status.
        allowed=('schema_version','hid','signature','statement','algorithm','task','strategy','parents',
                 'seed','round_no','equations','pseudocode','assumptions','risks','falsification',
                 'ablations','acceptance_gates','novelty_status','resources')
        _strict(d,allowed)
        if d.get('schema_version')!=VERSION:
            raise ValueError('unsupported hypothesis schema')
        try:
            h=cls(AlgorithmSpec.from_dict(d['algorithm']),TaskSpec.from_dict(d['task']),
                  d['strategy'],tuple(d['parents']),d['seed'],d['round_no'])
        except (KeyError,TypeError) as e:
            raise ValueError('invalid hypothesis record') from e
        if _json(h.to_dict())!=_json(d):
            raise ValueError('hypothesis payload differs from its executable definition')
        return h


@dataclass(frozen=True)
class Feedback:
    candidate_id: str
    utility: float
    source_role: str
    split_id: str
    evidence_digest: str


class HypothesisGenerator:
    """Bounded grammar search with typed mutation, crossover and optional feedback.

    Metadata-only proposal API. Caller-supplied provenance must be truthful;
    software cannot detect arbitrary data relabeled as development data.
    """
    def __init__(self,seed=42):
        if type(seed) is not int: raise ValueError('integer seed required')
        self.seed=seed;self.rng=random.Random(seed)
        self._issued={};self._ledger={};self._behavioral=set();self._parents=deque(maxlen=256);self._counts=Counter()
        self._rewards={m:[1.,1.] for m in MECHANISMS}
        self._feedback={};self._holdout=set();self._frozen=None
        self.duplicates_rejected=0;self.behavioral_aliases_rejected=0;self.constraints_rejected=0

    def _controller(self,task):
        favored=[]
        if set(task.goals)&{'outliers','target_noise'}: favored+=['dispersion','disagreement']
        if set(task.goals)&{'ill_conditioned','distribution_shift'}: favored+=['alignment','progress']
        names=list(SIGNALS)+favored
        def tree(depth):
            if depth==0 or self.rng.random()<.28:
                return Controller('signal',name=self.rng.choice(names))
            op=self.rng.choice(('mean','product','minimum','maximum','complement'))
            return Controller.make(op,tree(depth-1),*([] if op=='complement' else [tree(depth-1)]))
        for _ in range(100):
            c=tree(self.rng.choice((2,3)))
            if len(c.signals)>=2 and len(c.influences)>=2: return c
        raise RuntimeError('controller construction exhausted')

    def _mechanism(self):
        if len(self._issued)%4==0:
            return min(MECHANISMS,key=lambda m:self._counts[m])
        return self.rng.choices(MECHANISMS,weights=[a/(a+b)+.05 for a,b in self._rewards.values()])[0]

    def _fresh(self,task):
        mechanism=self._mechanism()
        return AlgorithmSpec(task.kind,self.rng.choice(MAPS),self.rng.choice(('standard','robust')),
            self.rng.choice(LOSSES[task.kind]),mechanism,self.rng.choice(('mean','median_of_means')),
            'rms' if mechanism=='variance_penalty' else self.rng.choice(('identity','rms')),
            self._controller(task))

    def _propose(self,task,strategy):
        parents=[h for h in self._parents if h.algorithm.fits(task)]
        if strategy=='construct' or not parents:
            return self._fresh(task),(), 'construct'
        p=self.rng.choice(parents)
        if strategy=='mutate':
            field=self.rng.choice(('feature_map','scaling','loss','mechanism','aggregation','preconditioner','controller'))
            pools={'feature_map':MAPS,'scaling':('standard','robust'),'loss':LOSSES[task.kind],
                   'mechanism':MECHANISMS,'aggregation':('mean','median_of_means'),'preconditioner':('identity','rms')}
            value=self._controller(task) if field=='controller' else self.rng.choice(pools[field])
            return replace(p.algorithm,**{field:value}), (p.signature,),strategy
        other=self.rng.choice(parents)
        def small(c):
            while c.depth>1: c=self.rng.choice(c.children)
            return c
        c=Controller.make(self.rng.choice(('mean','product','minimum','maximum')),
                          small(p.algorithm.controller),small(other.algorithm.controller))
        a=p.algorithm
        return replace(a,mechanism=other.algorithm.mechanism,loss=other.algorithm.loss,
                       aggregation=other.algorithm.aggregation,controller=c), (p.signature,other.signature),strategy

    def iter_generate(self,task,count=1)->Iterator[Hypothesis]:
        if self._frozen is not None: raise RuntimeError('generator is frozen')
        if not isinstance(task,TaskSpec): raise TypeError('TaskSpec metadata required; do not pass datasets')
        if type(count) is not int or not 1<=count<=100000: raise ValueError('count must be in 1..100000')
        minimum=AlgorithmSpec(task.kind,'linear','standard',LOSSES[task.kind][0],
            'clip_memory','mean','identity',Controller('half'))
        if not minimum.fits(task): raise ValueError('no algorithm fits the requested parameter/state budget')
        for _ in range(count):
            if self._frozen is not None: raise RuntimeError('generator is frozen')
            for attempt in range(500):
                strategy=('construct','mutate','crossover')[len(self._issued)%3]
                try:
                    a,parents,strategy=self._propose(task,strategy)
                except ValueError:
                    self.constraints_rejected+=1;continue
                if not a.fits(task) or len(a.controller.signals)<2 or len(a.controller.influences)<2:
                    self.constraints_rejected+=1;continue
                if a.structural_key in self._issued:
                    self.duplicates_rejected+=1;continue
                if a.behavioral_key in self._behavioral:
                    self.behavioral_aliases_rejected+=1;continue
                h=Hypothesis(a,task,strategy,parents,self.seed,len(self._issued)+1)
                self._issued[h.signature]=a.mechanism;self._counts[a.mechanism]+=1
                self._behavioral.add(a.behavioral_key)
                self._ledger[h.signature]=_hash({'task':asdict(task),'strategy':strategy,'parents':parents,'index':h.round_no})
                self._parents.append(h)
                yield h
                break
            else:
                raise RuntimeError('proposal budget exhausted; feasible candidates may remain; no silent partial completion')

    def generate(self,task,count=1):
        return list(self.iter_generate(task,count))

    def reserve_holdout(self,split_id):
        if self._frozen is not None: raise RuntimeError('generator is frozen')
        if not isinstance(split_id,str) or not split_id: raise ValueError('split ID required')
        if any(split_id==f[1] for f in self._feedback): raise ValueError('split already used for feedback')
        self._holdout.add(split_id)

    def observe(self,feedback):
        if self._frozen is not None: raise RuntimeError('generator is frozen')
        if not isinstance(feedback,Feedback): raise TypeError('Feedback record required')
        f=feedback
        if f.source_role!='development' or f.split_id in self._holdout:
            raise ValueError('held-out/test feedback is forbidden')
        if f.candidate_id not in self._issued: raise ValueError('unknown generated candidate')
        if not isinstance(f.split_id,str) or not f.split_id: raise ValueError('split ID required')
        if isinstance(f.utility,bool) or not isinstance(f.utility,(int,float)) or not math.isfinite(f.utility) or not 0<=f.utility<=1:
            raise ValueError('finite development utility in [0,1] required')
        if not isinstance(f.evidence_digest,str) or len(f.evidence_digest)!=64 or any(c not in '0123456789abcdef' for c in f.evidence_digest):
            raise ValueError('SHA-256 development-evidence digest required')
        key=(f.candidate_id,f.split_id,f.evidence_digest)
        if key in self._feedback: raise ValueError('duplicate feedback receipt')
        self._feedback[key]=f.utility
        a=self._rewards[self._issued[f.candidate_id]];a[0]+=f.utility;a[1]+=1-f.utility

    def freeze(self):
        if self._frozen is None:
            self._frozen=_hash({'version':VERSION,'seed':self.seed,'issued':self._issued,
                'tasks_and_provenance':self._ledger,'feedback':sorted([list(k)+[v] for k,v in self._feedback.items()]),'priors':self._rewards,'holdout':sorted(self._holdout)})
        return self._frozen

    def summary(self):
        return {'emitted':len(self._issued),'mechanisms':dict(self._counts),
                'duplicates_rejected':self.duplicates_rejected,'behavioral_aliases_rejected':self.behavioral_aliases_rejected,'constraints_rejected':self.constraints_rejected,
                'development_receipts':len(self._feedback),'frozen_digest':self._frozen,
                'uniqueness_scope':'canonical component/controller structures; not semantic or literature novelty'}


def _features(x):
    x=np.array(x,dtype=float,copy=True)
    if x.ndim!=2 or not all(x.shape) or not np.isfinite(x).all() or np.max(np.abs(x))>1e100:
        raise ValueError('nonempty finite feature matrix with magnitudes <=1e100 required')
    return x


@dataclass(frozen=True)
class Partition:
    """Role-tagged independent copy. Tags/IDs must reflect real provenance."""
    role: str
    x: np.ndarray
    y: np.ndarray
    row_ids: tuple[str,...]

    def __post_init__(self):
        if self.role not in ('train','development','test'): raise ValueError('invalid partition role')
        x=_features(self.x);y=np.array(self.y,copy=True)
        if y.ndim!=1 or len(y)!=len(x) or y.dtype.kind not in 'biufUS': raise ValueError('invalid targets')
        if y.dtype.kind in 'biuf' and (not np.isfinite(y).all() or np.max(np.abs(y))>1e100): raise ValueError('nonfinite/out-of-range targets')
        if not isinstance(self.row_ids,tuple) or len(self.row_ids)!=len(x) or any(not isinstance(i,str) for i in self.row_ids) or len(set(self.row_ids))!=len(x):
            raise ValueError('distinct string row IDs required')
        x.setflags(write=False);y.setflags(write=False)
        object.__setattr__(self,'x',x);object.__setattr__(self,'y',y)


def _row_hash(row):
    a=np.array(row,dtype=np.float64,copy=True);a[a==0]=0.
    return hashlib.sha256(a.tobytes()).hexdigest()


def assert_disjoint(parts):
    ids=set();rows=set()
    for p in parts:
        current=set(p.row_ids);hashes={_row_hash(r) for r in p.x}
        if ids&current or rows&hashes: raise ValueError('overlapping row IDs or duplicate feature rows across partitions')
        ids.update(current);rows.update(hashes)


def split_data(x,y,seed=42,groups=None):
    """Group-disjoint approximate 60/20/20 split; duplicates stay together.

    Not label-stratified. Rare classes can make a train split invalid; fit then
    fails explicitly. Supply real entity/group IDs when rows are dependent.
    """
    x=_features(x);y=np.asarray(y)
    if y.ndim!=1 or len(x)!=len(y) or len(x)<10: raise ValueError('at least 10 aligned rows required')
    groups=None if groups is None else np.asarray(groups)
    if groups is not None and (groups.ndim!=1 or len(groups)!=len(x)): raise ValueError('group length mismatch')
    parent=list(range(len(x)))
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]];i=parent[i]
        return i
    def merge(i,j): parent[find(i)]=find(j)
    seen_rows={};seen_groups={}
    for i,row in enumerate(x):
        key=_row_hash(row)
        if key in seen_rows: merge(i,seen_rows[key])
        seen_rows[key]=i
        if groups is not None:
            value=groups[i].item() if isinstance(groups[i],np.generic) else groups[i]
            if not isinstance(value,(str,int,float)) or (isinstance(value,float) and not math.isfinite(value)):
                raise ValueError('finite scalar group IDs required')
            key=(type(value).__name__,value)
            if key in seen_groups: merge(i,seen_groups[key])
            seen_groups[key]=i
    buckets={}
    for i in range(len(x)): buckets.setdefault(find(i),[]).append(i)
    buckets=list(buckets.values());random.Random(seed).shuffle(buckets)
    if len(buckets)<3: raise ValueError('at least three independent groups required')
    indices=[[],[],[]];stage=0
    for j,b in enumerate(buckets):
        left=len(buckets)-j
        if stage==0 and ((len(indices[0])>=.6*len(x) and j>0) or left==2): stage=1
        if stage==1 and ((len(indices[1])>=.2*len(x) and indices[1]) or left==1): stage=2
        indices[stage].extend(b)
    parts=tuple(Partition(role,x[idx],y[idx],tuple(str(i) for i in idx))
                for role,idx in zip(('train','development','test'),indices))
    assert_disjoint(parts)
    return parts


def loss_and_residual(logits,y,kind,loss):
    """Per-example loss and exact derivative w.r.t. logits (not averaged)."""
    if kind=='regression':
        r=logits[:,0]-np.asarray(y,dtype=float)
        if loss=='squared': return .5*r*r,r[:,None]
        if loss=='huber':
            v=np.where(np.abs(r)<=1,.5*r*r,np.abs(r)-.5)
            return v,np.clip(r,-1,1)[:,None]
        if loss=='pseudo_huber': return np.sqrt(1+r*r)-1,(r/np.sqrt(1+r*r))[:,None]
        if loss=='cauchy': return .5*np.log1p(r*r),(r/(1+r*r))[:,None]
        raise ValueError('invalid regression loss')
    if kind!='classification' or loss not in LOSSES[kind]: raise ValueError('invalid classification loss')
    shifted=logits-logits.max(axis=1,keepdims=True)
    logp=shifted-np.log(np.exp(shifted).sum(axis=1,keepdims=True));p=np.exp(logp)
    target=np.eye(logits.shape[1])[np.asarray(y,dtype=int)]
    r=p-target
    if loss=='cross_entropy': return -(target*logp).sum(axis=1),r
    if loss=='brier': return .5*(r*r).sum(axis=1),p*(r-(p*r).sum(axis=1,keepdims=True))
    pt=(p*target).sum(axis=1);logpt=(logp*target).sum(axis=1)
    factor=(1-pt)**2-2*pt*(1-pt)*logpt
    return -(1-pt)**2*logpt,r*factor[:,None]


class GeneratedModel:
    """Reference executor for emitted proposals. fit accepts TRAIN partitions only.

    It does not test novelty, mathematical theorems, or the ten acceptance gates.
    No validation/test arrays enter fitting, preprocessing, controllers or seeds.
    """
    def __init__(self,hypothesis,steps=160,learning_rate=.08,seed=0):
        if not isinstance(hypothesis,Hypothesis): raise TypeError('Hypothesis required')
        if type(steps) is not int or not 1<=steps<=20000: raise ValueError('steps in 1..20000 required')
        if not math.isfinite(learning_rate) or not 0<learning_rate<=1: raise ValueError('learning_rate must be in (0,1]')
        if type(seed) is not int: raise ValueError('integer execution seed required')
        self.hypothesis=hypothesis;self.steps=steps;self.learning_rate=learning_rate;self.seed=seed;self._fitted=False

    def _map(self,x):
        z=np.clip((x-self.center_)/self.scale_,-10,10)
        f=self.hypothesis.algorithm.feature_map
        if f=='diagonal_quadratic': z=np.concatenate((z,z*z/math.sqrt(2)),axis=1)
        elif f=='linear_fourier':
            angle=z@self.projection_;m=self.projection_.shape[1]
            z=np.concatenate((z,np.cos(angle)/math.sqrt(m),np.sin(angle)/math.sqrt(m)),axis=1)
        return np.column_stack((z,np.ones(len(z))))

    def fit(self,training):
        self._fitted=False
        if not isinstance(training,Partition) or training.role!='train': raise ValueError('fit accepts only an explicitly TRAIN-tagged Partition')
        h=self.hypothesis;a=h.algorithm;x=training.x;y=training.y
        if len(x)<2 or x.shape[1]!=h.task.features: raise ValueError('training shape differs from declared task')
        if a.kind=='classification':
            self.classes_,target=np.unique(y,return_inverse=True)
            if len(self.classes_)!=h.task.classes: raise ValueError('train rows must contain every declared class')
            self.target_center_=0.;self.target_scale_=1.
        else:
            target=np.asarray(y,dtype=float);self.classes_=None
            self.target_center_=float(target.mean());self.target_scale_=float(target.std()) or 1.
            target=(target-self.target_center_)/self.target_scale_
        if a.scaling=='robust':
            self.center_=np.median(x,axis=0);scale=1.4826*np.median(np.abs(x-self.center_),axis=0)
            scale=np.where(scale>1e-12,scale,x.std(axis=0))
        else:
            self.center_=x.mean(axis=0);scale=x.std(axis=0)
        self.scale_=np.where(scale>1e-12,scale,1.)
        rng=np.random.default_rng(self.seed)
        if a.feature_map=='linear_fourier':
            self.projection_=rng.normal(size=(x.shape[1],min(x.shape[1],32)))/math.sqrt(x.shape[1])
        z=self._map(x);n=len(z);k=h.task.classes
        theta=np.zeros((z.shape[1],k));fast=np.zeros_like(theta)
        previous=np.zeros_like(theta)
        slow=np.zeros_like(theta) if a.mechanism=='dual_memory' else None
        v=np.zeros_like(theta) if a.preconditioner=='rms' else None
        blocks=np.array_split(rng.permutation(n),min(5,n))
        previous_loss=None
        for t in range(self.steps):
            losses,residual=loss_and_residual(z@theta,target,a.kind,a.loss)
            raw=z.T@residual/n
            residual_norm=np.linalg.norm(residual,axis=1)
            gi_norm=np.linalg.norm(z,axis=1)*residual_norm
            raw_norm=float(np.linalg.norm(raw));prev_norm=float(np.linalg.norm(previous))
            moment=float(np.mean(gi_norm*gi_norm))
            dispersion=float(residual_norm.std())
            mean_loss=float(losses.mean())
            if previous_loss is None: previous_loss=mean_loss
            signals={'disagreement':max(moment-raw_norm**2,0)/(moment+1e-12),
                'dispersion':dispersion/(float(residual_norm.mean())+dispersion+1e-12),
                'alignment':(1-float(np.sum(raw*previous))/(raw_norm*prev_norm))/2 if raw_norm>1e-12 and prev_norm>1e-12 else .5,
                'progress':mean_loss/(mean_loss+previous_loss+1e-12),
                'time':t/max(self.steps-1,1),'energy':raw_norm/(1+raw_norm)}
            signals={key:float(np.clip(value,0,1)) for key,value in signals.items()}
            q=a.controller.evaluate(signals);q2=q if a.coupling=='coupled' else .5
            clip=max(float(np.quantile(gi_norm,.75)),1e-12)
            eta=self.learning_rate/math.sqrt(1+t/self.steps);beta=.9;reg=.001
            if a.mechanism in ('clip_memory','clip_step'): clip*=.25+1.5*(1-q)
            if a.mechanism=='clip_memory': beta=.95*(1-q2)
            elif a.mechanism=='clip_step': eta*=.25+.75*q2
            elif a.mechanism=='memory_step': beta=.95*q;eta*=1-.75*q2
            elif a.mechanism=='dual_memory': beta=.5;eta*=1-.5*q2
            elif a.mechanism=='penalty_step': reg*=.25+1.5*q;eta/=1+q2
            elif a.mechanism=='variance_penalty': reg*=.25+1.5*q
            weights=np.minimum(1.,clip/(gi_norm+1e-12))
            r=residual*weights[:,None]
            if a.aggregation=='mean': grad=z.T@r/n
            else: grad=np.median(np.stack([z[idx].T@r[idx]/len(idx) for idx in blocks]),axis=0)
            penalty=theta.copy();penalty[-1]=0.
            grad+=reg*penalty
            fast=beta*fast+(1-beta)*grad
            momentum=fast
            if slow is not None:
                slow=.95*slow+.05*grad;momentum=(1-q)*fast+q*slow
            denominator=1.
            if v is not None:
                v=.95*v+.05*grad*grad;denominator=np.sqrt(v)+1e-6
                if a.mechanism=='variance_penalty': denominator=(1-q2)+q2*denominator
            direction=momentum/denominator
            theta-=eta*direction/max(1.,float(np.linalg.norm(direction)))
            previous=raw;previous_loss=mean_loss
            if not np.isfinite(theta).all(): raise FloatingPointError('nonfinite candidate training state')
        self.weights_=theta;self._fitted=True
        return self

    def _logits(self,x):
        if not self._fitted: raise RuntimeError('fit a training partition first')
        x=_features(x)
        if x.shape[1]!=self.hypothesis.task.features: raise ValueError('prediction feature count mismatch')
        return self._map(x)@self.weights_

    def predict_proba(self,x):
        if self.hypothesis.task.kind!='classification': raise ValueError('probabilities require classification')
        logits=self._logits(x);logits-=logits.max(axis=1,keepdims=True)
        p=np.exp(logits);return p/p.sum(axis=1,keepdims=True)

    def predict(self,x):
        if not self._fitted: raise RuntimeError('fit a training partition first')
        if self.hypothesis.task.kind=='classification': return self.classes_[np.argmax(self.predict_proba(x),axis=1)]
        return self._logits(x)[:,0]*self.target_scale_+self.target_center_


def demonstrate(seeds=20,count=1000):
    """Generator engineering audit, not a ten-gate algorithm benchmark."""
    if type(seeds) is not int or not 1<=seeds<=100: raise ValueError('seeds must be 1..100')
    rows=[]
    task=TaskSpec('classification',17,5,goals=('outliers','ill_conditioned'))
    for seed in range(seeds):
        g=HypothesisGenerator(seed);start=perf_counter();hs=g.generate(task,count);elapsed=perf_counter()-start
        # Independent behavioral tests additionally inspect every schema and kernel.
        rows.append({'seed':seed,'count':len(hs),'unique_structures':len({h.signature for h in hs}),
                     'seconds':elapsed,'per_second':len(hs)/elapsed,
                     'mechanisms':dict(Counter(h.algorithm.mechanism for h in hs)),
                     'strategies':dict(Counter(h.strategy for h in hs)),
                     'feature_maps':dict(Counter(h.algorithm.feature_map for h in hs)),
                     'digest':_hash([h.signature for h in hs])})
    return {'version':VERSION,'scope':'metadata-only generator engineering, not scientific novelty or predictive superiority',
            'seeds':seeds,'total_generated':sum(r['count'] for r in rows),
            'all_within_run_structurally_unique':all(r['count']==r['unique_structures'] for r in rows),
            'median_per_second':float(np.median([r['per_second'] for r in rows])),
            'rows':rows,'example':hs[0].to_dict()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('generate');g.add_argument('--kind',choices=list(LOSSES),required=True)
    g.add_argument('--features',type=int,required=True);g.add_argument('--classes',type=int)
    g.add_argument('--goal',choices=GOALS,action='append',default=[])
    g.add_argument('--count',type=int,default=1);g.add_argument('--seed',type=int,default=42)
    d=sub.add_parser('demonstrate');d.add_argument('--seeds',type=int,default=20);d.add_argument('--count',type=int,default=1000)
    args=p.parse_args()
    if args.command=='generate':
        task=TaskSpec(args.kind,args.features,args.classes,tuple(args.goal))
        for h in HypothesisGenerator(args.seed).iter_generate(task,args.count):
            print(json.dumps(h.to_dict(),allow_nan=False))
    else:
        print(json.dumps(demonstrate(args.seeds,args.count),indent=2,allow_nan=False))

if __name__=='__main__':
    main()
