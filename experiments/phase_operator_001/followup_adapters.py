#!/usr/bin/env python3
"""Causal WikiText-2 phase-centering vs spectrally bounded LoRA falsification."""
import os,json,time,math,traceback
from pathlib import Path
import torch
import torch.nn.functional as F
from torch import nn
import substantial_tinystories as s
assert torch.cuda.is_available(),'GPU_ONLY'
OUT=Path('/content/gcl_phase_followup');OUT.mkdir(exist_ok=True,parents=True)
s.OUT=OUT
base=torch.load('/content/gcl_phase_lm/base.pt',weights_only=True,map_location='cpu')
prior=json.load(open('/content/gcl_phase_wikitext/results.json'))
config=prior['config']
assert base['dataset_revision'] and config['original_val_sha']==base['val_sha']
trainfile='/content/gcl_phase_wikitext/train_650000.pt';valfile='/content/gcl_phase_wikitext/validation_90000.pt'
trainp=torch.load(trainfile,weights_only=True);valp=torch.load(valfile,weights_only=True)
assert trainp['sha']==config['train_sha'] and valp['sha']==config['val_sha'],'TOKEN_SOURCE_DRIFT'
orig=torch.load('/content/gcl_phase_lm/tokens_validation_50000.pt',weights_only=True)
assert orig['sha']==config['original_val_sha']
class CenteredPhase(s.Phase):
 def __init__(self):
  super().__init__(dynamic=True)
  with torch.no_grad():self.raw.fill_(1.5)
 def forward(self,x):
  u,v=s.ortho(self.u),s.ortho(self.v)
  a=self.b+self.ctrl(x);c=x@v
  return (.15*self.raw.sigmoid())*(((c*(a.cos()-1))@u[:,:self.r].T)+((c*a.sin())@u[:,self.r:].T))
 def bound(self):return .3
class BoundedLoRA(s.LoRA):
 def __init__(self):
  super().__init__(r=16)
 def forward(self,x):
  mat=self.b.weight@self.a.weight
  sig=torch.linalg.matrix_norm(mat.float(),ord=2)
  scale=(.3/(sig+1e-8)).clamp(max=1.)
  return F.linear(x,mat*scale)
def build(kind,seed):
 torch.manual_seed(seed+500)
 m=s.Tiny('phase' if kind=='centered' else 'lora').to('cuda')
 for b in m.blocks:
  b.proj.adapt=(CenteredPhase() if kind=='centered' else BoundedLoRA()).to('cuda')
 res=m.load_state_dict(base['model'],strict=False)
 assert not res.unexpected_keys and all('.adapt.' in k for k in res.missing_keys)
 s.test_model(m)
 with torch.no_grad():
  ref=s.Tiny('base').to('cuda');ref.load_state_dict(base['model'])
  idx=torch.randint(0,50257,(2,16),device='cuda')
  assert torch.allclose(m(idx),ref(idx),atol=1e-4,rtol=1e-4)
 return m
def main():
 base_nll=prior['baseline_wikitext']['nll']
 old_nll=prior['baseline_tinystories']['nll']
 allres=[]
 for kind in ('centered','bounded_lora16'):
  for seed in (42,43,44):
   model=build(kind,seed)
   r=s.fit(model,trainp['ids'],valp['ids'],seed,600,.001,kind,'base.pt')
   r['tinystories_nll']=s.evaluate(model,orig['ids'])['nll']
   r['wikitext_delta']=r['nll']-base_nll
   r['tinystories_delta']=r['tinystories_nll']-old_nll
   with torch.no_grad():
    if kind=='bounded_lora16':
     r['actual_max_update_spectral_norm']=max(float(torch.linalg.matrix_norm((b.proj.adapt.b.weight@b.proj.adapt.a.weight).float(),ord=2).clamp(max=.3)) for b in model.blocks)
    else:r['actual_max_update_spectral_norm'] = max(.15*float(b.proj.adapt.raw.sigmoid())*2 for b in model.blocks)
   allres.append(r)
   (OUT/'results.json').write_text(json.dumps({'base_sha256':config['base_sha256'],'train_sha256':config['train_sha'],'val_sha256':config['val_sha'],'results':allres},indent=2))
   s.log(event='followup_result',kind=kind,seed=seed,nll=r['nll'],old_nll=r['tinystories_nll'],max_norm=r['actual_max_update_spectral_norm'])
   del model;torch.cuda.empty_cache()
 s.log(event='DONE',n=len(allres))
if __name__=='__main__':
 try:main()
 except BaseException as e:s.log(event='FATAL',error=str(e),traceback=traceback.format_exc());raise
