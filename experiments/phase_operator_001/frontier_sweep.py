#!/usr/bin/env python3
"""Norm-budget transfer Pareto sweep: fixed rank-16 LoRA, seeds 42–44."""
import json,torch,traceback
from pathlib import Path
import torch.nn.functional as F
import substantial_tinystories as s
assert torch.cuda.is_available()
OUT=Path('/content/gcl_phase_frontier');OUT.mkdir(exist_ok=True,parents=True)
s.OUT=OUT
base=torch.load('/content/gcl_phase_lm/base.pt',weights_only=True)
wiki=json.load(open('/content/gcl_phase_wikitext/results.json'))
q=wiki['config']
tr=torch.load('/content/gcl_phase_wikitext/train_650000.pt',weights_only=True)
val=torch.load('/content/gcl_phase_wikitext/validation_90000.pt',weights_only=True)
story=torch.load('/content/gcl_phase_lm/tokens_validation_50000.pt',weights_only=True)
assert tr['sha']==q['train_sha'] and val['sha']==q['val_sha'] and story['sha']==q['original_val_sha']
class ClippedLoRA(s.LoRA):
 def __init__(self,cap):
  super().__init__(r=16);self.cap=cap
 def forward(self,x):
  m=self.b.weight@self.a.weight
  sig=torch.linalg.matrix_norm(m.float(),ord=2)
  scale=(self.cap/(sig+1e-8)).clamp(max=1.)
  return F.linear(x,m*scale)
def main():
 allr=[]
 for cap in (.10,.15,.20,.25):
  for seed in (42,43,44):
   torch.manual_seed(seed+500)
   m=s.Tiny('lora').to('cuda')
   for b in m.blocks:b.proj.adapt=ClippedLoRA(cap).to('cuda')
   ck=m.load_state_dict(base['model'],strict=False)
   assert not ck.unexpected_keys and all('.adapt.' in k for k in ck.missing_keys)
   s.test_model(m)
   with torch.no_grad():
    v=s.Tiny('base').to('cuda');v.load_state_dict(base['model'])
    inp=torch.randint(0,50257,(2,16),device='cuda')
    assert torch.allclose(m(inp),v(inp),atol=1e-4,rtol=1e-4)
   r=s.fit(m,tr['ids'],val['ids'],seed,600,.001,f'lora16_bound_{cap}',q['base_sha256'])
   old=s.evaluate(m,story['ids'])
   r['original_nll']=old['nll'];r['original_delta']=old['nll']-wiki['baseline_tinystories']['nll']
   r['wikitext_delta']=r['nll']-wiki['baseline_wikitext']['nll']
   r['cap']=cap
   allr.append(r)
   (OUT/'results.json').write_text(json.dumps({'train_sha':q['train_sha'],'val_sha':q['val_sha'],'original_sha':q['original_val_sha'],'base_sha':q['base_sha256'],'results':allr},indent=2))
   s.log(event='frontier_result',seed=seed,cap=cap,wikitext_nll=r['nll'],old_nll=old['nll'])
   del m;torch.cuda.empty_cache()
 s.log(event='DONE',runs=len(allr))
if __name__=='__main__':
 try:main()
 except BaseException as e:s.log(event='FATAL',error=str(e),traceback=traceback.format_exc());raise
