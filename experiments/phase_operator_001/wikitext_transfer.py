#!/usr/bin/env python3
"""Pinned WikiText-2 domain-transfer tests from the shared TinyStories checkpoint."""
import os,sys,json,math,time,random,hashlib,traceback
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import HfApi
import substantial_tinystories as s
assert torch.cuda.is_available(),'GPU_REQUIRED'
s.OUT=Path('/content/gcl_phase_wikitext');s.OUT.mkdir(exist_ok=True,parents=True)
OUT=s.OUT
def load_ids(split,n,revision,tok):
 path=OUT/f'{split}_{n}.pt'
 if path.exists():
  p=torch.load(path,weights_only=True)
  assert p['revision']==revision and p['tokenizer_sha']==TOKREV,'BAD_CACHE_REVISION'
  return p['ids'],p
 ds=load_dataset('Salesforce/wikitext',name='wikitext-2-raw-v1',split=split,revision=revision,streaming=True)
 if split=='train':ds=ds.shuffle(seed=42,buffer_size=256)
 arr=[];rows=0
 for doc in ds:
  t=doc['text']
  if not isinstance(t,str) or not t.strip():continue
  arr.extend(tok.encode(t,add_special_tokens=False));arr.append(tok.eos_token_id)
  rows+=1
  if len(arr)>=n:break
 assert len(arr)>=n,(split,len(arr))
 ids=torch.tensor(arr[:n],dtype=torch.long)
 p={'ids':ids,'revision':revision,'tokenizer_sha':TOKREV,'sha':hashlib.sha256(ids.numpy().tobytes()).hexdigest(),'rows':rows}
 torch.save(p,path);return ids,p
class LR16(s.LoRA):
 def __init__(self):super().__init__(r=16)
def make(kind,seed,state):
 torch.manual_seed(seed+500)
 model=s.Tiny('lora' if kind=='lora16' else kind).to(s.DEV)
 if kind=='lora16':
  for b in model.blocks:b.proj.adapt=LR16().to(s.DEV)
 ck=model.load_state_dict(state,strict=False)
 assert not ck.unexpected_keys and all('.adapt.' in k for k in ck.missing_keys),(kind,ck)
 s.test_model(model)
 if kind!='full':
  q=torch.randint(0,50257,(2,16),device=s.DEV)
  with torch.no_grad():
   baseline=s.Tiny('base').to(s.DEV)
   baseline.load_state_dict(state)
   assert torch.allclose(model(q),baseline(q),atol=1e-4,rtol=1e-4),'INITIAL_ADAPTER_NOT_NEUTRAL'
 return model
def main():
 global TOKREV
 s.random.seed(42);torch.manual_seed(42)
 ck=Path('/content/gcl_phase_lm/base.pt')
 assert ck.exists(),'BASE_CKPT_MISSING'
 base=torch.load(ck,map_location='cpu',weights_only=True)
 base_sha=hashlib.sha256(ck.read_bytes()).hexdigest()
 api=HfApi()
 dsrev=api.repo_info('Salesforce/wikitext',repo_type='dataset').sha
 TOKREV=base['tokenizer_revision']
 tok=AutoTokenizer.from_pretrained('openai-community/gpt2',revision=TOKREV)
 s.log(event='source_revision',dataset='Salesforce/wikitext',revision=dsrev,tokenizer_revision=TOKREV,base_sha=base_sha)
 train,ti=load_ids('train',650000,dsrev,tok)
 val,vi=load_ids('validation',90000,dsrev,tok)
 story=torch.load('/content/gcl_phase_lm/tokens_validation_50000.pt',weights_only=True)
 assert ti['sha']!=vi['sha'] and ti['sha']!=base['train_sha']
 s.log(event='dataset_ready',train_sha=ti['sha'],validation_sha=vi['sha'],train_rows=ti['rows'],validation_rows=vi['rows'])
 baseline=s.Tiny('base').to(s.DEV);baseline.load_state_dict(base['model']);s.test_model(baseline)
 b0=s.evaluate(baseline,val);old0=s.evaluate(baseline,story['ids'])
 s.log(event='baseline',wikitext=b0,tinystories=old0)
 scores=[]
 for kind in ('lora','lora16','real','phase','static','full'):
  for seed in (42,43,44):
   model=make(kind,seed,base['model'])
   result=s.fit(model,train,val,seed,int(os.getenv('GCL_ADAPT_STEPS','600')),.0003 if kind=='full' else .001,kind,base_ref=base_sha)
   old=s.evaluate(model,story['ids'])
   result['original_tinystories_nll']=old['nll'];result['original_tinystories_ppl']=old['ppl']
   result['wikitext_delta_nll']=result['nll']-b0['nll'];result['tinystories_delta_nll']=old['nll']-old0['nll']
   scores.append(result)
   (OUT/'results.json').write_text(json.dumps({'config':{'dataset':'Salesforce/wikitext','dataset_revision':dsrev,'tokenizer_rev':TOKREV,'base_sha256':base_sha,'steps':int(os.getenv('GCL_ADAPT_STEPS','600')),'train_sha':ti['sha'],'val_sha':vi['sha'],'original_val_sha':story['sha']},'baseline_wikitext':b0,'baseline_tinystories':old0,'results':scores},indent=2))
   s.log(event='transfer_summary',kind=kind,seed=seed,wikitext_nll=result['nll'],tinystories_nll=old['nll'])
   del model;torch.cuda.empty_cache()
 s.log(event='DONE',runs=len(scores),dataset='wikitext')
if __name__=='__main__':
 try:main()
 except BaseException as e:
  s.log(event='FATAL',error=str(e),traceback=traceback.format_exc());raise
