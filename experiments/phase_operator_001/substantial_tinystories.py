#!/usr/bin/env python3
"""PHASE-OPERATOR-001 real-data L4 gate. Correct causal next-token objective."""
import os,sys,math,time,json,random,hashlib,traceback
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import HfApi

OUT=Path('/content/gcl_phase_lm')
OUT.mkdir(exist_ok=True,parents=True)
D=96;HEADS=4;CTX=64;R=8;DEV='cuda'
assert torch.cuda.is_available(),'CUDA_REQUIRED_NO_CPU_FALLBACK'
torch.set_float32_matmul_precision('high')
def log(**kv):
 print(json.dumps(kv,default=str),flush=True)
 with (OUT/'events.jsonl').open('a') as f:f.write(json.dumps(kv,default=str)+'\n')
def ortho(x):return torch.linalg.qr(x.float(),mode='reduced')[0].to(x.dtype)
def basis(d,r):return torch.linalg.qr(torch.randn(d,r))[0]
class Phase(nn.Module):
 def __init__(self,d=D,r=R,dynamic=True):
  super().__init__();self.r=r;self.u=nn.Parameter(basis(d,2*r));self.v=nn.Parameter(basis(d,r));self.ctrl=nn.Linear(d,r,bias=False) if dynamic else None
  if self.ctrl:nn.init.zeros_(self.ctrl.weight)
  self.b=nn.Parameter(torch.zeros(r));self.raw=nn.Parameter(torch.zeros(()))
 def forward(self,x):
  u,v=ortho(self.u),ortho(self.v)
  ang=self.b+(self.ctrl(x) if self.ctrl else 0)
  h=x@v
  return (.3*self.raw.tanh())*((h*ang.cos())@u[:,:self.r].T+(h*ang.sin())@u[:,self.r:].T)
 def bound(self):return float((.3*self.raw.tanh()).abs().detach())
class Real(nn.Module):
 def __init__(self,d=D,r=12):
  super().__init__();self.u=nn.Parameter(basis(d,r));self.v=nn.Parameter(basis(d,r));self.ctrl=nn.Linear(d,r)
  nn.init.zeros_(self.ctrl.weight);nn.init.zeros_(self.ctrl.bias);self.raw=nn.Parameter(torch.tensor(1.))
 def forward(self,x):
  return (.3*self.raw.sigmoid())*(((x@ortho(self.v))*self.ctrl(x).tanh())@ortho(self.u).T)
class LoRA(nn.Module):
 def __init__(self,d=D,r=R):
  super().__init__();self.a=nn.Linear(d,r,bias=False);self.b=nn.Linear(r,d,bias=False)
  nn.init.normal_(self.a.weight,std=1/math.sqrt(d));nn.init.zeros_(self.b.weight)
 def forward(self,x):return self.b(self.a(x))
class Projection(nn.Module):
 def __init__(self,kind):
  super().__init__();self.base=nn.Linear(D,D,bias=False);self.adapt=None if kind in ('base','full') else (Phase(dynamic=(kind=='phase')) if kind in ('phase','static') else Real() if kind=='real' else LoRA())
 def forward(self,x):return self.base(x) if self.adapt is None else self.base(x)+self.adapt(x)
class Block(nn.Module):
 def __init__(self,kind):
  super().__init__();self.a=nn.LayerNorm(D);self.b=nn.LayerNorm(D);self.qkv=nn.Linear(D,3*D,bias=False);self.proj=Projection(kind);self.ff=nn.Sequential(nn.Linear(D,4*D),nn.GELU(),nn.Linear(4*D,D))
 def forward(self,x):
  y=self.a(x);q,k,v=self.qkv(y).chunk(3,dim=-1);B,T,_=x.shape
  def spl(a):return a.reshape(B,T,HEADS,D//HEADS).transpose(1,2)
  out=F.scaled_dot_product_attention(spl(q),spl(k),spl(v),is_causal=True)
  out=out.transpose(1,2).contiguous().reshape(B,T,D)
  x=x+self.proj(out)
  return x+self.ff(self.b(x))
class Tiny(nn.Module):
 def __init__(self,kind):
  super().__init__();self.kind=kind;self.token=nn.Embedding(50257,D);self.pos=nn.Embedding(CTX,D);self.blocks=nn.ModuleList([Block(kind) for _ in range(2)]);self.norm=nn.LayerNorm(D);self.head=nn.Linear(D,50257,bias=False)
  for n,m in self.named_modules():
   if '.adapt.' in n:continue
   if isinstance(m,(nn.Linear,nn.Embedding)):
    nn.init.normal_(m.weight,std=.02)
    if isinstance(m,nn.Linear) and m.bias is not None:nn.init.zeros_(m.bias)
  self.head.weight=self.token.weight
 def forward(self,ids):
  B,T=ids.shape;assert T<=CTX
  x=self.token(ids)+self.pos(torch.arange(T,device=ids.device))[None,:,:]
  for b in self.blocks:x=b(x)
  return self.head(self.norm(x))
def source():
 api=HfApi()
 dsrev=api.repo_info('roneneldan/TinyStories',repo_type='dataset').sha
 tokrev=api.model_info('openai-community/gpt2').sha
 log(event='source_revision',dataset='roneneldan/TinyStories',dataset_revision=dsrev,tokenizer='openai-community/gpt2',tokenizer_revision=tokrev)
 return dsrev,tokrev
def cache_tokens(split,target,dsrev,tok,seed):
 path=OUT/f'tokens_{split}_{target}.pt'
 if path.exists():
  q=torch.load(path,weights_only=True)
  a=q['ids'];digest=hashlib.sha256(a.numpy().tobytes()).hexdigest()
  assert digest==q['sha'] and q['dsrev']==dsrev and q['tokrev']==tok.init_kwargs.get('_commit_hash',None),'CACHE_SOURCE_MISMATCH'
  return a,q
 stream=load_dataset('roneneldan/TinyStories',split=split,streaming=True,revision=dsrev)
 if split=='train':stream=stream.shuffle(seed=seed,buffer_size=256)
 ids=[];docs=0
 for row in stream:
  txt=row['text']
  if not isinstance(txt,str) or not txt.strip():continue
  ids.extend(tok.encode(txt,add_special_tokens=False))
  ids.append(tok.eos_token_id);docs+=1
  if len(ids)>=target:break
 assert len(ids)>=target,'DATASET_INSUFFICIENT'
 a=torch.tensor(ids[:target],dtype=torch.long)
 q={'ids':a,'sha':hashlib.sha256(a.numpy().tobytes()).hexdigest(),'docs':docs,'dsrev':dsrev,'tokrev':tok.init_kwargs.get('_commit_hash',None)}
 torch.save(q,path);return a,q
def sample(ids,bs,
g):
 starts=torch.randint(len(ids)-CTX-1,(bs,),generator=g)
 ar=torch.arange(CTX+1)
 a=ids[starts[:,None]+ar[None,:]].to(DEV)
 return a[:,:-1],a[:,1:]
@torch.no_grad()
def evaluate(model,ids):
 model.eval();loss_sum=0.;correct=0;count=0
 st=torch.linspace(0,len(ids)-CTX-1,steps=256).long()
 off=torch.arange(CTX+1)
 for k in range(0,256,8):
  win=ids[st[k:k+8,None]+off[None,:]].to(DEV);inp,target=win[:,:-1],win[:,1:]
  lg=model(inp);loss_sum+=float(F.cross_entropy(lg.flatten(0,1).float(),target.flatten(),reduction='sum'))
  correct+=(lg.argmax(-1)==target).sum().item();count+=target.numel()
 nll=loss_sum/count
 return {'nll':nll,'ppl':math.exp(nll),'top1':correct/count,'val_tokens':count}
def test_model(model):
 model.eval()
 a=torch.randint(0,50257,(2,12),device=DEV)
 b=a.clone();b[:,7:]=torch.randint(0,50257,(2,5),device=DEV)
 with torch.no_grad():
  l1=model(a);l2=model(b);solo=model(a[:1])
 assert torch.allclose(l1[:,:7],l2[:,:7],atol=5e-5,rtol=5e-5),'CAUSAL_LEAK'
 assert torch.allclose(l1[:1],solo,atol=5e-5,rtol=5e-5),'BATCH_LEAK'
def fit(model,ids,val,seed,steps,lr,label,base_ref=None):
 if label not in ('pretrain','full'):
  for p in model.parameters():p.requires_grad_(False)
  for b in model.blocks:
   for p in b.proj.adapt.parameters():p.requires_grad_(True)
 trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
 total=sum(p.numel() for p in model.parameters())
 torch.cuda.reset_peak_memory_stats()
 opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=lr,weight_decay=.01)
 rng=torch.Generator().manual_seed(97832+seed)
 tic=time.perf_counter();model.train();batch=8
 initial=evaluate(model,val)
 log(event='start_training',label=label,seed=seed,steps=steps,params=trainable,total_params=total,initial=initial,base_ref=base_ref)
 for step in range(1,steps+1):
  model.train();x,y=sample(ids,batch,rng)
  logits=model(x);loss=F.cross_entropy(logits.flatten(0,1).float(),y.flatten())
  assert torch.isfinite(loss),'NONFINITE_LOSS'
  opt.zero_grad(set_to_none=True);loss.backward()
  gn=torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.,error_if_nonfinite=True)
  opt.step()
  if step%100==0 or step==steps:
   valm=evaluate(model,val);torch.cuda.synchronize()
   log(event='train_metric',label=label,seed=seed,step=step,train_loss=float(loss.detach()),grad_norm=float(gn),val=valm,elapsed=round(time.perf_counter()-tic,2))
 final=evaluate(model,val);secs=time.perf_counter()-tic
 result={'label':label,'seed':seed,'steps':steps,'params':trainable,'total_params':total,
         'initial_nll':initial['nll'],'nll':final['nll'],'ppl':final['ppl'],'top1':final['top1'],
         'tokens_seen':steps*batch*CTX,'seconds':secs,'train_tok_s':steps*batch*CTX/secs,
         'peak_mem_mb':torch.cuda.max_memory_allocated()/1e6}
 log(event='finish',**result)
 return result
def main():
 random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
 log(event='environment',torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0))
 dsrev,tokrev=source()
 tok=AutoTokenizer.from_pretrained('openai-community/gpt2',revision=tokrev)
 train,ti=cache_tokens('train',450000,dsrev,tok,42)
 val,vi=cache_tokens('validation',50000,dsrev,tok,42)
 log(event='dataset_ready',train_sha=ti['sha'],val_sha=vi['sha'],train_docs=ti['docs'],val_docs=vi['docs'])
 assert ti['sha']!=vi['sha'] and train.max()<50257 and val.max()<50257
 base=Tiny('base').to(DEV);test_model(base)
 base_result=fit(base,train,val,42,int(os.getenv('GCL_BASE_STEPS','500')),3e-4,'pretrain')
 base_state={k:v.detach().cpu().clone() for k,v in base.state_dict().items()}
 torch.save({'model':base_state,'config':{'width':D,'depth':2,'heads':HEADS,'context':CTX},'train_sha':ti['sha'],'val_sha':vi['sha'],'dataset_revision':dsrev,'tokenizer_revision':tokrev},OUT/'base.pt')
 results=[base_result]
 for kind in ('lora','real','phase','static','full'):
  for seed in (42,43,44):
   torch.manual_seed(seed+500)
   model=Tiny(kind).to(DEV)
   ck=model.load_state_dict(base_state,strict=False)
   assert not ck.unexpected_keys and all('.adapt.' in k for k in ck.missing_keys),str(ck)
   test_model(model)
   if kind in ('lora','real','phase','static'):
    with torch.no_grad():
     check=torch.randint(0,50257,(2,16),device=DEV)
     assert torch.allclose(model(check),base(check),atol=1e-4,rtol=1e-4),'NONZERO_ADAPTER_INITIALIZATION'
   if kind=='phase':
    for m in model.modules():
     if isinstance(m,Phase):assert m.bound()<=.3+1e-8
   res=fit(model,train,val,seed,int(os.getenv('GCL_ADAPT_STEPS','200')),3e-4 if kind=='full' else 1e-3,kind,base_ref='base.pt')
   results.append(res)
   (OUT/'results.json').write_text(json.dumps({'environment':{'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__},'train_sha':ti['sha'],'val_sha':vi['sha'],'dataset_rev':dsrev,'tokenizer_rev':tokrev,'results':results},indent=2))
   del model;torch.cuda.empty_cache()
 log(event='DONE',models=len(results))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  log(event='FATAL',error=str(e),traceback=traceback.format_exc());raise
