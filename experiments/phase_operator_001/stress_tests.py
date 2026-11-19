#!/usr/bin/env python3
"""Randomized CUDA certificates and Transformer causality stress checks."""
import json,math,io,random,torch
import torch.nn.functional as F
import substantial_tinystories as s
assert torch.cuda.is_available(),'GPU_REQUIRED'
device='cuda'
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32=False
checks={'spectral':0,'phase_lipschitz':0,'dense_equivalence':0,'autograd_fd':0,'causal_prefix':0,'batch_independence':0,'state_roundtrip':0}
max_err=0.
for seed in range(30):
 torch.manual_seed(seed+100)
 d,r=(32,4) if seed%3==0 else ((64,8) if seed%3==1 else (96,12))
 p=s.Phase(d=d,r=r,dynamic=True).to(device)
 with torch.no_grad():
  p.ctrl.weight.normal_(std=.2);p.b.uniform_(-1,1);p.raw.uniform_(-2,2)
 u,v=s.ortho(p.u),s.ortho(p.v)
 for k in range(4):
  ctx=torch.randn(d,device=device);other=torch.randn(d,device=device)
  t=p.b+p.ctrl(ctx);t2=p.b+p.ctrl(other)
  g=.3*p.raw.tanh()
  mat=g*((u[:,:r]*t.cos())@v.T+(u[:,r:]*t.sin())@v.T)
  mat2=g*((u[:,:r]*t2.cos())@v.T+(u[:,r:]*t2.sin())@v.T)
  spectral=float(torch.linalg.matrix_norm(mat,ord=2))
  assert spectral<=abs(float(g))+2e-5,(seed,d,spectral,float(g));checks['spectral']+=1
  diff=float(torch.linalg.matrix_norm(mat-mat2,ord=2))
  angle_bound=abs(float(g))*float((t-t2).abs().max())
  assert diff<=angle_bound+2e-5,(seed,d,diff,angle_bound);checks['phase_lipschitz']+=1
  y=p(ctx[None,:])[0]
  err=float((y-(mat@ctx)).abs().max())
  assert torch.allclose(y,mat@ctx,atol=1e-4,rtol=2e-4),(seed,err);max_err=max(max_err,err);checks['dense_equivalence']+=1
 target=torch.randn(4,d,device=device);z=torch.randn(4,d,device=device)
 loss=F.mse_loss(p(z),target);loss.backward()
 grads=[p.u.grad,p.v.grad,p.ctrl.weight.grad,p.b.grad,p.raw.grad]
 assert all(g is not None and g.isfinite().all() and float(g.norm())>1e-8 for g in grads),(seed,[float(x.norm()) if x is not None else None for x in grads])
 if seed%3==0:
  gfloat=float(p.ctrl.weight.grad[0,0]);eps=.001
  with torch.no_grad():
   p.ctrl.weight[0,0]+=eps;plus=float(F.mse_loss(p(z),target))
   p.ctrl.weight[0,0]-=2*eps;minus=float(F.mse_loss(p(z),target))
   p.ctrl.weight[0,0]+=eps
  fd=(plus-minus)/(2*eps)
  assert abs(gfloat-fd)<max(2e-4,abs(gfloat)*.06),(seed,gfloat,fd)
  checks['autograd_fd']+=1
for kind in ('base','phase','real','static','lora'):
 torch.manual_seed(101)
 m=s.Tiny(kind).to(device);m.eval()
 for length in (8,16,32,64):
  for B in (1,2,4):
   x=torch.randint(0,50257,(B,length),device=device)
   y=x.clone();pivot=length//2;y[:,pivot:]=torch.randint(0,50257,(B,length-pivot),device=device)
   with torch.no_grad():a=m(x);b=m(y);solo=m(x[:1])
   assert torch.allclose(a[:,:pivot],b[:,:pivot],atol=2e-4,rtol=2e-4),(kind,length,B,'CAUSAL')
   checks['causal_prefix']+=1
   assert torch.allclose(a[:1],solo,atol=2e-4,rtol=2e-4),(kind,length,B,'BATCH')
   checks['batch_independence']+=1
 buffer=io.BytesIO();torch.save(m.state_dict(),buffer);buffer.seek(0)
 n=s.Tiny(kind).to(device);n.load_state_dict(torch.load(buffer,map_location=device,weights_only=True));n.eval()
 with torch.no_grad():
  tok=torch.randint(0,50257,(2,32),device=device)
  assert torch.allclose(m(tok),n(tok),atol=0,rtol=0),(kind,'ROUNDTRIP')
 checks['state_roundtrip']+=1
print(json.dumps({'event':'gpu_stress_PASS','device':torch.cuda.get_device_name(0),'torch':torch.__version__,'counts':checks,'max_dense_diff':max_err}),flush=True)
