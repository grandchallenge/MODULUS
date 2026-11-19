import os,json,math,time,argparse,torch
from torch import nn
from torch.nn import functional as F
class Phase(nn.Module):
 def __init__(self,d=64,r=8,c=32):
  super().__init__();self.r=r;self.U=nn.Parameter(torch.linalg.qr(torch.randn(d,2*r))[0]);self.V=nn.Parameter(torch.linalg.qr(torch.randn(d,r))[0]);self.C=nn.Linear(c,r,bias=False);nn.init.zeros_(self.C.weight);self.b=nn.Parameter(torch.zeros(r));self.g=nn.Parameter(torch.tensor(1.5))
 def bases(self):return torch.linalg.qr(self.U)[0],torch.linalg.qr(self.V)[0]
 def angles(self,z):return self.C(z)+self.b
 def forward(self,x,z):
  u,v=self.bases();a=self.angles(z);h=x@v
  return self.g.sigmoid()*((h*a.cos())@u[:,:self.r].T+(h*a.sin())@u[:,self.r:].T)
 def matrices(self,z):
  u,v=self.bases();a=self.angles(z)
  return self.g.sigmoid()*(torch.einsum('ir,nr,jr->nij',u[:,:self.r],a.cos(),v)+torch.einsum('ir,nr,jr->nij',u[:,self.r:],a.sin(),v))
class Real(nn.Module):
 def __init__(self,d=64,r=12,c=32):
  super().__init__();self.U=nn.Parameter(torch.linalg.qr(torch.randn(d,r))[0]);self.V=nn.Parameter(torch.linalg.qr(torch.randn(d,r))[0]);self.C=nn.Linear(c,r);nn.init.zeros_(self.C.weight);nn.init.zeros_(self.C.bias);self.g=nn.Parameter(torch.tensor(1.5))
 def forward(self,x,z):
  u=torch.linalg.qr(self.U)[0];v=torch.linalg.qr(self.V)[0]
  return self.g.sigmoid()*(((x@v)*self.C(z).tanh())@u.T)
class LoRA(nn.Module):
 def __init__(self,d=64,r=12):
  super().__init__();self.a=nn.Linear(d,r,bias=False);self.b=nn.Linear(r,d,bias=False);nn.init.normal_(self.a.weight,std=1/math.sqrt(d));nn.init.zeros_(self.b.weight)
 def forward(self,x,z):return self.b(self.a(x))
def data(seed,n,dev,shift=0):
 gen=torch.Generator(device=dev).manual_seed(seed)
 return torch.randn(n,64,device=dev,generator=gen),torch.randn(n,32,device=dev,generator=gen)+shift
@torch.no_grad()
def score(t,s,seed,dev,shift):
 x,z=data(seed,2048,dev,shift);a=t(x,z);b=s(x,z)
 return float((a-b).square().sum()/a.square().sum().clamp_min(1e-10))
def check(dev):
 torch.manual_seed(17);p=Phase(32,4,16).to(dev)
 with torch.no_grad():p.C.weight.normal_(std=.35);p.b.uniform_(-1,1)
 x=torch.randn(2,32,device=dev);z=torch.randn(2,16,device=dev)
 M=p.matrices(z[:1])[0];v=p(x[:1],z[:1])[0]
 assert torch.allclose(M@x[0],v,atol=1e-5)
 g=float(p.g.sigmoid());spec=float(torch.linalg.matrix_norm(M,ord=2));assert spec<=g+1e-5
 delta=float(torch.linalg.matrix_norm(p.matrices(z[:1])[0]-p.matrices(z[1:])[0],ord=2))
 bound=float(p.g.sigmoid()*(p.angles(z[:1])-p.angles(z[1:])).abs().max());assert delta<=bound+1e-5
 target=torch.randn_like(x)
 loss=F.mse_loss(p(x,z),target);loss.backward();grad=float(p.C.weight.grad[0,0]);eps=1e-3
 with torch.no_grad():
  p.C.weight[0,0]+=eps;plus=float(F.mse_loss(p(x,z),target));p.C.weight[0,0]-=2*eps;minus=float(F.mse_loss(p(x,z),target));p.C.weight[0,0]+=eps
 fd=(plus-minus)/(2*eps);assert abs(fd-grad)<max(1e-4,abs(grad)*.03) and abs(grad)>1e-6,(grad,fd)
 return {'spectral_norm':spec,'gain':g,'difference':delta,'bound':bound,'gradient':grad,'finite_difference':fd}
def run(seed,steps):
 assert torch.cuda.is_available(),'NO_GPU_CPU_FALLBACK_FORBIDDEN'
 dev='cuda';torch.manual_seed(seed)
 print(json.dumps({'event':'env','seed':seed,'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'test':check(dev)}),flush=True)
 for scale in (.3,1.5):
  torch.manual_seed(seed+int(scale*100));t=Phase().to(dev)
  with torch.no_grad():t.C.weight.normal_(std=scale/math.sqrt(32));t.b.uniform_(-1,1)
  t.eval();t.requires_grad_(False)
  for kind in ('joint','svd','oracle','real','lora'):
   torch.manual_seed(seed+902)
   s=(Real() if kind=='real' else LoRA() if kind=='lora' else Phase()).to(dev)
   if kind=='oracle':
    with torch.no_grad():s.U.copy_(t.U);s.V.copy_(t.V)
    s.U.requires_grad_(False);s.V.requires_grad_(False)
   if kind=='svd':
    with torch.no_grad():
     _,z=data(seed+313,48,dev);mat=t.matrices(z)
     u=torch.linalg.svd(mat.permute(1,0,2).reshape(64,-1),full_matrices=False).U[:,:16]
     v=torch.linalg.svd(mat.permute(2,0,1).reshape(64,-1),full_matrices=False).U[:,:8]
     s.U.copy_(u);s.V.copy_(v)
   opt=torch.optim.AdamW((p for p in s.parameters() if p.requires_grad),lr=.003,weight_decay=0.)
   rng=torch.Generator(device=dev).manual_seed(seed+917);tic=time.perf_counter()
   for step in range(steps):
    x=torch.randn(256,64,device=dev,generator=rng);z=torch.randn(256,32,device=dev,generator=rng)
    with torch.no_grad():y=t(x,z)
    loss=F.mse_loss(s(x,z),y);assert torch.isfinite(loss)
    opt.zero_grad(set_to_none=True);loss.backward()
    torch.nn.utils.clip_grad_norm_(s.parameters(),1.,error_if_nonfinite=True);opt.step()
   torch.cuda.synchronize()
   print(json.dumps({'event':'result','seed':seed,'scale':scale,'kind':kind,'steps':steps,'params':sum(p.numel() for p in s.parameters() if p.requires_grad),'loss':float(loss),'id':score(t,s,seed+200,dev,0),'ood':score(t,s,seed+300,dev,2),'seconds':round(time.perf_counter()-tic,2)}),flush=True)
 print(json.dumps({'event':'DONE','seed':seed}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--steps',type=int,default=1500);a,_=p.parse_known_args();run(int(os.getenv("GCL_SEED",a.seed)),int(os.getenv("GCL_STEPS",a.steps)))
