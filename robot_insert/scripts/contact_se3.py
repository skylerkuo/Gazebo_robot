"""FP32 SE(3) right residuals; xi=[rho metres, phi radians], pose=xyz+xyzw."""
import torch


def unit(q):
    q=q.float()
    if not torch.isfinite(q).all() or (q.norm(dim=-1)<1e-8).any():
        raise ValueError('Invalid quaternion')
    return q/q.norm(dim=-1,keepdim=True)


def canonical(q):
    q=unit(q)
    # Deterministic axis at pi, including q/-q with signed zero scalar parts.
    v=q[...,:3];first=v.gather(-1,(v.abs()>1e-7).int().argmax(-1,keepdim=True))
    flip=torch.where(q[...,3:].abs()<1e-7,first<0,q[...,3:]<0)
    return torch.where(flip,-q,q)


def multiply(a,b):
    av,aw=a[...,:3],a[...,3:];bv,bw=b[...,:3],b[...,3:]
    return torch.cat((aw*bv+bw*av+torch.cross(av,bv,dim=-1),aw*bw-(av*bv).sum(-1,keepdim=True)),-1)


def rotate(q,v):
    q=unit(q);xyz=q[...,:3]
    return v+2*torch.cross(xyz,torch.cross(xyz,v,dim=-1)+q[...,3:]*v,dim=-1)


def so3_log(q):
    q=canonical(q);v=q[...,:3];n=v.norm(dim=-1,keepdim=True)
    return v*torch.where(n>1e-6,2*torch.atan2(n,q[...,3:])/n.clamp_min(1e-8),2+n.square()/3)


def so3_exp(phi):
    theta=phi.norm(dim=-1,keepdim=True);t2=theta.square()
    ratio=torch.where(theta>1e-4,torch.sin(theta/2)/theta.clamp_min(1e-8),.5-t2/48+t2.square()/3840)
    return unit(torch.cat((ratio*phi,torch.cos(theta/2)),-1))


def exp_se3(xi):
    xi=xi.float();rho,phi=xi[...,:3],xi[...,3:];theta=phi.norm(dim=-1,keepdim=True);t2=theta.square()
    a=torch.where(theta>.05,(1-torch.cos(theta))/t2.clamp_min(1e-12),.5-t2/24+t2.square()/720)
    b=torch.where(theta>.05,(theta-torch.sin(theta))/(theta*t2).clamp_min(1e-12),1/6-t2/120+t2.square()/5040)
    cross=torch.cross(phi,rho,dim=-1)
    return torch.cat((rho+a*cross+b*torch.cross(phi,cross,dim=-1),so3_exp(phi)),-1)


def log_se3(pose):
    pose=pose.float();t,phi=pose[...,:3],so3_log(pose[...,3:]);theta=phi.norm(dim=-1,keepdim=True);t2=theta.square()
    half=theta/2
    c=torch.where(theta>.05,(1-half*torch.cos(half)/torch.sin(half).clamp_min(1e-8))/t2.clamp_min(1e-12),
                  1/12+t2/720+t2.square()/30240)
    cross=torch.cross(phi,t,dim=-1)
    return torch.cat((t-.5*cross+c*torch.cross(phi,cross,dim=-1),phi),-1)


def relative(base,target):
    base,target=base.float(),target.float();q=unit(base[...,3:]);inverse=torch.cat((-q[...,:3],q[...,3:]),-1)
    return torch.cat((rotate(inverse,target[...,:3]-base[...,:3]),multiply(inverse,unit(target[...,3:]))),-1)


def compose(base,delta):
    base,delta=base.float(),delta.float();q=unit(base[...,3:])
    return torch.cat((base[...,:3]+rotate(q,delta[...,:3]),canonical(multiply(q,unit(delta[...,3:])))),-1)


def residual(base,target):
    return log_se3(relative(base,target))


def apply_residual(base,xi):
    return compose(base,exp_se3(xi.float()))


def rotation_error(a,b):
    return so3_log(multiply(torch.cat((-unit(a)[...,:3],unit(a)[...,3:]),-1),unit(b))).norm(dim=-1)


def cap_twist(xi,translation_cap,angle_cap):
    """Bound physical translation and rotation independently, then recover xi."""
    pose=exp_se3(xi);p=pose[...,:3];phi=xi[...,3:].float()
    p=p*torch.clamp(translation_cap/p.norm(dim=-1,keepdim=True).clamp_min(1e-9),max=1.)
    phi=phi*torch.clamp(angle_cap/phi.norm(dim=-1,keepdim=True).clamp_min(1e-9),max=1.)
    return log_se3(torch.cat((p,so3_exp(phi)),-1))
