"""Local operator intent, TCP link, manual TCP jogging and expert-window selection."""
import json
import math
import secrets
import socket
import socketserver
import threading
import time

import numpy as np

from ftact_control import unit_pose

KEY_AXES={'w':(0,1),'s':(0,-1),'a':(1,1),'d':(1,-1),'r':(2,1),'f':(2,-1),
          'i':(3,1),'k':(3,-1),'j':(4,1),'l':(4,-1),'u':(5,1),'o':(5,-1)}


class OperatorState:
    def __init__(self):
        self.lock=threading.RLock()
        self.mode='model';self.version=0;self.keys=set()
        self.translation_mm=.5;self.rotation_deg=.5
        self.finish=False;self.discard=False;self.telemetry={};self.last_poll=None

    def set_mode(self,mode):
        if mode not in ('model','human','hold'): raise ValueError('Invalid operator mode')
        with self.lock:
            self.mode=mode;self.version+=1;self.keys.clear()

    def press(self,key):
        if key not in KEY_AXES:return
        with self.lock:
            if self.mode=='human':self.keys.add(key)

    def release(self,key):
        with self.lock:self.keys.discard(key)

    def finish_episode(self,discard=False):
        with self.lock:
            self.finish=True;self.discard=bool(discard);self.set_mode('hold')

    def snapshot(self):
        with self.lock:
            axes=np.zeros(6)
            for key in self.keys:
                axis,sign=KEY_AXES[key];axes[axis]+=sign
            return {'mode':self.mode,'version':self.version,'axes':np.clip(axes,-1,1).tolist(),
                'translation_step_m':self.translation_mm/1000,'rotation_step_rad':math.radians(self.rotation_deg),
                'finish':self.finish,'discard':self.discard}

    def update(self,status):
        with self.lock:
            self.telemetry=status;self.last_poll=time.monotonic()
            if status.get('request_hold') and self.mode!='hold':self.set_mode('hold')


class OperatorServer:
    def __init__(self,state=None):
        self.state=state or OperatorState();self.token=secrets.token_hex(24)
        owner=self
        class Handler(socketserver.StreamRequestHandler):
            def handle(self):
                self.connection.settimeout(5.)
                try:
                    while True:
                        line=self.rfile.readline(1024*1024)
                        if not line:return
                        request=json.loads(line)
                        if request.get('token')!=owner.token:return
                        owner.state.update(request.get('status',{}))
                        self.wfile.write((json.dumps(owner.state.snapshot(),allow_nan=False)+'\n').encode())
                        self.wfile.flush()
                except (OSError,ValueError):return
        class Server(socketserver.ThreadingTCPServer):
            daemon_threads=True;allow_reuse_address=False
        self.server=Server(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()

    def endpoint(self):
        return {'host':'127.0.0.1','port':self.server.server_address[1],'token':self.token}

    def close(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2.)


class OperatorLink:
    def __init__(self,endpoint):
        self.endpoint=endpoint;self.socket=None;self.stream=None;self.failed=False

    def close(self):
        if self.stream is not None:self.stream.close()
        if self.socket is not None:self.socket.close()
        self.stream=self.socket=None

    def poll(self,status):
        try:
            if self.socket is None:
                self.socket=socket.create_connection((self.endpoint['host'],self.endpoint['port']),timeout=.25)
                self.socket.settimeout(.25);self.stream=self.socket.makefile('rwb')
            self.stream.write((json.dumps({'token':self.endpoint['token'],'status':status},allow_nan=False)+'\n').encode())
            self.stream.flush()
            line=self.stream.readline(65536)
            if not line:raise ConnectionError('Operator window disconnected')
            state=json.loads(line)
            if state['mode'] not in ('model','human','hold'):raise ValueError('Invalid control mode')
            axes=np.asarray(state['axes'],dtype=float)
            if axes.shape!=(6,) or not np.isfinite(axes).all() or (np.abs(axes)>1).any():raise ValueError('Invalid jog axes')
            if not 0<state['translation_step_m']<=.002 or not 0<state['rotation_step_rad']<=math.radians(2):
                raise ValueError('Invalid jog step size')
            self.failed=False
            return state
        except (OSError,ValueError,KeyError,ConnectionError):
            self.close();self.failed=True
            return {'mode':'hold','version':-1,'axes':[0.]*6,'translation_step_m':.0005,
                    'rotation_step_rad':math.radians(.5),'finish':False,'discard':False,'link_lost':True}


def manual_pose_step(previous,control,minimum,maximum,ticks=1.):
    pose=unit_pose(previous)
    axes=np.asarray(control['axes'],dtype=float)
    ticks=float(np.clip(ticks,0.,2.))
    pose[:3]=np.clip(pose[:3]+axes[:3]*control['translation_step_m']*ticks,minimum,maximum)
    rotation=axes[3:]*control['rotation_step_rad']*ticks
    angle=np.linalg.norm(rotation)
    if angle>0:
        vector=rotation/angle*np.sin(angle/2);scalar=np.cos(angle/2)
        old=pose[3:6].copy();w=pose[6]
        # Left multiply: incremental rotations are about the base_link axes.
        pose[3:6]=scalar*old+w*vector+np.cross(vector,old)
        pose[6]=scalar*w-np.dot(vector,old)
    return unit_pose(pose)


def expert_anchor_frames(sources,segments,active,times,chunk_size=20,*,source='human'):
    sources=np.asarray(sources);segments=np.asarray(segments);active=np.asarray(active,dtype=bool)
    times=np.asarray(times,dtype=float)
    if not (sources.shape==segments.shape==active.shape==times.shape):raise ValueError('Provenance length mismatch')
    result=[]
    for frame in range(max(0,len(times)-chunk_size)):
        future=slice(frame+1,frame+chunk_size+1)
        if (np.all(sources[future]==source) and np.all(segments[future]==segments[frame+1])
                and active[future].any() and np.allclose(np.diff(times[frame:frame+chunk_size+1]),.05,rtol=0,atol=1e-5)):
            result.append(frame)
    return result
