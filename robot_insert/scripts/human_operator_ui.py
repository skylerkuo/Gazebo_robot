"""Focused keyboard controls and live RGB previews for human-assisted insertion."""
import base64
import io
import os
import signal
import time
import tkinter as tk
from tkinter import ttk

from PIL import Image,ImageTk

from human_control import KEY_AXES


class OperatorWindow:
    def __init__(self,state):
        self.state=state;self.root=tk.Tk()
        self.root.title('AIC 人工協助收集 — 鍵盤控制')
        self.root.geometry('850x720')
        self.mode=tk.StringVar(value='模型控制');self.status=tk.StringVar(value='等待模型與模擬器啟動…')
        self.pose=tk.StringVar();self.force=tk.StringVar();self.remaining=tk.StringVar()
        self.photos={};self.last_preview=None;self.mode_keys=set()
        header=ttk.Frame(self.root,padding=8);header.pack(fill='x')
        ttk.Label(header,textvariable=self.mode,font=('',16,'bold')).pack(side='left')
        for label,mode in [('模型 M','model'),('人工 H','human'),('保持 Space','hold')]:
            ttk.Button(header,text=label,command=lambda mode=mode:self.state.set_mode(mode)).pack(side='left',padx=5)
        ttk.Label(self.root,textvariable=self.status,wraplength=820).pack(anchor='w',padx=10)
        body=ttk.Frame(self.root,padding=8);body.pack(fill='both',expand=True)
        views=ttk.Frame(body);views.pack(side='left',anchor='n')
        self.preview={}
        for i,name in enumerate(('left','center','right','side')):
            box=ttk.LabelFrame(views,text=name);box.grid(row=i//2,column=i%2,padx=3,pady=3)
            label=ttk.Label(box,text='等待影像',width=30,anchor='center');label.pack()
            self.preview[name]=label
        controls=ttk.Frame(body,padding=8);controls.pack(side='left',fill='both')
        ttk.Label(controls,text='base_link 座標微調',font=('',12,'bold')).pack(anchor='w')
        ttk.Label(controls,text='按住移動，放開停止\nW/S：X ±　A/D：Y ±\nR/F：Z ±（上／下）\nI/K：Roll ±\nJ/L：Pitch ±\nU/O：Yaw ±',justify='left').pack(anchor='w',pady=5)
        grid=ttk.Frame(controls);grid.pack()
        pairs=[('X+ W','w','X− S','s'),('Y+ A','a','Y− D','d'),('Z+ R','r','Z− F','f'),
               ('Roll+ I','i','Roll− K','k'),('Pitch+ J','j','Pitch− L','l'),('Yaw+ U','u','Yaw− O','o')]
        for row,(a,key_a,b,key_b) in enumerate(pairs):
            for col,(label,key) in enumerate(((a,key_a),(b,key_b))):
                button=ttk.Button(grid,text=label,width=12);button.grid(row=row,column=col,padx=2,pady=2)
                button.bind('<ButtonPress-1>',lambda e,key=key:self.state.press(key))
                button.bind('<ButtonRelease-1>',lambda e,key=key:self.state.release(key))
                button.bind('<Leave>',lambda e,key=key:self.state.release(key))
        ttk.Label(controls,text='每個 20 Hz 控制步的大小').pack(anchor='w',pady=(10,0))
        self.mm=tk.StringVar(value='0.5');self.deg=tk.StringVar(value='0.5')
        for title,var in [('位移 mm',self.mm),('旋轉 °',self.deg)]:
            row=ttk.Frame(controls);row.pack(fill='x')
            ttk.Label(row,text=title).pack(side='left')
            ttk.OptionMenu(row,var,var.get(),'0.1','0.2','0.5','1.0',command=lambda _:self.steps()).pack(side='left')
        ttk.Label(controls,textvariable=self.pose,wraplength=235).pack(anchor='w',pady=8)
        ttk.Label(controls,textvariable=self.force).pack(anchor='w')
        ttk.Label(controls,textvariable=self.remaining).pack(anchor='w')
        footer=ttk.Frame(self.root,padding=8);footer.pack(fill='x')
        ttk.Button(footer,text='完成本回合／送官方評分',command=lambda:self.finish(False)).pack(side='left',padx=6)
        ttk.Button(footer,text='放棄本回合（保留原始紀錄）',command=lambda:self.finish(True)).pack(side='left',padx=6)
        ttk.Label(self.root,text='先點控制視窗再按鍵。人工模式切離視窗會保持位置。人工片段需 ≥1 秒模擬時間，才能形成未來 20 步訓練樣本。',wraplength=820).pack(padx=8,pady=5)
        self.root.bind_class('HumanKeys','<KeyPress>',self.keydown)
        self.root.bind_class('HumanKeys','<KeyRelease>',self.keyup)
        def bind_keys(widget):
            widget.bindtags(('HumanKeys',)+widget.bindtags())
            for child in widget.winfo_children():bind_keys(child)
        bind_keys(self.root)
        self.root.bind('<FocusOut>',lambda _:self.root.after_idle(self.focus_check))
        self.root.protocol('WM_DELETE_WINDOW',lambda:self.finish(True,hide=True))
        self.root.after(100,self.refresh)

    def steps(self):
        with self.state.lock:
            self.state.translation_mm=float(self.mm.get());self.state.rotation_deg=float(self.deg.get())

    def keydown(self,event):
        key=event.keysym.lower()
        if key in ('m','h','space','escape'):
            if key in self.mode_keys:return 'break'
            self.mode_keys.add(key)
        if key=='m':self.state.set_mode('model')
        elif key=='h':self.state.set_mode('human')
        elif key in ('space','escape'):self.state.set_mode('hold')
        elif key in KEY_AXES:self.state.press(key)
        else:return
        return 'break'

    def keyup(self,event):
        key=event.keysym.lower();self.mode_keys.discard(key);self.state.release(key)
        if key in KEY_AXES or key in ('m','h','space','escape'):return 'break'

    def focus_check(self):
        if self.root.focus_displayof() is None:
            self.mode_keys.clear()
            if self.state.mode=='human':self.state.set_mode('hold')

    def finish(self,discard,hide=False):
        self.state.finish_episode(discard);self.status.set('正在保持位置、結束錄製並等待官方評分…')
        if hide:self.root.withdraw()

    def refresh(self):
        with self.state.lock:
            status=dict(self.state.telemetry);mode=self.state.mode;last=self.state.last_poll
        self.mode.set({'model':'模型控制','human':'人工控制','hold':'保持位置'}[mode])
        if status:
            self.status.set(status.get('message','模型運作中'))
            xyz=status.get('actual_tcp',[])[:3]
            if len(xyz)==3:self.pose.set('TCP XYZ (m)\n'+', '.join(f'{x:.4f}' for x in xyz))
            self.force.set(f"力：{status.get('force_n',0):.2f} N　已錄：{status.get('frames',0)} 幀")
            self.remaining.set(f"剩餘模擬時間：{status.get('remaining_seconds',0):.1f} s")
            previews=status.get('previews',{})
            if status.get('preview_id')!=self.last_preview and previews:
                self.last_preview=status.get('preview_id')
                for name,encoded in previews.items():
                    picture=Image.open(io.BytesIO(base64.b64decode(encoded)));picture.thumbnail((250,222))
                    self.photos[name]=ImageTk.PhotoImage(picture)
                    self.preview[name].configure(image=self.photos[name],text='')
        if self.state.finish:self.status.set('正在結束錄製並等待官方評分，請稍候…')
        elif last is not None and time.monotonic()-last>2.:
            if mode!='hold':self.state.set_mode('hold')
            self.status.set('模型連線暫停，已切到保持位置')
        self.root.after(100,self.refresh)

    def run(self,process,timeout):
        started=time.monotonic();stopping=[]
        def check():
            if process.poll() is not None:self.root.quit();return
            if time.monotonic()-started>timeout and not stopping:
                stopping.append(True);self.state.finish_episode(True)
                os.killpg(process.pid,signal.SIGINT)
                self.root.after(15000,force_stop)
            self.root.after(200,check)
        def force_stop():
            if process.poll() is None:os.killpg(process.pid,signal.SIGKILL)
        self.root.after(200,check)
        try:self.root.mainloop()
        except KeyboardInterrupt:
            stopping.append(True)
            self.state.finish_episode(True)
            if process.poll() is None:os.killpg(process.pid,signal.SIGINT)
        finally:self.root.destroy()
        return bool(stopping)
