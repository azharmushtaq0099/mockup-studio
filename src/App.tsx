import React, { useState, useRef, useEffect, useCallback } from 'react';
import { computeHomography, invertHomography, defaultCorners, type Quad } from './homography';

// ─── Shaders ──────────────────────────────────────────────────────────────────

const VERT = `
attribute vec2 aPos;
attribute vec3 aUVW;
varying vec3 vUVW;
void main() { vUVW = aUVW; gl_Position = vec4(aPos, 0.0, 1.0); }
`;

// uEdge=1 → soft feather at UV edges (eliminates hard cut lines in manual mode)
const FRAG_PLAIN = `
precision mediump float;
uniform sampler2D uTex;
uniform float uEdge;
varying vec3 vUVW;
void main() {
  vec2 uv = vUVW.xy / vUVW.z;
  vec4 c = texture2D(uTex, uv);
  if (uEdge > 0.5) {
    float f = 0.022;
    float a = smoothstep(0.0, f, uv.x)   * smoothstep(0.0, f, 1.0-uv.x) *
              smoothstep(0.0, f, uv.y)   * smoothstep(0.0, f, 1.0-uv.y);
    c.a *= a;
  }
  gl_FragColor = c;
}
`;

// Improved chroma key: combined RGB + direction distance, per-channel spill
const FRAG_CHROMA = `
precision mediump float;
uniform sampler2D uTex;
uniform vec3 uKey;
uniform float uThresh;
uniform float uSoft;
varying vec3 vUVW;
void main() {
  vec4 c = texture2D(uTex, vUVW.xy / vUVW.z);
  float cLen = max(0.001, length(c.rgb));
  float kLen = max(0.001, length(uKey));
  float cosAngle = dot(c.rgb/cLen, uKey/kLen);
  float dist = distance(c.rgb, uKey);
  float combined = dist * (1.0 + max(0.0, 1.0 - cosAngle) * 0.4);
  float alpha = smoothstep(uThresh - uSoft, uThresh + uSoft, combined);
  vec3 col = c.rgb;
  float spill = 1.0 - alpha;
  if (uKey.g >= uKey.r && uKey.g >= uKey.b)
    col.g = mix(col.g, min(col.g, max(col.r, col.b)), spill * 0.85);
  else if (uKey.b >= uKey.r && uKey.b >= uKey.g)
    col.b = mix(col.b, min(col.b, max(col.r, col.g)), spill * 0.85);
  else
    col.r = mix(col.r, min(col.r, max(col.g, col.b)), spill * 0.85);
  gl_FragColor = vec4(col, alpha);
}
`;

// Post: sharpen · contrast · sat · temp · bloom · vignette · grain
const FRAG_POST = `
precision mediump float;
uniform sampler2D uTex;
uniform float uInvW; uniform float uInvH;
uniform float uSharp;
uniform float uBright; uniform float uContrast;
uniform float uSat;
uniform float uTemp;
uniform float uVig;
uniform float uBloom;
uniform float uGrain; uniform float uTime;
varying vec3 vUVW;
void main() {
  vec2 uv = vUVW.xy / vUVW.z;
  vec2 px = vec2(uInvW, uInvH);
  vec4 c = texture2D(uTex, uv);

  if (uSharp > 0.001) {
    vec4 bl = (texture2D(uTex,uv+vec2(px.x,0.0))+texture2D(uTex,uv-vec2(px.x,0.0))+
               texture2D(uTex,uv+vec2(0.0,px.y))+texture2D(uTex,uv-vec2(0.0,px.y)))*0.25;
    c.rgb += (c.rgb - bl.rgb) * uSharp;
  }

  c.rgb = (c.rgb - 0.5) * uContrast + 0.5 + uBright;
  float lum = dot(c.rgb, vec3(0.2126,0.7152,0.0722));
  c.rgb = mix(vec3(lum), c.rgb, uSat);
  c.r += uTemp*0.15; c.b -= uTemp*0.2;

  if (uBloom > 0.001) {
    vec4 b4 = (texture2D(uTex,uv+vec2(px.x*5.0,0.0))+texture2D(uTex,uv-vec2(px.x*5.0,0.0))+
               texture2D(uTex,uv+vec2(0.0,px.y*5.0))+texture2D(uTex,uv-vec2(0.0,px.y*5.0)))*0.25;
    float bl2 = dot(b4.rgb, vec3(0.2126,0.7152,0.0722));
    c.rgb += b4.rgb * max(0.0, bl2 - 0.4) * uBloom * 3.5;
  }

  vec2 vc = uv - 0.5;
  c.rgb *= 1.0 - clamp(dot(vc,vc)/0.5,0.0,1.0) * uVig * 0.95;

  if (uGrain > 0.001) {
    vec2 sd = uv + fract(uTime * 0.1337);
    float gr = fract(sin(dot(sd,vec2(127.1,311.7)))*43758.5453) - 0.5;
    c.rgb += gr * uGrain * 0.1;
  }

  gl_FragColor = clamp(c, 0.0, 1.0);
}
`;

// ─── WebGL helpers ────────────────────────────────────────────────────────────

function mkShader(gl: WebGLRenderingContext, type: number, src: string) {
  const s = gl.createShader(type)!; gl.shaderSource(s,src); gl.compileShader(s); return s;
}
function mkProgram(gl: WebGLRenderingContext, frag: string) {
  const p = gl.createProgram()!;
  gl.attachShader(p, mkShader(gl,gl.VERTEX_SHADER,VERT));
  gl.attachShader(p, mkShader(gl,gl.FRAGMENT_SHADER,frag));
  gl.linkProgram(p); return p;
}
function mkTex(gl: WebGLRenderingContext): WebGLTexture {
  const t = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  return t;
}
function uploadTex(gl: WebGLRenderingContext, tex: WebGLTexture, src: TexImageSource) {
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
}
function drawQuad(
  gl: WebGLRenderingContext, prog: WebGLProgram, tex: WebGLTexture,
  verts: Float32Array, uniforms?: Record<string, number | number[]>,
) {
  gl.useProgram(prog);
  const buf = gl.createBuffer()!;
  gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.bufferData(gl.ARRAY_BUFFER, verts, gl.STREAM_DRAW);
  const aPos = gl.getAttribLocation(prog,'aPos'), aUVW = gl.getAttribLocation(prog,'aUVW');
  gl.enableVertexAttribArray(aPos); gl.enableVertexAttribArray(aUVW);
  gl.vertexAttribPointer(aPos,2,gl.FLOAT,false,20,0);
  gl.vertexAttribPointer(aUVW,3,gl.FLOAT,false,20,8);
  gl.uniform1i(gl.getUniformLocation(prog,'uTex'),0);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, tex);
  if (uniforms) for (const [k,v] of Object.entries(uniforms)) {
    const loc = gl.getUniformLocation(prog,k);
    if (typeof v === 'number') gl.uniform1f(loc,v);
    else if (v.length===3) gl.uniform3fv(loc,v);
    else if (v.length===2) gl.uniform2fv(loc,v);
  }
  gl.drawArrays(gl.TRIANGLES, 0, verts.length/5);
  gl.deleteBuffer(buf);
}

type FBO = { fb: WebGLFramebuffer; tex: WebGLTexture };
function createFBO(gl: WebGLRenderingContext, w: number, h: number): FBO {
  const tex = mkTex(gl);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,w,h,0,gl.RGBA,gl.UNSIGNED_BYTE,null);
  const fb = gl.createFramebuffer()!;
  gl.bindFramebuffer(gl.FRAMEBUFFER,fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,tex,0);
  gl.bindFramebuffer(gl.FRAMEBUFFER,null);
  return {fb,tex};
}
function resizeFBO(gl: WebGLRenderingContext, fbo: FBO, w: number, h: number) {
  gl.bindTexture(gl.TEXTURE_2D,fbo.tex);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,w,h,0,gl.RGBA,gl.UNSIGNED_BYTE,null);
}

// ─── Geometry ─────────────────────────────────────────────────────────────────

function bgVerts(): Float32Array {
  return new Float32Array([-1,-1,0,0,1, 1,-1,1,0,1, -1,1,0,1,1, 1,-1,1,0,1, 1,1,1,1,1, -1,1,0,1,1]);
}
function coverVerts(sw:number,sh:number,dw:number,dh:number): Float32Array {
  const sa=sw/sh,da=dw/dh; let u0=0,u1=1,v0=0,v1=1;
  if(sa>da){const m=(1-da/sa)/2;u0=m;u1=1-m;}
  else if(sa<da){const m=(1-sa/da)/2;v0=m;v1=1-m;}
  return new Float32Array([-1,-1,u0,v0,1, 1,-1,u1,v0,1, -1,1,u0,v1,1, 1,-1,u1,v0,1, 1,1,u1,v1,1, -1,1,u0,v1,1]);
}

// 32×32 subdivided mesh — ultra-smooth perspective warp, zero corner artefacts
function pinVerts(pins: Quad, W: number, H: number, N=32): Float32Array | null {
  try {
    const src: Quad = [{x:0,y:0},{x:1,y:0},{x:1,y:1},{x:0,y:1}];
    const hFwd = computeHomography(src, pins);
    const hInv = invertHomography(hFwd);
    function uv2v(u:number,v:number): number[]|null {
      const dw=hFwd[6]*u+hFwd[7]*v+hFwd[8]; if(Math.abs(dw)<1e-10) return null;
      const sx=(hFwd[0]*u+hFwd[1]*v+hFwd[2])/dw, sy=(hFwd[3]*u+hFwd[4]*v+hFwd[5])/dw;
      const w=hInv[6]*sx+hInv[7]*sy+hInv[8]; if(!isFinite(w)||Math.abs(w)<1e-10) return null;
      return [(sx/W)*2-1, 1-(sy/H)*2, u*w, v*w, w];
    }
    const out: number[]=[];
    for(let r=0;r<N;r++) for(let c=0;c<N;c++){
      const u0=c/N,u1=(c+1)/N,v0=r/N,v1=(r+1)/N;
      const p00=uv2v(u0,v0),p10=uv2v(u1,v0),p01=uv2v(u0,v1),p11=uv2v(u1,v1);
      if(!p00||!p10||!p01||!p11) return null;
      out.push(...p00,...p10,...p01, ...p10,...p11,...p01);
    }
    return new Float32Array(out);
  } catch { return null; }
}

// ─── Color utils ─────────────────────────────────────────────────────────────

// Sample center 60% region of mockup — avoids device frame, better key detection
function detectKeyColor(img: HTMLImageElement): string {
  const W=Math.min(img.naturalWidth,300),H=Math.min(img.naturalHeight,300);
  const c=document.createElement('canvas'); c.width=W; c.height=H;
  const ctx=c.getContext('2d')!; ctx.drawImage(img,0,0,W,H);
  const x0=Math.floor(W*0.2),y0=Math.floor(H*0.2),sw=Math.floor(W*0.6),sh=Math.floor(H*0.6);
  const d=ctx.getImageData(x0,y0,sw,sh).data;
  const hist: Record<string,number>={};
  for(let i=0;i<d.length;i+=4){
    const r=d[i],g=d[i+1],b=d[i+2];
    const max=Math.max(r,g,b),min=Math.min(r,g,b);
    if(max<55||max>245||(max-min)/max<0.3) continue;
    const k=`${Math.round(r/32)},${Math.round(g/32)},${Math.round(b/32)}`;
    hist[k]=(hist[k]||0)+1;
  }
  const best=Object.entries(hist).sort((a,b)=>b[1]-a[1])[0];
  if(!best) return '#00ff00';
  const [qr,qg,qb]=best[0].split(',').map(n=>Math.min(255,Number(n)*32));
  return `#${qr.toString(16).padStart(2,'0')}${qg.toString(16).padStart(2,'0')}${qb.toString(16).padStart(2,'0')}`;
}
function hexToRgb(h:string):[number,number,number]{
  const v=parseInt(h.slice(1),16); return [(v>>16&255)/255,(v>>8&255)/255,(v&255)/255];
}
function dl(blob:Blob,name:string){
  const url=URL.createObjectURL(blob),a=document.createElement('a');
  a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),5000);
}

// ─── Types & presets ──────────────────────────────────────────────────────────

type Enhance={sharp:number;bright:number;contrast:number;sat:number;vignette:number;temp:number;grain:number;bloom:number};
type GradeName='none'|'natural'|'cinematic'|'vivid';
const GRADES:Record<GradeName,Enhance>={
  none:      {sharp:0,    bright:0,     contrast:1.0,  sat:1.0,  vignette:0,    temp:0,     grain:0,    bloom:0   },
  natural:   {sharp:0.45, bright:0.018, contrast:1.06, sat:0.95, vignette:0.12, temp:0.05,  grain:0.08, bloom:0   },
  cinematic: {sharp:0.35, bright:-0.02, contrast:1.18, sat:0.72, vignette:0.42, temp:-0.08, grain:0.28, bloom:0.1 },
  vivid:     {sharp:0.65, bright:0.02,  contrast:1.24, sat:1.32, vignette:0.14, temp:0.04,  grain:0.04, bloom:0.08},
};

type Preset={name:string;mode:Mode;grade:GradeName;enhance:Enhance;keyColor:string;keyThresh:number;keySoft:number};
type Mode='manual'|'auto';
type ExportFmt='png'|'webm';
type Quality='high'|'ultra';

// ─── Library ─────────────────────────────────────────────────────────────────

interface UPhoto{id:string;urls:{thumb:string;regular:string};alt_description:string;user:{name:string}}

const LIB_CATS=[
  {label:'💻 MacBook',   q:'macbook mockup clean desk'},
  {label:'📱 iPhone',    q:'iphone smartphone mockup minimal'},
  {label:'⬛ iPad',      q:'ipad tablet mockup flat lay'},
  {label:'🖥️ iMac',     q:'imac desktop workspace minimal'},
  {label:'📐 Workspace', q:'clean minimal desk setup laptop'},
  {label:'🎨 Gradient',  q:'gradient abstract tech background'},
];

function Library({onSelectPhoto}:{onSelectPhoto:(url:string,name:string)=>void}){
  const [apiKey,setApiKey]=useState(()=>localStorage.getItem('unsplash-key')||'');
  const [keyInput,setKeyInput]=useState(apiKey);
  const [catIdx,setCatIdx]=useState(0);
  const [query,setQuery]=useState('');
  const [results,setResults]=useState<UPhoto[]>([]);
  const [loading,setLoading]=useState(false);

  const search=useCallback(async(q:string)=>{
    const k=localStorage.getItem('unsplash-key'); if(!k) return;
    setLoading(true);
    try{
      const res=await fetch(`https://api.unsplash.com/search/photos?query=${encodeURIComponent(q)}&per_page=30&orientation=landscape&client_id=${k}`);
      const data=await res.json(); setResults(data.results||[]);
    }catch{setResults([]);}
    setLoading(false);
  },[]);

  useEffect(()=>{if(apiKey) search(query||LIB_CATS[catIdx].q);},[apiKey,catIdx]);

  const saveKey=(k:string)=>{localStorage.setItem('unsplash-key',k);setApiKey(k);};

  if(!apiKey) return(
    <div style={{padding:'16px 14px'}}>
      <div style={{fontSize:13,fontWeight:700,marginBottom:8,color:'var(--text)'}}>Connect Unsplash</div>
      <p style={{fontSize:10.5,color:'var(--muted)',lineHeight:1.75,marginBottom:10}}>
        Browse thousands of real device photos. Get a free key at{' '}
        <a href="https://unsplash.com/developers" target="_blank" rel="noreferrer"
           style={{color:'var(--blue)'}}>unsplash.com/developers</a>
        {' '}— takes 2 minutes.
      </p>
      <input style={{width:'100%',padding:'7px 10px',background:'var(--surface)',border:'1px solid var(--border)',
        borderRadius:7,fontSize:11,color:'var(--text)',outline:'none',marginBottom:8,boxSizing:'border-box'}}
        placeholder="Paste Unsplash Access Key…"
        value={keyInput} onChange={e=>setKeyInput(e.target.value)}
        onKeyDown={e=>{if(e.key==='Enter'&&keyInput.trim()) saveKey(keyInput.trim());}}/>
      <button style={{width:'100%',padding:9,background:'var(--accent)',color:'#fff',border:'none',
        borderRadius:7,fontWeight:700,fontSize:12,cursor:'pointer'}}
        onClick={()=>keyInput.trim()&&saveKey(keyInput.trim())}>Connect →</button>
      <div style={{marginTop:14,padding:'10px 12px',background:'var(--surface)',borderRadius:8,
        border:'1px solid var(--border)',fontSize:10,color:'var(--muted)',lineHeight:2}}>
        <div>① unsplash.com/developers → New Application</div>
        <div>② Accept terms, fill basic details</div>
        <div>③ Copy the Access Key and paste above</div>
      </div>
    </div>
  );

  return(<>
    <div style={{display:'flex',flexWrap:'wrap',gap:4,padding:'8px 14px',borderBottom:'1px solid var(--border)'}}>
      {LIB_CATS.map((c,i)=>(
        <div key={i} onClick={()=>{setCatIdx(i);search(c.q);}}
          style={{padding:'4px 9px',borderRadius:14,fontSize:10,fontWeight:600,cursor:'pointer',
            border:`1.5px solid ${catIdx===i?'var(--accent)':'var(--border)'}`,
            background:catIdx===i?'rgba(124,106,247,.12)':'var(--surface)',
            color:catIdx===i?'var(--text)':'var(--muted)',whiteSpace:'nowrap'}}>
          {c.label}
        </div>
      ))}
    </div>
    <div style={{display:'flex',gap:6,padding:'8px 14px',borderBottom:'1px solid var(--border)'}}>
      <input style={{flex:1,padding:'6px 9px',background:'var(--surface)',border:'1px solid var(--border)',
        borderRadius:7,fontSize:11,color:'var(--text)',outline:'none'}}
        placeholder="Search…" value={query} onChange={e=>setQuery(e.target.value)}
        onKeyDown={e=>{if(e.key==='Enter'&&query.trim()) search(query.trim());}}/>
      <button style={{padding:'6px 12px',background:'var(--accent)',color:'#fff',border:'none',
        borderRadius:7,fontSize:11,fontWeight:600,cursor:'pointer'}}
        onClick={()=>query.trim()&&search(query.trim())}>→</button>
    </div>
    {loading&&<div style={{padding:'20px',textAlign:'center',color:'var(--muted)',fontSize:11}}>Loading…</div>}
    {!loading&&results.length===0&&<div style={{padding:'20px',textAlign:'center',color:'var(--muted)',fontSize:11,lineHeight:1.7}}>
      No results — try a different search.
    </div>}
    {!loading&&results.length>0&&(
      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:6,padding:'10px 14px',overflowY:'auto'}}>
        {results.map(p=>(
          <div key={p.id} onClick={()=>onSelectPhoto(p.urls.regular+'&w=2560&q=90',p.alt_description||'Unsplash')}
            style={{borderRadius:8,overflow:'hidden',cursor:'pointer',aspectRatio:'4/3',
              background:'var(--surface)',border:'1.5px solid var(--border)',
              transition:'all .15s',position:'relative'}}
            onMouseEnter={e=>{(e.currentTarget as HTMLElement).style.borderColor='var(--accent)';(e.currentTarget as HTMLElement).style.transform='scale(1.03)';}}
            onMouseLeave={e=>{(e.currentTarget as HTMLElement).style.borderColor='var(--border)';(e.currentTarget as HTMLElement).style.transform='scale(1)';}}>
            <img src={p.urls.thumb} alt={p.alt_description||''} loading="lazy"
              style={{width:'100%',height:'100%',objectFit:'cover',display:'block'}}/>
          </div>
        ))}
      </div>
    )}
    <div style={{padding:'6px 14px 10px',fontSize:9,color:'#383858',borderTop:'1px solid var(--border)',
      display:'flex',justifyContent:'space-between'}}>
      <span>Photos via Unsplash</span>
      <span style={{cursor:'pointer',textDecoration:'underline'}}
        onClick={()=>{localStorage.removeItem('unsplash-key');setApiKey('');setKeyInput('');}}>
        Disconnect
      </span>
    </div>
  </>);
}

// ─── CSS ─────────────────────────────────────────────────────────────────────

const CSS = `
:root{--bg:#07070F;--panel:#0D0D18;--surface:#12121C;--border:#1C1C2E;
  --accent:#7C6AF7;--blue:#5B9CF6;--text:#EEEEF5;--muted:#5A5A80}
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,#root{height:100%;overflow:hidden}
body{font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--text);font-size:13px}
.app{display:flex;flex-direction:column;height:100vh}

.hdr{display:flex;align-items:center;justify-content:space-between;
  height:48px;padding:0 18px;background:var(--panel);
  border-bottom:1px solid var(--border);flex-shrink:0;
  box-shadow:0 1px 0 rgba(255,255,255,.04)}
.logo{display:flex;align-items:center;gap:9px;font-weight:700;font-size:13.5px;letter-spacing:-.4px}
.logo-gem{width:20px;height:20px;background:linear-gradient(135deg,#7C6AF7,#5B9CF6);border-radius:5px;
  box-shadow:0 2px 10px rgba(124,106,247,.4)}
.hdr-r{display:flex;align-items:center;gap:8px}

.btn{display:inline-flex;align-items:center;gap:5px;padding:6px 13px;
  border-radius:7px;font-size:11.5px;font-weight:500;border:none;cursor:pointer;
  transition:all .13s;white-space:nowrap}
.btn-ghost{background:transparent;color:var(--muted);border:1px solid var(--border)}
.btn-ghost:hover:not(:disabled){background:var(--surface);color:var(--text)}
.btn-danger{background:rgba(239,68,68,.1);color:#F87171;border:1px solid rgba(239,68,68,.25)}
.btn-danger:hover{background:rgba(239,68,68,.2)}
.btn-export{background:linear-gradient(130deg,#7C6AF7,#5B9CF6);color:#fff;font-weight:700;font-size:12px;
  box-shadow:0 4px 16px rgba(124,106,247,.35)}
.btn-export:hover:not(:disabled){opacity:.9;transform:translateY(-1px);box-shadow:0 6px 20px rgba(124,106,247,.5)}
.btn:disabled{opacity:.3;cursor:not-allowed}

.rec-badge{display:flex;align-items:center;gap:6px;padding:4px 11px;
  background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.35);
  border-radius:18px;font-size:11px;color:#F87171;font-weight:600}
.rec-dot{width:6px;height:6px;border-radius:50%;background:#EF4444;animation:blink 1s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.25}}

.body{display:flex;flex:1;overflow:hidden;min-height:0}
.sidebar{width:260px;flex-shrink:0;overflow-y:auto;background:var(--panel);
  border-right:1px solid var(--border);display:flex;flex-direction:column}
.sidebar::-webkit-scrollbar{width:3px}
.sidebar::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}

.top-tabs{display:flex;border-bottom:1px solid var(--border);flex-shrink:0}
.top-tab{flex:1;padding:10px;font-size:11px;font-weight:600;border:none;
  background:transparent;color:var(--muted);cursor:pointer;transition:all .13s;
  border-bottom:2px solid transparent}
.top-tab.active{color:var(--text);border-bottom-color:var(--accent);background:rgba(124,106,247,.06)}

.sec{padding:11px 14px;border-bottom:1px solid var(--border)}
.sec-title{font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--muted);margin-bottom:9px}

.mode-row{display:flex;gap:5px;padding:9px 14px;border-bottom:1px solid var(--border)}
.mode-btn{flex:1;padding:6px 8px;border-radius:7px;font-size:11px;font-weight:600;
  border:1.5px solid var(--border);background:transparent;color:var(--muted);cursor:pointer;transition:all .13s}
.mode-btn.active{border-color:var(--accent);background:rgba(124,106,247,.1);color:var(--text)}

.drop{border:1.5px dashed var(--border);border-radius:9px;padding:11px 10px;
  text-align:center;cursor:pointer;transition:all .18s;background:var(--surface)}
.drop:hover,.drop.over{border-color:var(--accent);background:rgba(124,106,247,.05)}
.drop-ico{font-size:17px;margin-bottom:4px}
.drop-tx{font-size:10.5px;color:var(--muted);line-height:1.5}
.drop-tx strong{display:block;color:var(--text);font-size:11px;margin-bottom:2px}
.chip{margin-top:5px;padding:3px 8px;border-radius:5px;font-size:9px;font-weight:500;
  background:rgba(91,156,246,.1);color:#5B9CF6;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

.grade-row{display:flex;gap:4px;flex-wrap:wrap;margin-bottom:9px}
.grade-btn{padding:4px 10px;border-radius:16px;font-size:10px;font-weight:600;
  border:1.5px solid var(--border);background:var(--surface);color:var(--muted);cursor:pointer;transition:all .13s}
.grade-btn.active{border-color:var(--accent);background:rgba(124,106,247,.14);color:var(--text)}

.sl-row{display:flex;flex-direction:column;gap:4px;margin-bottom:7px}
.sl-lbl{display:flex;justify-content:space-between;font-size:10.5px;color:var(--muted)}
.sl-lbl span{color:var(--text);font-variant-numeric:tabular-nums}
input[type=range]{width:100%;accent-color:var(--accent);cursor:pointer;margin:1px 0}

.expand-row{display:flex;align-items:center;justify-content:space-between;cursor:pointer;margin-bottom:7px}
.expand-lbl{font-size:9px;font-weight:700;color:var(--muted);letter-spacing:.8px;text-transform:uppercase}
.expand-arrow{font-size:9px;color:var(--muted);transition:transform .15s;user-select:none}
.expand-arrow.open{transform:rotate(180deg)}

input[type=color]{width:32px;height:32px;border-radius:6px;border:1.5px solid var(--border);
  padding:2px;cursor:pointer;background:var(--surface);flex-shrink:0}

.trim-wrap{position:relative;height:18px;margin:6px 0}
.trim-track{position:absolute;top:50%;left:0;right:0;height:3px;
  background:var(--border);border-radius:2px;transform:translateY(-50%)}
.trim-fill{position:absolute;top:50%;height:3px;background:var(--accent);
  border-radius:2px;transform:translateY(-50%)}
.trim-wrap input[type=range]{position:absolute;width:100%;top:0;height:100%;
  opacity:0;cursor:pointer;margin:0;pointer-events:all}
.trim-wrap input[type=range]:first-of-type{z-index:2}

.preset-list{display:flex;flex-direction:column;gap:4px;margin-bottom:8px}
.preset-item{display:flex;align-items:center;justify-content:space-between;
  padding:5px 8px;border-radius:7px;background:var(--surface);
  border:1px solid var(--border);cursor:pointer;transition:border-color .13s}
.preset-item:hover{border-color:var(--accent)}
.preset-item span{font-size:11px;color:var(--text)}
.preset-item button{font-size:10px;color:var(--muted);background:none;border:none;cursor:pointer;padding:0 2px}
.preset-input{flex:1;padding:5px 8px;background:var(--surface);border:1px solid var(--border);
  border-radius:7px;font-size:11px;color:var(--text);outline:none}
.preset-input::placeholder{color:var(--muted)}
.preset-input:focus{border-color:var(--accent)}

.canvas-area{flex:1;display:flex;align-items:center;justify-content:center;overflow:hidden;
  background:radial-gradient(ellipse at center, #0E0E1C 0%, #07070F 100%)}
.canvas-wrap{position:relative;
  box-shadow:0 32px 100px rgba(0,0,0,.9),0 12px 36px rgba(0,0,0,.7),0 0 0 1px rgba(255,255,255,.05)}
.pin-handle{position:absolute;width:11px;height:11px;border-radius:50%;
  background:rgba(124,106,247,.9);border:2px solid rgba(255,255,255,.85);
  transform:translate(-50%,-50%);cursor:grab;z-index:10;user-select:none;
  touch-action:none;transition:all .12s}
.pin-handle:hover{transform:translate(-50%,-50%) scale(1.5)}
.pin-handle.active{background:var(--blue);cursor:grabbing;transform:translate(-50%,-50%) scale(1.7);
  box-shadow:0 0 0 3px rgba(91,156,246,.4)}
.pin-lbl{position:absolute;top:-14px;left:50%;transform:translateX(-50%);
  font-size:8px;font-weight:700;color:rgba(255,255,255,.35);white-space:nowrap;pointer-events:none}
.empty{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:10px;pointer-events:none;color:var(--muted)}
.empty-ico{font-size:44px;opacity:.3}
.empty h3{font-size:15px;font-weight:600;color:rgba(238,238,245,.55)}
.empty p{font-size:11px;text-align:center;max-width:240px;line-height:1.75}

.modal-ov{position:fixed;inset:0;background:rgba(0,0,0,.8);
  display:flex;align-items:center;justify-content:center;z-index:200;animation:fin .14s}
.modal{background:var(--panel);border:1px solid var(--border);border-radius:14px;
  width:340px;overflow:hidden;box-shadow:0 32px 80px rgba(0,0,0,.75)}
.modal-hdr{display:flex;align-items:center;justify-content:space-between;
  padding:14px 18px;border-bottom:1px solid var(--border)}
.modal-hdr h2{font-size:14px;font-weight:700}
.modal-x{background:none;border:none;color:var(--muted);cursor:pointer;font-size:16px;padding:2px 6px}
.modal-x:hover{color:var(--text)}
.modal-body{padding:18px}
.m-stat{display:flex;justify-content:space-between;margin-bottom:10px;font-size:11.5px;color:var(--muted)}
.m-stat strong{color:var(--text)}
.m-lbl{font-size:9px;color:var(--muted);letter-spacing:.8px;text-transform:uppercase;font-weight:700;margin-bottom:7px;margin-top:12px}
.fmt-tabs{display:flex;gap:5px;margin-bottom:4px}
.fmt-tab{flex:1;padding:9px;border-radius:7px;font-size:11px;font-weight:600;
  border:1.5px solid var(--border);background:var(--surface);color:var(--muted);cursor:pointer;transition:all .13s;text-align:center}
.fmt-tab.active{border-color:var(--accent);background:rgba(124,106,247,.1);color:var(--text)}
.q-row{display:flex;gap:5px;margin-bottom:9px}
.q-btn{flex:1;padding:8px;border-radius:7px;font-size:10.5px;font-weight:600;
  border:1.5px solid var(--border);background:var(--surface);color:var(--muted);cursor:pointer;transition:all .13s;text-align:center;line-height:1.7}
.q-btn.active{border-color:var(--accent);background:rgba(124,106,247,.1);color:var(--text)}
.q-note{font-size:10px;color:var(--muted);margin-bottom:12px;line-height:1.7}

.toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);
  background:#1A1A2E;border:1px solid var(--border);border-radius:9px;
  padding:9px 18px;font-size:12px;box-shadow:0 12px 40px rgba(0,0,0,.65);z-index:999;
  animation:fin .18s}
.toast.err{border-color:#F87171;color:#F87171}
@keyframes fin{from{opacity:0;transform:translateX(-50%) translateY(6px)}to{opacity:1;transform:translateX(-50%) translateY(0)}}
`;

// ─── DropZone ────────────────────────────────────────────────────────────────

function DropZone({label,hint,file,onFile,accept,icon}:{
  label:string;hint:string;file:string;onFile:(f:File)=>void;accept:string;icon:string;
}){
  const [over,setOver]=useState(false);
  const open=()=>{const i=document.createElement('input');i.type='file';i.accept=accept;
    i.onchange=e=>{const f=(e.target as HTMLInputElement).files?.[0];if(f)onFile(f);};i.click();};
  return(
    <div className={`drop${over?' over':''}`}
      onDragOver={e=>{e.preventDefault();setOver(true)}} onDragLeave={()=>setOver(false)}
      onDrop={e=>{e.preventDefault();setOver(false);const f=e.dataTransfer.files[0];if(f)onFile(f)}}
      onClick={open}>
      <div className="drop-ico">{icon}</div>
      <div className="drop-tx"><strong>{file?'Replace file':label}</strong>{hint}</div>
      {file&&<div className="chip">{file}</div>}
    </div>
  );
}

function Slider({label,min,max,step,value,onChange}:{
  label:string;min:number;max:number;step:number;value:number;onChange:(v:number)=>void;
}){
  return(
    <div className="sl-row">
      <div className="sl-lbl">{label}<span>{value.toFixed(2)}</span></div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={e=>onChange(Number(e.target.value))}/>
    </div>
  );
}

// ─── App ─────────────────────────────────────────────────────────────────────

export default function App(){
  const [tab,       setTab]      = useState<'edit'|'library'>('edit');
  const [mode,      setMode]     = useState<Mode>('manual');
  const [mockupSrc, setMockupSrc]= useState<string|null>(null);
  const [mockupIsV, setMockupIsV]= useState(false);
  const [recSrc,    setRecSrc]   = useState<string|null>(null);
  const [recNative, setRecNative]= useState({w:1920,h:1080});
  const [csz,       setCsz]      = useState({w:840,h:520});
  const [native,    setNative]   = useState({w:840,h:520});
  const [pins,      setPins]     = useState<Quad>(()=>defaultCorners(840,520));
  const [keyColor,  setKeyColor] = useState('#00ff00');
  const [keyThresh, setKeyThresh]= useState(0.38);
  const [keySoft,   setKeySoft]  = useState(0.14);
  const [grade,     setGrade]    = useState<GradeName>('natural');
  const [enhance,   setEnhance]  = useState<Enhance>(GRADES.natural);
  const [expandEnh, setExpandEnh]= useState(false);
  const [trimIn,    setTrimIn]   = useState(0);
  const [trimOut,   setTrimOut]  = useState(1);
  const [recDur,    setRecDur]   = useState(0);
  const [presets,   setPresets]  = useState<Preset[]>(()=>{try{return JSON.parse(localStorage.getItem('mockup-presets')||'[]');}catch{return [];}});
  const [presetName,setPresetName]=useState('');
  const [showPre,   setShowPre]  = useState(false);
  const [mockupFile,setMockupFile]=useState('');
  const [recFile,   setRecFile]  = useState('');
  const [activePin, setActivePin]= useState<number|null>(null);
  const [showExp,   setShowExp]  = useState(false);
  const [expFmt,    setExpFmt]   = useState<ExportFmt>('png');
  const [quality,   setQuality]  = useState<Quality>('high');
  const [isRec,     setIsRec]    = useState(false);
  const [recTime,   setRecTime]  = useState(0);
  const [toast,     setToast]    = useState<{msg:string;err?:boolean}|null>(null);

  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const glRef       = useRef<WebGLRenderingContext|null>(null);
  const plainRef    = useRef<WebGLProgram|null>(null);
  const chromaRef   = useRef<WebGLProgram|null>(null);
  const postRef     = useRef<WebGLProgram|null>(null);
  const fboRef      = useRef<FBO|null>(null);
  const mTexRef     = useRef<WebGLTexture|null>(null);
  const rTexRef     = useRef<WebGLTexture|null>(null);
  const mockupVidRef= useRef<HTMLVideoElement>(null);
  const recVidRef   = useRef<HTMLVideoElement>(null);

  const mReadyRef   = useRef(false);
  const rReadyRef   = useRef(false);
  const mIsVRef     = useRef(false);
  const rStaticRef  = useRef(false);
  const pinsRef     = useRef<Quad>(pins);
  const cszRef      = useRef(csz);
  const recNatRef   = useRef({w:1920,h:1080});
  const modeRef     = useRef<Mode>('manual');
  const keyClrRef   = useRef<[number,number,number]>([0,1,0]);
  const keyTRef     = useRef(0.38);
  const keySRef     = useRef(0.14);
  const enhRef      = useRef<Enhance>(GRADES.natural);
  const trimInRef   = useRef(0);
  const trimOutRef  = useRef(1);
  const rafRef      = useRef(0);
  const recorderRef = useRef<MediaRecorder|null>(null);
  const chunksRef   = useRef<Blob[]>([]);
  const timerRef    = useRef<ReturnType<typeof setInterval>|null>(null);

  useEffect(()=>{pinsRef.current=pins},[pins]);
  useEffect(()=>{cszRef.current=csz},[csz]);
  useEffect(()=>{recNatRef.current=recNative},[recNative]);
  useEffect(()=>{modeRef.current=mode},[mode]);
  useEffect(()=>{keyClrRef.current=hexToRgb(keyColor)},[keyColor]);
  useEffect(()=>{keyTRef.current=keyThresh},[keyThresh]);
  useEffect(()=>{keySRef.current=keySoft},[keySoft]);
  useEffect(()=>{enhRef.current=enhance},[enhance]);
  useEffect(()=>{trimInRef.current=trimIn},[trimIn]);
  useEffect(()=>{trimOutRef.current=trimOut},[trimOut]);

  // ── Init WebGL ──────────────────────────────────────────────────────────────
  useEffect(()=>{
    const canvas=canvasRef.current!;
    // antialias:true — hardware MSAA eliminates mesh boundary jagging
    const gl=canvas.getContext('webgl',{preserveDrawingBuffer:true,alpha:false,antialias:true});
    if(!gl) return;
    glRef.current=gl;
    plainRef.current =mkProgram(gl,FRAG_PLAIN);
    chromaRef.current=mkProgram(gl,FRAG_CHROMA);
    postRef.current  =mkProgram(gl,FRAG_POST);
    mTexRef.current  =mkTex(gl); rTexRef.current=mkTex(gl);
    fboRef.current   =createFBO(gl,canvas.width,canvas.height);
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);

    function frame(){
      const gl=glRef.current!,plain=plainRef.current!,chroma=chromaRef.current!,
            post=postRef.current!,fbo=fboRef.current!,mt=mTexRef.current!,rt=rTexRef.current!;
      const mVid=mockupVidRef.current,rVid=recVidRef.current;
      const W=canvas.width,H=canvas.height;
      const {w:dW,h:dH}=cszRef.current,{w:rW,h:rH}=recNatRef.current;
      const e=enhRef.current,t=performance.now()*0.001;

      gl.bindFramebuffer(gl.FRAMEBUFFER,fbo.fb);
      gl.viewport(0,0,W,H); gl.clearColor(0.02,0.02,0.055,1); gl.clear(gl.COLOR_BUFFER_BIT);

      if(modeRef.current==='manual'){
        if(mReadyRef.current){
          if(mIsVRef.current&&mVid&&mVid.readyState>=2) uploadTex(gl,mt,mVid);
          drawQuad(gl,plain,mt,bgVerts(),{uEdge:0});
        }
        if(rReadyRef.current){
          if(!rStaticRef.current&&rVid&&rVid.readyState>=2) uploadTex(gl,rt,rVid);
          const sx=dW>0?W/dW:1,sy=dH>0?H/dH:1;
          const np=pinsRef.current.map(p=>({x:p.x*sx,y:p.y*sy})) as Quad;
          const vt=pinVerts(np,W,H);
          if(vt) drawQuad(gl,plain,rt,vt,{uEdge:1}); // soft feathered edges
        }
      } else {
        if(rReadyRef.current){
          if(!rStaticRef.current&&rVid&&rVid.readyState>=2) uploadTex(gl,rt,rVid);
          drawQuad(gl,plain,rt,coverVerts(rW,rH,W,H),{uEdge:0});
        }
        if(mReadyRef.current){
          if(mIsVRef.current&&mVid&&mVid.readyState>=2) uploadTex(gl,mt,mVid);
          drawQuad(gl,chroma,mt,bgVerts(),{uKey:keyClrRef.current,uThresh:keyTRef.current,uSoft:keySRef.current});
        }
      }

      gl.bindFramebuffer(gl.FRAMEBUFFER,null); gl.viewport(0,0,W,H);
      drawQuad(gl,post,fbo.tex,bgVerts(),{
        uInvW:1/W,uInvH:1/H,uSharp:e.sharp,uBright:e.bright,uContrast:e.contrast,
        uSat:e.sat,uTemp:e.temp,uVig:e.vignette,uBloom:e.bloom,uGrain:e.grain,uTime:t,
      });

      rafRef.current=requestAnimationFrame(frame);
    }
    rafRef.current=requestAnimationFrame(frame);
    return()=>cancelAnimationFrame(rafRef.current);
  },[]);

  useEffect(()=>{
    const c=canvasRef.current; if(!c) return;
    c.width=native.w; c.height=native.h;
    const gl=glRef.current,fbo=fboRef.current;
    if(gl&&fbo) resizeFBO(gl,fbo,native.w,native.h);
  },[native]);

  function loadMockupMedia(src:string,isVid:boolean){
    mReadyRef.current=false; mIsVRef.current=isVid;
    if(isVid){
      const vid=mockupVidRef.current; if(!vid) return;
      vid.src=src; vid.loop=true; vid.muted=true; vid.playsInline=true;
      vid.oncanplay=()=>{
        applyMockupSize(vid.videoWidth||1920,vid.videoHeight||1080);
        uploadTex(glRef.current!,mTexRef.current!,vid);
        mReadyRef.current=true; vid.play().catch(()=>{});
      };
      vid.load();
    } else {
      const img=new Image(); img.crossOrigin='anonymous';
      img.onload=()=>{
        const gl=glRef.current; if(!gl||!mTexRef.current) return;
        uploadTex(gl,mTexRef.current,img); mReadyRef.current=true;
        applyMockupSize(img.naturalWidth,img.naturalHeight);
        if(modeRef.current==='auto') setKeyColor(detectKeyColor(img));
      };
      img.src=src;
    }
  }

  useEffect(()=>{if(mockupSrc) loadMockupMedia(mockupSrc,mockupIsV);},[mockupSrc,mockupIsV]);

  function applyMockupSize(nW:number,nH:number){
    const avW=window.innerWidth-280,avH=window.innerHeight-80;
    const scale=Math.min(avW/nW,avH/nH,1);
    const dW=Math.round(nW*scale),dH=Math.round(nH*scale);
    setNative({w:nW,h:nH}); setCsz({w:dW,h:dH}); setPins(defaultCorners(dW,dH));
  }

  useEffect(()=>{
    const vid=recVidRef.current; if(!vid||!recSrc) return;
    rReadyRef.current=false; rStaticRef.current=false;
    vid.src=recSrc; vid.loop=false; vid.muted=true; vid.playsInline=true;
    vid.onloadedmetadata=()=>{setRecNative({w:vid.videoWidth||1920,h:vid.videoHeight||1080});setRecDur(vid.duration||0);};
    vid.oncanplay=()=>{rReadyRef.current=true;vid.play().catch(()=>{});};
    vid.ontimeupdate=()=>{
      const dur=vid.duration; if(!dur) return;
      const out=trimOutRef.current,inP=trimInRef.current;
      if(vid.currentTime>=dur*out) vid.currentTime=dur*inP;
      if(vid.currentTime<dur*inP)  vid.currentTime=dur*inP;
    };
    vid.load();
  },[recSrc]);

  const loadMockup=useCallback((file:File)=>{
    const isV=file.type.startsWith('video/')||/\.(mp4|webm|mov|mkv)$/i.test(file.name);
    setMockupIsV(isV); setMockupSrc(URL.createObjectURL(file)); setMockupFile(file.name);
  },[]);

  const loadRec=useCallback((file:File)=>{
    setRecFile(file.name);
    if(file.type.startsWith('image/')){
      rStaticRef.current=true; rReadyRef.current=false;
      const img=new Image(); img.onload=()=>{
        if(glRef.current&&rTexRef.current){uploadTex(glRef.current,rTexRef.current,img);rReadyRef.current=true;}
        setRecNative({w:img.naturalWidth,h:img.naturalHeight});
      };
      img.src=URL.createObjectURL(file);
    } else { setRecSrc(URL.createObjectURL(file)); }
  },[]);

  const loadMockupFromUrl=useCallback((url:string,name:string)=>{
    setMockupIsV(false); setMockupFile(name); setMockupSrc(url); setTab('edit');
  },[]);

  const onPinDown=useCallback((i:number)=>(e:React.PointerEvent)=>{
    e.preventDefault();e.stopPropagation();setActivePin(i);
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  },[]);
  const onMove=useCallback((e:React.PointerEvent)=>{
    if(activePin===null) return;
    const rect=canvasRef.current!.getBoundingClientRect();
    const {w,h}=cszRef.current;
    setPins(prev=>{const n=[...prev] as Quad;n[activePin]={x:Math.max(0,Math.min(w,e.clientX-rect.left)),y:Math.max(0,Math.min(h,e.clientY-rect.top))};return n;});
  },[activePin]);
  const onUp=useCallback(()=>setActivePin(null),[]);

  const applyGrade=useCallback((g:GradeName)=>{setGrade(g);setEnhance(GRADES[g]);},[]);

  const savePreset=useCallback(()=>{
    if(!presetName.trim()) return;
    const p:Preset={name:presetName.trim(),mode,grade,enhance,keyColor,keyThresh,keySoft};
    const updated=[...presets,p]; setPresets(updated);
    localStorage.setItem('mockup-presets',JSON.stringify(updated));
    setPresetName(''); showToast(`"${p.name}" saved`);
  },[presetName,mode,grade,enhance,keyColor,keyThresh,keySoft,presets]);

  const loadPreset=useCallback((p:Preset)=>{
    setMode(p.mode);setGrade(p.grade);setEnhance(p.enhance);
    setKeyColor(p.keyColor);setKeyThresh(p.keyThresh);setKeySoft(p.keySoft);
    showToast(`Loaded "${p.name}"`);
  },[]);

  const deletePreset=useCallback((i:number)=>{
    const updated=presets.filter((_,idx)=>idx!==i);
    setPresets(updated); localStorage.setItem('mockup-presets',JSON.stringify(updated));
  },[presets]);

  const doExportPNG=useCallback(()=>{
    canvasRef.current?.toBlob(b=>{
      if(b){dl(b,'mockup.png');showToast('PNG saved!');setShowExp(false);}
      else showToast('Export failed.',true);
    },'image/png');
  },[]);

  const startRec=useCallback(()=>{
    const c=canvasRef.current; if(!c) return;
    const rVid=recVidRef.current;
    if(rVid&&rVid.duration) rVid.currentTime=rVid.duration*trimInRef.current;
    const mime=MediaRecorder.isTypeSupported('video/webm;codecs=vp9')?'video/webm;codecs=vp9':'video/webm';
    const bitrate=quality==='ultra'?60_000_000:30_000_000;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const rec=new MediaRecorder((c as any).captureStream(60),{mimeType:mime,videoBitsPerSecond:bitrate});
    chunksRef.current=[];
    rec.ondataavailable=e=>{if(e.data.size>0)chunksRef.current.push(e.data);};
    rec.onstop=()=>{
      dl(new Blob(chunksRef.current,{type:mime}),'mockup.webm');
      setIsRec(false);setRecTime(0);if(timerRef.current)clearInterval(timerRef.current);
      showToast('Recording saved!');
    };
    recorderRef.current=rec; rec.start(100); setIsRec(true); setRecTime(0); setShowExp(false);
    timerRef.current=setInterval(()=>setRecTime(t=>t+1),1000);
  },[quality]);

  const stopRec=useCallback(()=>{
    recorderRef.current?.stop(); if(timerRef.current)clearInterval(timerRef.current);
  },[]);

  function showToast(msg:string,err=false){setToast({msg,err});setTimeout(()=>setToast(null),3000);}
  const fmt=(s:number)=>`${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;
  const trimInPct=trimIn*100,trimOutPct=trimOut*100;

  return(
    <>
      <style>{CSS}</style>
      <video ref={mockupVidRef} style={{display:'none'}}/>
      <video ref={recVidRef}    style={{display:'none'}}/>

      <div className="app">
        <header className="hdr">
          <div className="logo"><div className="logo-gem"/>Mockup Studio</div>
          <div className="hdr-r">
            {isRec&&<div className="rec-badge"><span className="rec-dot"/>REC {fmt(recTime)}</div>}
            {isRec
              ?<button className="btn btn-danger" onClick={stopRec}>■ Stop &amp; Save</button>
              :<button className="btn btn-export" onClick={()=>setShowExp(true)} disabled={!mockupSrc}>↑ Export</button>}
          </div>
        </header>

        <div className="body">
          <aside className="sidebar">
            <div className="top-tabs">
              <button className={`top-tab${tab==='edit'?' active':''}`}    onClick={()=>setTab('edit')}>✦ Edit</button>
              <button className={`top-tab${tab==='library'?' active':''}`} onClick={()=>setTab('library')}>⊞ Library</button>
            </div>

            {tab==='library'&&<Library onSelectPhoto={loadMockupFromUrl}/>}

            {tab==='edit'&&<>
              <div className="mode-row">
                <button className={`mode-btn${mode==='manual'?' active':''}`} onClick={()=>setMode('manual')}>✦ Manual</button>
                <button className={`mode-btn${mode==='auto'?' active':''}`}   onClick={()=>setMode('auto')}>⚡ Auto</button>
              </div>

              <div className="sec">
                <div className="sec-title">{mode==='auto'?'Mockup (solid-colour screen)':'Mockup / Background'}</div>
                <DropZone label="Upload mockup" hint="PNG · JPG · MP4 · WebM"
                  file={mockupFile} onFile={loadMockup} accept="image/*,video/*" icon="🖼️"/>
                {mode==='auto'&&!mockupSrc&&(
                  <div style={{marginTop:8,padding:'8px 10px',background:'rgba(91,156,246,.07)',borderRadius:7,
                    border:'1px solid rgba(91,156,246,.15)',fontSize:10,color:'#5B9CF6',lineHeight:1.8}}>
                    Auto mode keys out the screen colour. Best with mockup templates that have a flat green, white, or any solid-colour screen.
                  </div>
                )}
              </div>

              <div className="sec">
                <div className="sec-title">Screen Recording</div>
                <DropZone label="Upload recording" hint="MP4 · WebM · MOV · PNG"
                  file={recFile} onFile={loadRec} accept="video/*,image/*" icon="🎬"/>
                {recDur>0&&<>
                  <div style={{display:'flex',justifyContent:'space-between',fontSize:10,color:'var(--muted)',marginTop:8,marginBottom:2}}>
                    <span>Trim</span>
                    <span>{fmt(Math.round(trimIn*recDur))} – {fmt(Math.round(trimOut*recDur))}</span>
                  </div>
                  <div className="trim-wrap">
                    <div className="trim-track"/>
                    <div className="trim-fill" style={{left:`${trimInPct}%`,right:`${100-trimOutPct}%`}}/>
                    <input type="range" min={0} max={1} step={0.01} value={trimIn}
                      onChange={e=>setTrimIn(Math.min(Number(e.target.value),trimOut-0.01))}/>
                    <input type="range" min={0} max={1} step={0.01} value={trimOut}
                      onChange={e=>setTrimOut(Math.max(Number(e.target.value),trimIn+0.01))}/>
                  </div>
                </>}
              </div>

              {mode==='manual'&&mockupSrc&&(
                <div className="sec">
                  <div className="sec-title">Corner Pins</div>
                  <p style={{fontSize:10.5,color:'var(--muted)',lineHeight:1.7,marginBottom:9}}>
                    Drag 4 handles onto the device screen. Edges are softly feathered — no hard cut lines.
                  </p>
                  <button className="btn btn-ghost" style={{width:'100%',justifyContent:'center',fontSize:11}}
                    onClick={()=>setPins(defaultCorners(csz.w,csz.h))}>Reset pins</button>
                </div>
              )}

              {mode==='auto'&&(
                <div className="sec">
                  <div className="sec-title">Chroma Key</div>
                  <div style={{display:'flex',alignItems:'center',gap:7,marginBottom:8}}>
                    <input type="color" value={keyColor} onChange={e=>setKeyColor(e.target.value)}/>
                    <button className="btn btn-ghost" style={{flex:1,fontSize:10.5,padding:5}} onClick={()=>{
                      if(!mockupSrc||mockupIsV) return;
                      const img=new Image();img.crossOrigin='anonymous';
                      img.onload=()=>setKeyColor(detectKeyColor(img));img.src=mockupSrc;
                    }}>Auto-detect</button>
                  </div>
                  <Slider label="Threshold" min={0.05} max={0.8} step={0.01} value={keyThresh} onChange={setKeyThresh}/>
                  <Slider label="Softness"  min={0.01} max={0.4} step={0.01} value={keySoft}   onChange={setKeySoft}/>
                </div>
              )}

              <div className="sec">
                <div className="sec-title">Color Grade</div>
                <div className="grade-row">
                  {(['none','natural','cinematic','vivid'] as GradeName[]).map(g=>(
                    <button key={g} className={`grade-btn${grade===g?' active':''}`} onClick={()=>applyGrade(g)}>
                      {g==='none'?'Off':g.charAt(0).toUpperCase()+g.slice(1)}
                    </button>
                  ))}
                </div>
                <div className="expand-row" onClick={()=>setExpandEnh(v=>!v)}>
                  <span className="expand-lbl">Custom controls</span>
                  <span className={`expand-arrow${expandEnh?' open':''}`}>▾</span>
                </div>
                {expandEnh&&<>
                  <Slider label="Sharpen"     min={0}    max={2}   step={0.05} value={enhance.sharp}    onChange={v=>{setGrade('none');setEnhance(p=>({...p,sharp:v}));}}/>
                  <Slider label="Brightness"  min={-.25} max={.25} step={0.01} value={enhance.bright}   onChange={v=>{setGrade('none');setEnhance(p=>({...p,bright:v}));}}/>
                  <Slider label="Contrast"    min={0.5}  max={2}   step={0.01} value={enhance.contrast} onChange={v=>{setGrade('none');setEnhance(p=>({...p,contrast:v}));}}/>
                  <Slider label="Saturation"  min={0}    max={2}   step={0.01} value={enhance.sat}      onChange={v=>{setGrade('none');setEnhance(p=>({...p,sat:v}));}}/>
                  <Slider label="Vignette"    min={0}    max={1}   step={0.01} value={enhance.vignette} onChange={v=>{setGrade('none');setEnhance(p=>({...p,vignette:v}));}}/>
                  <Slider label="Temperature" min={-.3}  max={.3}  step={0.01} value={enhance.temp}     onChange={v=>{setGrade('none');setEnhance(p=>({...p,temp:v}));}}/>
                  <Slider label="Bloom"       min={0}    max={1}   step={0.01} value={enhance.bloom}    onChange={v=>{setGrade('none');setEnhance(p=>({...p,bloom:v}));}}/>
                  <Slider label="Film Grain"  min={0}    max={1}   step={0.01} value={enhance.grain}    onChange={v=>{setGrade('none');setEnhance(p=>({...p,grain:v}));}}/>
                </>}
              </div>

              <div className="sec">
                <div className="expand-row" onClick={()=>setShowPre(v=>!v)}>
                  <span className="expand-lbl">Presets</span>
                  <span className={`expand-arrow${showPre?' open':''}`}>▾</span>
                </div>
                {showPre&&<>
                  {presets.length>0&&(
                    <div className="preset-list">
                      {presets.map((p,i)=>(
                        <div key={i} className="preset-item" onClick={()=>loadPreset(p)}>
                          <span>{p.name}</span>
                          <button onClick={e=>{e.stopPropagation();deletePreset(i);}}>✕</button>
                        </div>
                      ))}
                    </div>
                  )}
                  <div style={{display:'flex',gap:5}}>
                    <input className="preset-input" placeholder="Preset name…"
                      value={presetName} onChange={e=>setPresetName(e.target.value)}
                      onKeyDown={e=>{if(e.key==='Enter') savePreset();}}/>
                    <button className="btn btn-ghost" style={{padding:'5px 10px',fontSize:11}} onClick={savePreset}>Save</button>
                  </div>
                </>}
              </div>

              <div className="sec">
                <div className="sec-title">Quick guide</div>
                {mode==='manual'
                  ?<div style={{fontSize:10.5,color:'var(--muted)',lineHeight:2.1}}>
                    <div>① Library → pick a device photo</div>
                    <div>② Upload your screen recording</div>
                    <div>③ Drag 4 pins onto the screen</div>
                    <div>④ Choose Natural · Cinematic · Vivid</div>
                    <div>⑤ Export PNG or record WebM</div>
                  </div>
                  :<div style={{fontSize:10.5,color:'var(--muted)',lineHeight:2.1}}>
                    <div>① Upload mockup with solid-colour screen</div>
                    <div>② Upload screen recording</div>
                    <div>③ Hit Auto-detect key colour</div>
                    <div>④ Tune Threshold to clean edges</div>
                    <div>⑤ Export</div>
                  </div>}
              </div>
            </>}
          </aside>

          <main className="canvas-area">
            <div className="canvas-wrap" style={{width:csz.w,height:csz.h}}
              onPointerMove={onMove} onPointerUp={onUp} onPointerLeave={onUp}>
              <canvas ref={canvasRef} width={native.w} height={native.h}
                style={{display:'block',width:csz.w,height:csz.h}}/>
              {mode==='manual'&&mockupSrc&&(
                <svg style={{position:'absolute',inset:0,pointerEvents:'none'}} width={csz.w} height={csz.h}>
                  <polyline points={[...pins,pins[0]].map(p=>`${p.x},${p.y}`).join(' ')}
                    fill="none" stroke="rgba(124,106,247,.2)" strokeWidth="1" strokeDasharray="5 4"/>
                </svg>
              )}
              {mode==='manual'&&mockupSrc&&pins.map((p,i)=>(
                <div key={i} className={`pin-handle${activePin===i?' active':''}`}
                  style={{left:p.x,top:p.y}} onPointerDown={onPinDown(i)}>
                  <div className="pin-lbl">{['TL','TR','BR','BL'][i]}</div>
                </div>
              ))}
              {!mockupSrc&&(
                <div className="empty" style={{width:csz.w,height:csz.h}}>
                  <div className="empty-ico">🖼️</div>
                  <h3>Upload a mockup or browse Library</h3>
                  <p>Grab a real device photo from the Library tab, or drag your own mockup file here.</p>
                </div>
              )}
            </div>
          </main>
        </div>
      </div>

      {showExp&&(
        <div className="modal-ov" onClick={()=>setShowExp(false)}>
          <div className="modal" onClick={e=>e.stopPropagation()}>
            <div className="modal-hdr">
              <h2>Export</h2>
              <button className="modal-x" onClick={()=>setShowExp(false)}>✕</button>
            </div>
            <div className="modal-body">
              <div className="m-stat"><span>Resolution</span><strong>{native.w} × {native.h}</strong></div>
              <div className="m-stat"><span>Grade</span><strong>{grade==='none'?'None':grade.charAt(0).toUpperCase()+grade.slice(1)}</strong></div>
              {recDur>0&&<div className="m-stat"><span>Clip</span><strong>{fmt(Math.round(trimIn*recDur))} – {fmt(Math.round(trimOut*recDur))}</strong></div>}
              <div className="m-lbl">Format</div>
              <div className="fmt-tabs">
                <div className={`fmt-tab${expFmt==='png'?' active':''}`} onClick={()=>setExpFmt('png')}>📷 PNG Still</div>
                <div className={`fmt-tab${expFmt==='webm'?' active':''}`} onClick={()=>setExpFmt('webm')}>🎬 WebM Video</div>
              </div>
              {expFmt==='webm'&&<>
                <div className="m-lbl">Quality</div>
                <div className="q-row">
                  <div className={`q-btn${quality==='high'?' active':''}`} onClick={()=>setQuality('high')}>
                    High<br/><span style={{fontSize:9,opacity:.6}}>30 Mbps VP9</span>
                  </div>
                  <div className={`q-btn${quality==='ultra'?' active':''}`} onClick={()=>setQuality('ultra')}>
                    Ultra<br/><span style={{fontSize:9,opacity:.6}}>60 Mbps VP9</span>
                  </div>
                </div>
                <div className="q-note">60 fps · native res · grade baked in · starts from trim point</div>
              </>}
              {expFmt==='png'
                ?<button className="btn btn-export" style={{width:'100%',justifyContent:'center',padding:10}} onClick={doExportPNG}>Export PNG</button>
                :<button className="btn btn-export" style={{width:'100%',justifyContent:'center',padding:10}} onClick={startRec}>● Start Recording</button>}
            </div>
          </div>
        </div>
      )}

      {toast&&<div className={`toast${toast.err?' err':''}`}>{toast.msg}</div>}
    </>
  );
}
