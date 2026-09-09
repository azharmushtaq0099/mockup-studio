import React, { useState, useRef, useEffect, useCallback } from 'react';
import { computeHomography, invertHomography, defaultCorners, type Quad } from './homography';

// ─── Shaders ──────────────────────────────────────────────────────────────────

const VERT = `
attribute vec2 aPos;
attribute vec3 aUVW;
varying vec3 vUVW;
void main() { vUVW = aUVW; gl_Position = vec4(aPos, 0.0, 1.0); }
`;

// uEdge=1 → soft feather at UV edges; uOpacity < 1 for ghost/layer transparency
const FRAG_PLAIN = `
precision mediump float;
uniform sampler2D uTex;
uniform float uEdge;
uniform float uOpacity;
varying vec3 vUVW;
void main() {
  vec2 uv = vUVW.xy / vUVW.z;
  vec4 c = texture2D(uTex, uv);
  if (uEdge > 0.5) {
    float f = 0.05;
    float a = smoothstep(0.0, f, uv.x)   * smoothstep(0.0, f, 1.0-uv.x) *
              smoothstep(0.0, f, uv.y)   * smoothstep(0.0, f, 1.0-uv.y);
    c.a *= a;
  }
  float op = uOpacity > 0.001 ? uOpacity : 1.0;
  c.a *= op;
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

// Single-pass two-texture blend — always fully opaque output, zero bleed possible.
// Outside pin quad → mockup. Inside → mix(mockup, recording, 1-luma): dark screen shows
// recording, bright pixels (hand/bezel) stay as mockup.
// uRec bound to TEXTURE1. uRecCrop = vec4(u0,u1,v0,v1) for cover-scale recording UV.
const FRAG_BLEND = `
precision mediump float;
uniform sampler2D uTex;
uniform sampler2D uRec;
uniform vec2 uQ0,uQ1,uQ2,uQ3;
uniform float uLuma;
uniform float uLumaSoft;
uniform float uChroma;
uniform vec4 uRecCrop;
varying vec3 vUVW;
float cx(vec2 a,vec2 b){return a.x*b.y-a.y*b.x;}
void main(){
  vec2 uv=vUVW.xy/vUVW.z;
  vec4 mock=texture2D(uTex,uv);
  float d0=cx(uQ1-uQ0,uv-uQ0),d1=cx(uQ2-uQ1,uv-uQ1);
  float d2=cx(uQ3-uQ2,uv-uQ2),d3=cx(uQ0-uQ3,uv-uQ3);
  bool inside=(d0>=0.0&&d1>=0.0&&d2>=0.0&&d3>=0.0)||(d0<=0.0&&d1<=0.0&&d2<=0.0&&d3<=0.0);
  if(!inside){gl_FragColor=vec4(mock.rgb,1.0);return;}
  float t;
  if(uChroma>0.001){
    float ge=mock.g-max(mock.r,mock.b);
    t=smoothstep(max(0.0,uChroma-uLumaSoft),uChroma+uLumaSoft,ge);
  } else {
    float lum=dot(mock.rgb,vec3(0.2126,0.7152,0.0722));
    t=1.0-smoothstep(max(0.0,uLuma-uLumaSoft),uLuma+uLumaSoft,lum);
  }
  vec2 ruv=vec2(uRecCrop.x+uv.x*(uRecCrop.y-uRecCrop.x),uRecCrop.z+uv.y*(uRecCrop.w-uRecCrop.z));
  vec4 rec=texture2D(uRec,ruv);
  gl_FragColor=vec4(mix(mock.rgb,rec.rgb,t),1.0);
}
`;

// Mockup with screen area punched out — lets recording show through, keeps foreground (hand) on top
const FRAG_CUTOUT = `
precision mediump float;
uniform sampler2D uTex;
uniform vec2 uQ0,uQ1,uQ2,uQ3;
uniform float uOpacity;
varying vec3 vUVW;
float cx(vec2 a,vec2 b){return a.x*b.y-a.y*b.x;}
void main(){
  vec2 uv=vUVW.xy/vUVW.z;
  float d0=cx(uQ1-uQ0,uv-uQ0),d1=cx(uQ2-uQ1,uv-uQ1);
  float d2=cx(uQ3-uQ2,uv-uQ2),d3=cx(uQ0-uQ3,uv-uQ3);
  if((d0>=0.0&&d1>=0.0&&d2>=0.0&&d3>=0.0)||(d0<=0.0&&d1<=0.0&&d2<=0.0&&d3<=0.0)) discard;
  vec4 col=texture2D(uTex,uv);
  float op = uOpacity > 0.001 ? uOpacity : 1.0;
  col.a *= op;
  gl_FragColor=col;
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
  gl.linkProgram(p);
  if(!gl.getProgramParameter(p,gl.LINK_STATUS)) console.error('GL link:',gl.getProgramInfoLog(p));
  return p;
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
  tex2?: WebGLTexture | null,
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
  if(tex2){
    gl.uniform1i(gl.getUniformLocation(prog,'uRec'),1);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D,tex2);
    gl.activeTexture(gl.TEXTURE0);
  }
  if (uniforms) for (const [k,v] of Object.entries(uniforms)) {
    const loc = gl.getUniformLocation(prog,k);
    if (typeof v === 'number') gl.uniform1f(loc,v);
    else if (v.length===4) gl.uniform4fv(loc,v);
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
function coverUVBounds(sw:number,sh:number,dw:number,dh:number):[number,number,number,number]{
  const sa=sw/sh,da=dw/dh; let u0=0,u1=1,v0=0,v1=1;
  if(sa>da){const m=(1-da/sa)/2;u0=m;u1=1-m;}
  else if(sa<da){const m=(1-sa/da)/2;v0=m;v1=1-m;}
  return [u0,u1,v0,v1];
}

// 32×32 subdivided mesh — ultra-smooth perspective warp, zero corner artefacts
function pinVerts(pins: Quad, W: number, H: number, N=32): Float32Array | null {
  try {
    // V-flipped: UNPACK_FLIP_Y_WEBGL makes V=0 → image bottom, so map top corners to V=1
    const src: Quad = [{x:0,y:1},{x:1,y:1},{x:1,y:0},{x:0,y:0}];
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
type TextItem={id:string;text:string;x:number;y:number;size:number;color:string;family:string;bold:boolean;italic:boolean;};
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
type ExportRatio='16:9'|'9:16'|'1:1';

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
  position:relative;background:radial-gradient(ellipse at center, #0E0E1C 0%, #07070F 100%)}
.zoom-pill{position:absolute;bottom:14px;left:50%;transform:translateX(-50%);
  display:flex;align-items:center;gap:2px;
  background:rgba(10,10,20,0.92);border:1px solid var(--border);
  border-radius:20px;padding:3px 8px;box-shadow:0 4px 20px rgba(0,0,0,.6);z-index:50;user-select:none}
.zoom-btn{background:none;border:none;color:var(--muted);font-size:14px;cursor:pointer;
  padding:2px 7px;border-radius:12px;line-height:1;transition:all .1s}
.zoom-btn:hover{background:var(--surface);color:var(--text)}
.zoom-pct{font-size:11px;color:var(--text);cursor:pointer;min-width:38px;text-align:center;padding:0 4px}
.canvas-wrap{position:relative;
  box-shadow:0 32px 100px rgba(0,0,0,.9),0 12px 36px rgba(0,0,0,.7),0 0 0 1px rgba(255,255,255,.05)}
/* L-bracket corner pins — sit outside the corner, never obscure the screen edge */
.pin-corner{position:absolute;width:20px;height:20px;cursor:crosshair;z-index:10;
  user-select:none;touch-action:none;box-sizing:border-box;}
.pin-corner .arm-h,.pin-corner .arm-v{position:absolute;background:rgba(255,255,255,0.95);
  transition:background .1s;}
.pin-corner .arm-h{height:2px;width:12px;}
.pin-corner .arm-v{width:2px;height:12px;}
.pin-corner:hover .arm-h,.pin-corner:hover .arm-v{background:#5B9CF6;}
.pin-corner.active .arm-h,.pin-corner.active .arm-v{background:#7C6AF7;}
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
  const [exportRatio,setExportRatio]=useState<ExportRatio>('16:9');
  const [isRec,     setIsRec]    = useState(false);
  const [recTime,   setRecTime]  = useState(0);
  const [toast,     setToast]    = useState<{msg:string;err?:boolean}|null>(null);
  const [zoom,      setZoom]     = useState(1.0);
  const [pan,       setPan]      = useState({x:0,y:0});
  const [edgeBlend,  setEdgeBlend]  = useState(true);
  const [mockupOp,   setMockupOp]   = useState(1.0);
  const [lumaKey,    setLumaKey]    = useState(0.0);
  const [lumaSoft,   setLumaSoft]   = useState(0.08);
  const [chromaKey,  setChromaKey]  = useState(0.0);
  const [textItems,  setTextItems]  = useState<TextItem[]>([]);
  const [selTextId,  setSelTextId]  = useState<string|null>(null);
  const [audioSrc,   setAudioSrc]   = useState<string|null>(null);
  const [audioName,  setAudioName]  = useState('');
  const [audioVol,   setAudioVol]   = useState(0.8);

  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const reflRef     = useRef<HTMLCanvasElement>(null);
  const mouseXRef   = useRef(0.5);
  const zoomRef     = useRef(1.0);
  const panRef      = useRef({x:0,y:0});
  const panStartRef = useRef<{mx:number;my:number;px:number;py:number}|null>(null);
  const glRef       = useRef<WebGLRenderingContext|null>(null);
  const plainRef    = useRef<WebGLProgram|null>(null);
  const chromaRef   = useRef<WebGLProgram|null>(null);
  const postRef     = useRef<WebGLProgram|null>(null);
  const cutoutRef   = useRef<WebGLProgram|null>(null);
  const blendRef    = useRef<WebGLProgram|null>(null);
  const fboRef      = useRef<FBO|null>(null);
  const mTexRef     = useRef<WebGLTexture|null>(null);
  const rTexRef     = useRef<WebGLTexture|null>(null);
  const mockupVidRef= useRef<HTMLVideoElement>(null);
  const recVidRef   = useRef<HTMLVideoElement>(null);

  const mReadyRef   = useRef(false);
  const rReadyRef   = useRef(false);
  const mIsVRef     = useRef(false);
  const rStaticRef  = useRef(false);
  const pinsRef        = useRef<Quad>(pins);
  const cszRef         = useRef(csz);
  const activePinRef   = useRef<number|null>(null);
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
  const edgeBlendRef = useRef(true);
  const mockupOpRef  = useRef(1.0);
  const lumaKeyRef   = useRef(0.0);
  const lumaSoftRef  = useRef(0.08);
  const chromaKeyRef  = useRef(0.0);
  const textItemsRef  = useRef<TextItem[]>([]);
  const textDragRef   = useRef<{id:string;sx:number;sy:number;ox:number;oy:number}|null>(null);
  const textCanvasRef = useRef<HTMLCanvasElement|null>(null);
  const textTexRef    = useRef<WebGLTexture|null>(null);
  const audioElRef    = useRef<HTMLAudioElement|null>(null);
  const audioCtxRef   = useRef<AudioContext|null>(null);
  const audioVolRef   = useRef(0.8);
  const videoTrackRef  = useRef<{requestFrame():void}|null>(null);
  const outCanvasRef   = useRef<HTMLCanvasElement|null>(null);
  const exportRatioRef = useRef<ExportRatio>('16:9');

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
  useEffect(()=>{zoomRef.current=zoom},[zoom]);
  useEffect(()=>{panRef.current=pan},[pan]);
  useEffect(()=>{edgeBlendRef.current=edgeBlend},[edgeBlend]);
  useEffect(()=>{mockupOpRef.current=mockupOp},[mockupOp]);
  useEffect(()=>{lumaKeyRef.current=lumaKey},[lumaKey]);
  useEffect(()=>{lumaSoftRef.current=lumaSoft},[lumaSoft]);
  useEffect(()=>{chromaKeyRef.current=chromaKey},[chromaKey]);
  useEffect(()=>{textItemsRef.current=textItems},[textItems]);
  useEffect(()=>{audioVolRef.current=audioVol; if(audioElRef.current) audioElRef.current.volume=audioVol;},[audioVol]);
  useEffect(()=>{ exportRatioRef.current=exportRatio; },[exportRatio]);

  // ── Init WebGL ──────────────────────────────────────────────────────────────
  useEffect(()=>{
    const canvas=canvasRef.current!;
    // antialias:true — hardware MSAA eliminates mesh boundary jagging
    const gl=canvas.getContext('webgl',{preserveDrawingBuffer:true,alpha:false,antialias:true});
    if(!gl) return;
    glRef.current=gl;
    plainRef.current  =mkProgram(gl,FRAG_PLAIN);
    chromaRef.current =mkProgram(gl,FRAG_CHROMA);
    postRef.current   =mkProgram(gl,FRAG_POST);
    cutoutRef.current =mkProgram(gl,FRAG_CUTOUT);
    blendRef.current  =mkProgram(gl,FRAG_BLEND);
    mTexRef.current  =mkTex(gl); rTexRef.current=mkTex(gl); textTexRef.current=mkTex(gl);
    fboRef.current   =createFBO(gl,canvas.width,canvas.height);
    const tc=document.createElement('canvas'); tc.width=canvas.width; tc.height=canvas.height;
    textCanvasRef.current=tc;
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);

    let frameCount=0;
    function frame(){
      frameCount++;
      const gl=glRef.current!,plain=plainRef.current!,chroma=chromaRef.current!,
            post=postRef.current!,fbo=fboRef.current!,mt=mTexRef.current!,rt=rTexRef.current!;
      const mVid=mockupVidRef.current,rVid=recVidRef.current;
      const W=canvas.width,H=canvas.height;
      const {w:dW,h:dH}=cszRef.current,{w:rW,h:rH}=recNatRef.current;
      const e=enhRef.current,t=performance.now()*0.001;

      gl.bindFramebuffer(gl.FRAMEBUFFER,fbo.fb);
      gl.viewport(0,0,W,H); gl.clearColor(0.02,0.02,0.055,1); gl.clear(gl.COLOR_BUFFER_BIT);

      if(modeRef.current==='manual'){
        if(mReadyRef.current&&mIsVRef.current&&mVid&&mVid.readyState>=2) uploadTex(gl,mt,mVid);
        if(rReadyRef.current&&!rStaticRef.current&&rVid&&rVid.readyState>=2) uploadTex(gl,rt,rVid);
        const mop=mockupOpRef.current, ue=edgeBlendRef.current?1:0;
        const lk=lumaKeyRef.current, ls=lumaSoftRef.current, ck=chromaKeyRef.current;

        if(rReadyRef.current){
          const sx=dW>0?W/dW:1,sy=dH>0?H/dH:1;
          const np=pinsRef.current.map(p=>({x:p.x*sx,y:p.y*sy})) as Quad;
          const pinUV={
            uQ0:[np[0].x/W,1-np[0].y/H] as [number,number],
            uQ1:[np[1].x/W,1-np[1].y/H] as [number,number],
            uQ2:[np[2].x/W,1-np[2].y/H] as [number,number],
            uQ3:[np[3].x/W,1-np[3].y/H] as [number,number],
          };
          if((lk>0.001||ck>0.001) && mReadyRef.current && blendRef.current){
            // Single-pass blend: always opaque — no bleed possible
            const crop=coverUVBounds(rW,rH,W,H);
            drawQuad(gl,blendRef.current,mt,bgVerts(),
              {...pinUV,uLuma:lk,uLumaSoft:ls,uChroma:ck,uRecCrop:crop},rt);
          } else {
            const vt=pinVerts(np,W,H);
            if(vt){
              // Mockup first (full), then recording on top with soft feather — seamless embed
              if(mReadyRef.current){
                drawQuad(gl,plain,mt,bgVerts(),{uEdge:0,uOpacity:mop});
              }
              drawQuad(gl,plain,rt,vt,{uEdge:ue});
            } else if(mReadyRef.current){
              drawQuad(gl,plain,mt,bgVerts(),{uEdge:0,uOpacity:mop});
            }
          }
        } else if(mReadyRef.current){
          drawQuad(gl,plain,mt,bgVerts(),{uEdge:0,uOpacity:mop});
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

      // Text overlay — composite on top of post-processed canvas
      const tc=textCanvasRef.current, ttex=textTexRef.current;
      if(tc && ttex && textItemsRef.current.length>0){
        if(tc.width!==W||tc.height!==H){tc.width=W;tc.height=H;}
        const ctx2d=tc.getContext('2d');
        if(ctx2d){
          ctx2d.clearRect(0,0,W,H);
          const dw=cszRef.current.w||1;
          const scl=W/dw;
          ctx2d.textBaseline='top';
          for(const item of textItemsRef.current){
            ctx2d.save();
            const sz=Math.round(item.size*scl);
            const base=item.family.replace(/,?\s*(sans-serif|serif|cursive|monospace)\s*$/,'');
            const mainFont=`${item.italic?'italic ':''}${item.bold?'bold ':''}${sz}px ${base},'Segoe UI Emoji','Apple Color Emoji','Noto Color Emoji',sans-serif`;
            const emojiFont=`${sz}px 'Segoe UI Emoji','Apple Color Emoji','Noto Color Emoji',sans-serif`;
            ctx2d.fillStyle=item.color;
            ctx2d.shadowColor='rgba(0,0,0,0.6)'; ctx2d.shadowBlur=Math.round(6*scl);
            const lines=item.text.split('\n');
            const lineH=sz*1.25;
            lines.forEach((txt,li)=>{
              const emojiRe=/\p{Extended_Pictographic}/gu;
              let last=0, cx=item.x*W;
              const cy=item.y*H+li*lineH;
              for(const m of txt.matchAll(emojiRe)){
                if(m.index!>last){
                  ctx2d.font=mainFont;
                  const seg=txt.slice(last,m.index);
                  ctx2d.fillText(seg,cx,cy);
                  cx+=ctx2d.measureText(seg).width;
                }
                ctx2d.font=emojiFont;
                ctx2d.fillText(m[0],cx,cy);
                cx+=ctx2d.measureText(m[0]).width;
                last=m.index!+m[0].length;
              }
              if(last<txt.length){ctx2d.font=mainFont;ctx2d.fillText(txt.slice(last),cx,cy);}
            });
            ctx2d.restore();
          }
          gl.bindTexture(gl.TEXTURE_2D,ttex);
          gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true);
          gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,tc);
          gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);
          gl.bindFramebuffer(gl.FRAMEBUFFER,null);
          drawQuad(gl,plainRef.current!,ttex,bgVerts(),{uEdge:0,uOpacity:1});
        }
      }

      // Reflection — throttled to every 4 frames, skipped during recording
      const recording=recorderRef.current?.state==='recording';
      const refl = reflRef.current;
      if (refl && mReadyRef.current && !recording && frameCount%4===0) {
        gl.finish(); // block GPU only when needed for pixel readback
        const rfH = refl.height, rfW = Math.min(refl.width, W);
        const ctx = refl.getContext('2d');
        if (ctx) {
          // Read bottom rfH rows of the WebGL canvas (Y=0 is bottom in WebGL)
          const pixels = new Uint8Array(rfW * rfH * 4);
          gl.readPixels(0, 0, rfW, rfH, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
          const imgData = ctx.createImageData(rfW, rfH);
          for (let row = 0; row < rfH; row++) {
            // Flip: row 0 of imgData ← row rfH-1 of pixels (bottom→top)
            const srcRow = rfH - 1 - row;
            const s = srcRow * rfW * 4, d = row * rfW * 4;
            imgData.data.set(pixels.subarray(s, s + rfW * 4), d);
            // Gradient alpha: strong at top (row 0), fades to transparent
            const fade = 1.0 - row / rfH;
            const alpha = Math.round(fade * fade * 230);
            for (let c = 0; c < rfW; c++) {
              const bi = d + c * 4;
              // 2× brightness boost + blue glass tint so dark pixels become visible
              imgData.data[bi]   = Math.min(255, imgData.data[bi]   * 2);
              imgData.data[bi+1] = Math.min(255, imgData.data[bi+1] * 2);
              imgData.data[bi+2] = Math.min(255, imgData.data[bi+2] * 2 + 35);
              imgData.data[bi+3] = alpha;
            }
          }
          ctx.clearRect(0, 0, rfW, rfH);
          // Subtle parallax: shift ImageData by mouse position
          const shift = Math.round((mouseXRef.current - 0.5) * rfW * 0.025);
          ctx.putImageData(imgData, shift, 0);
        }
      }

      // Composite to output canvas (ratio conversion for Reels / Square)
      const ratio=exportRatioRef.current;
      if(ratio!=='16:9'){
        if(!outCanvasRef.current) outCanvasRef.current=document.createElement('canvas');
        const oc=outCanvasRef.current;
        if(ratio==='9:16'&&(oc.width!==1080||oc.height!==1920)){oc.width=1080;oc.height=1920;}
        if(ratio==='1:1'&&(oc.width!==1080||oc.height!==1080)){oc.width=1080;oc.height=1080;}
        const octx=oc.getContext('2d');
        const glc=canvasRef.current;
        if(octx&&glc){
          const dw=oc.width,dh=oc.height,sa=W/H,da=dw/dh;
          // Blurred background — slightly oversized to hide blur-edge artifacts
          octx.save();
          octx.filter='blur(28px) brightness(0.28) saturate(1.6)';
          octx.drawImage(glc,-60,-60,dw+120,dh+120);
          octx.restore();
          // Main content centered, aspect-correct
          let mw:number,mh:number;
          if(sa>da){mw=dw;mh=Math.round(dw/sa);}else{mh=dh;mw=Math.round(dh*sa);}
          octx.drawImage(glc,Math.round((dw-mw)/2),Math.round((dh-mh)/2),mw,mh);
        }
      }

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
        // Guard: applyMockupSize resets pins — only call once on initial load
        if(!mReadyRef.current) applyMockupSize(vid.videoWidth||1920,vid.videoHeight||1080);
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
    // Divide avH by 1.28 so canvas+reflection (22% extra) fits without overflow:hidden clipping
    const scale=Math.min(avW/nW,(avH/1.28)/nH,1);
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

  const addText=useCallback(()=>{
    const id=Date.now().toString();
    setTextItems(p=>[...p,{id,text:'Your text',x:0.5,y:0.15,size:48,color:'#ffffff',family:"'Bebas Neue', sans-serif",bold:false,italic:false}]);
    setSelTextId(id);
  },[]);
  const updateText=useCallback((id:string,key:keyof TextItem,val:unknown)=>{
    setTextItems(p=>p.map(it=>it.id===id?{...it,[key]:val}:it));
  },[]);
  const removeText=useCallback((id:string)=>{
    setTextItems(p=>p.filter(it=>it.id!==id));
    setSelTextId(null);
  },[]);

  const onPinDown=useCallback((i:number)=>(e:React.PointerEvent)=>{
    e.preventDefault();e.stopPropagation(); // stop bubbling so canvas-wrap doesn't start pan
    activePinRef.current=i; setActivePin(i);
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  },[]);
  const onCanvasDown=useCallback((e:React.PointerEvent)=>{
    // No pin active → start pan drag
    panStartRef.current={mx:e.clientX,my:e.clientY,px:panRef.current.x,py:panRef.current.y};
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
  },[]);
  const onMove=useCallback((e:React.PointerEvent)=>{
    const ap=activePinRef.current;
    const td=textDragRef.current;
    if(ap!==null){
      const rect=canvasRef.current!.getBoundingClientRect();
      const z=zoomRef.current,{w,h}=cszRef.current;
      setPins(prev=>{const n=[...prev] as Quad;n[ap]={x:Math.max(0,Math.min(w,(e.clientX-rect.left)/z)),y:Math.max(0,Math.min(h,(e.clientY-rect.top)/z))};return n;});
    } else if(td){
      const rect=canvasRef.current!.getBoundingClientRect();
      const z=zoomRef.current,{w,h}=cszRef.current;
      const cx=(e.clientX-rect.left)/z, cy=(e.clientY-rect.top)/z;
      setTextItems(prev=>prev.map(it=>it.id===td.id
        ?{...it,x:Math.max(0,Math.min(1,td.ox+(cx-td.sx)/w)),y:Math.max(0,Math.min(1,td.oy+(cy-td.sy)/h))}:it));
    } else if(panStartRef.current){
      const {mx,my,px,py}=panStartRef.current;
      setPan({x:px+e.clientX-mx,y:py+e.clientY-my});
    }
  },[]);
  const onUp=useCallback(()=>{activePinRef.current=null;setActivePin(null);panStartRef.current=null;textDragRef.current=null;},[]);
  const onAreaWheel=useCallback((e:React.WheelEvent)=>{
    e.preventDefault();
    const factor=e.deltaY>0?0.88:1.14;
    setZoom(z=>Math.max(0.2,Math.min(4,z*factor)));
  },[]);
  const fitZoom=useCallback(()=>{setZoom(1);setPan({x:0,y:0});},[]);

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
    const mime=MediaRecorder.isTypeSupported('video/mp4;codecs=avc1')?'video/mp4;codecs=avc1'
             :MediaRecorder.isTypeSupported('video/webm;codecs=vp9')?'video/webm;codecs=vp9':'video/webm';
    const ext=mime.startsWith('video/mp4')?'mp4':'webm';
    const bitrate=quality==='ultra'?80_000_000:40_000_000;
    // Use output canvas for ratio conversion (9:16 / 1:1), native canvas for 16:9
    const ratio=exportRatioRef.current;
    let recordCanvas:HTMLCanvasElement=c;
    if(ratio!=='16:9'&&outCanvasRef.current) recordCanvas=outCanvasRef.current;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const canvasStream=(recordCanvas as any).captureStream(60) as MediaStream;
    let recStream=canvasStream;
    const audioEl=audioElRef.current;
    if(audioEl&&audioEl.src){
      try{
        const actx=new AudioContext(); audioCtxRef.current=actx;
        const src=actx.createMediaElementSource(audioEl);
        const dest=actx.createMediaStreamDestination();
        src.connect(dest); src.connect(actx.destination);
        audioEl.volume=audioVolRef.current; audioEl.currentTime=0; audioEl.loop=true; audioEl.play();
        recStream=new MediaStream([...canvasStream.getVideoTracks(),...dest.stream.getAudioTracks()]);
      }catch(e){console.warn('Audio mix failed',e);}
    }
    const rec=new MediaRecorder(recStream,{mimeType:mime,videoBitsPerSecond:bitrate});
    chunksRef.current=[];
    rec.ondataavailable=e=>{if(e.data.size>0)chunksRef.current.push(e.data);};
    rec.onstop=()=>{
      dl(new Blob(chunksRef.current,{type:mime}),`mockup.${ext}`);
      setIsRec(false);setRecTime(0);if(timerRef.current)clearInterval(timerRef.current);
      showToast('Recording saved!');
    };
    recorderRef.current=rec; rec.start(500); setIsRec(true); setRecTime(0); setShowExp(false);
    timerRef.current=setInterval(()=>setRecTime(t=>t+1),1000);
  },[quality]);

  const stopRec=useCallback(()=>{
    recorderRef.current?.stop(); if(timerRef.current)clearInterval(timerRef.current);
    const ae=audioElRef.current; if(ae){ae.pause();ae.currentTime=0;}
    audioCtxRef.current?.close(); audioCtxRef.current=null;
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
                  <p style={{fontSize:10.5,color:'var(--muted)',lineHeight:1.6,marginBottom:8}}>
                    Drag 4 handles onto the device screen corners.
                  </p>
                  <button className="btn btn-ghost" style={{width:'100%',justifyContent:'center',fontSize:11,marginBottom:12}}
                    onClick={()=>setPins(defaultCorners(csz.w,csz.h))}>Reset pins</button>

                  <div className="sec-title" style={{marginBottom:5}}>Edge Style</div>
                  <div className="grade-row" style={{marginBottom:12}}>
                    <button className={`grade-btn${!edgeBlend?' active':''}`} onClick={()=>setEdgeBlend(false)}>Sharp</button>
                    <button className={`grade-btn${edgeBlend?' active':''}`}  onClick={()=>setEdgeBlend(true)}>Soft Blend</button>
                  </div>

                  <Slider label="Mockup Opacity" min={0.1} max={1} step={0.02} value={mockupOp} onChange={setMockupOp}/>
                  {mockupOp<0.98&&(
                    <button className="btn btn-ghost" style={{width:'100%',justifyContent:'center',fontSize:10.5,marginTop:5}}
                      onClick={()=>setMockupOp(1.0)}>Restore full opacity</button>
                  )}

                  <div className="sec-title" style={{marginTop:12,marginBottom:5}}>Hand in Front</div>
                  <p style={{fontSize:10.5,color:'var(--muted)',lineHeight:1.6,marginBottom:8}}>
                    Hand overlapping the screen? Pick your screen type, adjust pins to the screen corners, then raise the key value.
                  </p>

                  <div className="sec-title" style={{marginBottom:4,fontSize:9.5}}>SCREEN TYPE</div>
                  <div className="grade-row" style={{marginBottom:10}}>
                    <button className={`grade-btn${chromaKey<=0&&lumaKey<=0?' active':''}`}
                      onClick={()=>{setLumaKey(0);setChromaKey(0);}}>Off</button>
                    <button className={`grade-btn${lumaKey>0&&chromaKey<=0?' active':''}`}
                      onClick={()=>{setLumaKey(0.08);setChromaKey(0);}}>Dark Screen</button>
                    <button className={`grade-btn${chromaKey>0?' active':''}`}
                      onClick={()=>{setChromaKey(0.12);setLumaKey(0);}}>Green Screen</button>
                  </div>

                  {lumaKey>0&&chromaKey<=0&&(
                    <>
                      <Slider label="Luma Threshold" min={0.01} max={0.5} step={0.01} value={lumaKey} onChange={setLumaKey}/>
                      <Slider label="Softness" min={0.01} max={0.2} step={0.01} value={lumaSoft} onChange={setLumaSoft}/>
                      <p style={{fontSize:10,color:'var(--muted)',lineHeight:1.5,marginTop:5}}>
                        Start around 0.08. Raise until the screen area clears. Keep pins at the actual screen glass corners.
                      </p>
                    </>
                  )}
                  {chromaKey>0&&(
                    <>
                      <Slider label="Green Sensitivity" min={0.02} max={0.4} step={0.01} value={chromaKey} onChange={setChromaKey}/>
                      <Slider label="Softness" min={0.01} max={0.2} step={0.01} value={lumaSoft} onChange={setLumaSoft}/>
                      <p style={{fontSize:10,color:'var(--muted)',lineHeight:1.5,marginTop:5}}>
                        Use a solid green on the laptop screen. Raise until the green area disappears cleanly.
                      </p>
                    </>
                  )}

                  {/* ── Audio Track ── */}
                  <div className="sec-title" style={{marginTop:16,marginBottom:6}}>Audio Track</div>
                  <label style={{display:'block',cursor:'pointer',marginBottom:6}}>
                    <div className="btn btn-ghost" style={{width:'100%',justifyContent:'center',fontSize:11}}
                      onClick={()=>document.getElementById('audio-pick')?.click()}>
                      {audioName ? `🎵 ${audioName.slice(0,22)}${audioName.length>22?'…':''}` : '+ Add Music / Audio'}
                    </div>
                    <input id="audio-pick" type="file" accept="audio/*" style={{display:'none'}}
                      onChange={e=>{
                        const f=e.target.files?.[0]; if(!f) return;
                        const url=URL.createObjectURL(f);
                        if(!audioElRef.current){audioElRef.current=new Audio();}
                        audioElRef.current.src=url; audioElRef.current.volume=audioVolRef.current;
                        setAudioSrc(url); setAudioName(f.name);
                      }}/>
                  </label>
                  {audioSrc&&(
                    <div style={{marginBottom:8}}>
                      <div className="sl-lbl" style={{marginBottom:3}}>
                        <span>Volume</span><span>{Math.round(audioVol*100)}%</span>
                      </div>
                      <input type="range" min={0} max={1} step={0.01} value={audioVol}
                        onChange={e=>setAudioVol(+e.target.value)} style={{width:'100%'}}/>
                      <div style={{display:'flex',gap:5,marginTop:5}}>
                        <button className="btn btn-ghost" style={{flex:1,fontSize:10,justifyContent:'center'}}
                          onClick={()=>{const ae=audioElRef.current;if(ae){ae.currentTime=0;ae.play();}}}>▶ Preview</button>
                        <button className="btn btn-ghost" style={{flex:1,fontSize:10,justifyContent:'center'}}
                          onClick={()=>{audioElRef.current?.pause();}}>⏹ Stop</button>
                        <button className="btn btn-ghost" style={{flex:1,fontSize:10,justifyContent:'center',color:'#e05050'}}
                          onClick={()=>{
                            audioElRef.current?.pause();
                            if(audioElRef.current) audioElRef.current.src='';
                            setAudioSrc(null); setAudioName('');
                          }}>✕</button>
                      </div>
                      <p style={{fontSize:9,color:'var(--muted)',marginTop:4,lineHeight:1.5}}>
                        Audio will be mixed into your recording. Loops automatically.
                      </p>
                    </div>
                  )}

                  {/* ── Text Overlay ── */}
                  <div className="sec-title" style={{marginTop:16,marginBottom:6}}>Text Overlay</div>
                  <button className="btn btn-ghost" style={{width:'100%',justifyContent:'center',fontSize:11,marginBottom:8}}
                    onClick={addText}>+ Add Text</button>
                  {textItems.map(item=>(
                    <div key={item.id} style={{marginBottom:4,padding:'6px 8px',borderRadius:6,
                      background:selTextId===item.id?'var(--surface)':'transparent',
                      border:`1px solid ${selTextId===item.id?'var(--accent)':'var(--border)'}`,cursor:'pointer'}}
                      onClick={()=>setSelTextId(selTextId===item.id?null:item.id)}>
                      <div style={{fontSize:11,color:'var(--text)',overflow:'hidden',textOverflow:'ellipsis',whiteSpace:'nowrap'}}>
                        {item.text||'(empty)'}
                      </div>
                      {selTextId===item.id&&(
                        <div onClick={e=>e.stopPropagation()} style={{marginTop:6}}>
                          <textarea value={item.text} placeholder="Enter text… (Enter = new line)"
                            rows={3}
                            onChange={e=>updateText(item.id,'text',e.target.value)}
                            style={{width:'100%',marginBottom:5,fontSize:12,padding:'4px 6px',boxSizing:'border-box',
                              background:'var(--bg)',border:'1px solid var(--border)',color:'var(--text)',borderRadius:4,
                              resize:'none',fontFamily:'inherit',lineHeight:1.5}}/>
                          <div style={{marginBottom:7}}>
                            <div style={{fontSize:9,fontWeight:700,letterSpacing:'.8px',textTransform:'uppercase',color:'var(--muted)',marginBottom:4}}>Emojis — click to insert</div>
                            <div style={{display:'flex',flexWrap:'wrap',gap:2,background:'var(--bg)',border:'1px solid var(--border)',borderRadius:5,padding:'5px'}}>
                              {['😀','😂','😍','😎','🥰','😭','🤩','😤','🥳','🔥','✨','💯','👏','🙌','👍','❤️','💕','🎉','🚀','⭐','💪','🏆','💎','⚡','🎯','💡','📱','💻','🎬','🎵'].map(em=>(
                                <span key={em}
                                  onClick={()=>updateText(item.id,'text',item.text+em)}
                                  style={{fontSize:17,cursor:'pointer',padding:'2px 3px',borderRadius:4,lineHeight:1,
                                    transition:'transform .1s'}}
                                  onMouseEnter={e=>(e.currentTarget as HTMLElement).style.transform='scale(1.3)'}
                                  onMouseLeave={e=>(e.currentTarget as HTMLElement).style.transform='scale(1)'}>
                                  {em}
                                </span>
                              ))}
                            </div>
                          </div>
                          <div style={{display:'flex',gap:5,alignItems:'center',marginBottom:6}}>
                            <input type="number" value={item.size} min={8} max={400}
                              onChange={e=>updateText(item.id,'size',Math.max(8,+e.target.value))}
                              style={{width:56,fontSize:11,padding:'3px 4px',
                                background:'var(--bg)',border:'1px solid var(--border)',color:'var(--text)',borderRadius:4}}/>
                            <span style={{fontSize:9,color:'var(--muted)'}}>px</span>
                            <input type="color" value={item.color}
                              onChange={e=>updateText(item.id,'color',e.target.value)}
                              style={{width:26,height:22,border:'none',background:'none',cursor:'pointer',padding:0}}/>
                            <button className={`grade-btn${item.bold?' active':''}`}
                              onClick={()=>updateText(item.id,'bold',!item.bold)}
                              style={{padding:'2px 8px',fontWeight:'bold',fontSize:12,minWidth:28}}>B</button>
                            <button className={`grade-btn${item.italic?' active':''}`}
                              onClick={()=>updateText(item.id,'italic',!item.italic)}
                              style={{padding:'2px 8px',fontStyle:'italic',fontSize:12,minWidth:28}}>I</button>
                          </div>
                          <select value={item.family} onChange={e=>updateText(item.id,'family',e.target.value)}
                            style={{width:'100%',marginBottom:6,fontSize:11,padding:'3px 6px',
                              background:'var(--bg)',border:'1px solid var(--border)',color:'var(--text)',borderRadius:4}}>
                            <optgroup label="── Reel / Bold ──">
                              <option value="'Bebas Neue', sans-serif">Bebas Neue</option>
                              <option value="'Anton', sans-serif">Anton</option>
                              <option value="'Bungee', sans-serif">Bungee</option>
                              <option value="'Bangers', cursive">Bangers</option>
                              <option value="'Racing Sans One', sans-serif">Racing Sans One</option>
                              <option value="'Oswald', sans-serif">Oswald Heavy</option>
                              <option value="'Montserrat', sans-serif">Montserrat Black</option>
                              <option value="'Black Han Sans', sans-serif">Black Han Sans</option>
                            </optgroup>
                            <optgroup label="── Handwriting ──">
                              <option value="'Permanent Marker', cursive">Permanent Marker</option>
                              <option value="'Pacifico', cursive">Pacifico</option>
                              <option value="'Satisfy', cursive">Satisfy</option>
                              <option value="'Righteous', cursive">Righteous</option>
                            </optgroup>
                            <optgroup label="── Elegant ──">
                              <option value="'Cinzel', serif">Cinzel</option>
                              <option value="Georgia, serif">Georgia</option>
                            </optgroup>
                            <optgroup label="── Clean ──">
                              <option value="'Inter', sans-serif">Inter</option>
                              <option value="'Roboto', sans-serif">Roboto</option>
                              <option value="'Arial Black', sans-serif">Arial Black</option>
                            </optgroup>
                          </select>
                          <button className="btn btn-ghost"
                            style={{width:'100%',justifyContent:'center',fontSize:10.5,color:'#e05050'}}
                            onClick={()=>removeText(item.id)}>Remove</button>
                        </div>
                      )}
                    </div>
                  ))}
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

          <main className="canvas-area"
            onMouseMove={e=>{
              const r=(e.currentTarget as HTMLElement).getBoundingClientRect();
              mouseXRef.current=(e.clientX-r.left)/r.width;
            }}
            onWheel={onAreaWheel}>
            <div style={{display:'flex',flexDirection:'column',alignItems:'center',
              transform:`translate(${pan.x}px,${pan.y}px)`,transition:'none'}}>
            <div className="canvas-wrap" style={{width:csz.w,height:csz.h,
              transform:`scale(${zoom})`,transformOrigin:'center top',
              cursor:activePin!==null?'grabbing':'grab'}}
              onPointerDown={onCanvasDown}
              onPointerMove={onMove} onPointerUp={onUp} onPointerLeave={onUp}>
              <canvas ref={canvasRef} width={native.w} height={native.h}
                style={{display:'block',width:csz.w,height:csz.h}}/>
              {mode==='manual'&&mockupSrc&&(
                <svg style={{position:'absolute',inset:0,pointerEvents:'none'}} width={csz.w} height={csz.h}>
                  {/* Thin outline of the mapped screen rect */}
                  <polyline points={[...pins,pins[0]].map(p=>`${p.x},${p.y}`).join(' ')}
                    fill="none" stroke="rgba(255,255,255,.18)" strokeWidth="1"/>
                </svg>
              )}
              {mode==='manual'&&mockupSrc&&pins.map((p,i)=>{
                // L-bracket sits OUTSIDE the screen corner — never covers the actual edge point
                // Each bracket extends 12px along the two edges meeting at this corner, offset 1px outward
                const O=1; // outward offset in px
                const A=12; // arm length
                const offsets=[
                  {left:p.x-O-A, top:p.y-O-A}, // TL: bracket opens bottom-right
                  {left:p.x+O,   top:p.y-O-A}, // TR: bracket opens bottom-left
                  {left:p.x+O,   top:p.y+O},   // BR: bracket opens top-left
                  {left:p.x-O-A, top:p.y+O},   // BL: bracket opens top-right
                ];
                // Arm positions inside the 20×20 drag zone for each corner
                const arms=[
                  [{bottom:0,right:0},{bottom:0,right:0}],  // TL: H at bottom-right, V at bottom-right
                  [{bottom:0,left:0},{bottom:0,left:0}],    // TR
                  [{top:0,left:0},{top:0,left:0}],          // BR
                  [{top:0,right:0},{top:0,right:0}],        // BL
                ];
                const hStyle=[
                  {bottom:0,right:0,width:A,height:2},
                  {bottom:0,left:0,width:A,height:2},
                  {top:0,left:0,width:A,height:2},
                  {top:0,right:0,width:A,height:2},
                ];
                const vStyle=[
                  {bottom:0,right:0,width:2,height:A},
                  {bottom:0,left:0,width:2,height:A},
                  {top:0,left:0,width:2,height:A},
                  {top:0,right:0,width:2,height:A},
                ];
                return(
                  <div key={i} className={`pin-corner${activePin===i?' active':''}`}
                    style={{...offsets[i]}} onPointerDown={onPinDown(i)}>
                    <div className="arm-h" style={hStyle[i]}/>
                    <div className="arm-v" style={vStyle[i]}/>
                  </div>
                );
              })}
              {mockupSrc&&textItems.map(item=>(
                <div key={item.id}
                  style={{position:'absolute',left:item.x*csz.w,top:item.y*csz.h,
                    fontSize:item.size,fontFamily:item.family,
                    fontWeight:item.bold?'bold':'normal',fontStyle:item.italic?'italic':'normal',
                    color:item.color,cursor:'move',userSelect:'none',pointerEvents:'all',
                    outline:selTextId===item.id?'1px dashed var(--accent)':'1px dashed transparent',
                    padding:'2px 4px',whiteSpace:'pre',
                    textShadow:'0 2px 8px rgba(0,0,0,0.6)',lineHeight:1.25}}
                  onPointerDown={e=>{
                    e.stopPropagation();
                    setSelTextId(item.id);
                    const rect=canvasRef.current!.getBoundingClientRect();
                    const z=zoomRef.current;
                    textDragRef.current={id:item.id,sx:(e.clientX-rect.left)/z,sy:(e.clientY-rect.top)/z,ox:item.x,oy:item.y};
                    (e.target as HTMLElement).setPointerCapture(e.pointerId);
                  }}>
                  {item.text||'✦ Text'}
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
            {/* Glass-table reflection — only when mockup loaded */}
            {mockupSrc&&(
              <canvas ref={reflRef}
                width={native.w} height={Math.round(native.h*0.26)}
                style={{
                  display:'block',
                  width:csz.w,
                  height:Math.round(csz.h*0.26),
                  opacity:0.7,
                  pointerEvents:'none',
                  marginTop:0,
                  transform:`scale(${zoom})`,transformOrigin:'center top',
                  filter:'blur(1.5px) brightness(1.2)',
                  mixBlendMode:'screen' as React.CSSProperties['mixBlendMode'],
                }}/>
            )}
            </div>
            {/* Zoom pill */}
            <div className="zoom-pill">
              <button className="zoom-btn" onClick={()=>setZoom(z=>Math.max(0.2,z*0.8))}>−</button>
              <span className="zoom-pct" title="Click to reset" onClick={fitZoom}>{Math.round(zoom*100)}%</span>
              <button className="zoom-btn" onClick={()=>setZoom(z=>Math.min(4,z*1.25))}>+</button>
              <div style={{width:1,height:14,background:'var(--border)',margin:'0 3px'}}/>
              <button className="zoom-btn" style={{fontSize:10,padding:'2px 6px'}} onClick={fitZoom} title="Fit to screen">⊡</button>
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
              <div className="m-stat"><span>Resolution</span><strong>
                {exportRatio==='9:16'?'1080 × 1920':exportRatio==='1:1'?'1080 × 1080':`${native.w} × ${native.h}`}
              </strong></div>
              <div className="m-stat"><span>Grade</span><strong>{grade==='none'?'None':grade.charAt(0).toUpperCase()+grade.slice(1)}</strong></div>
              {recDur>0&&<div className="m-stat"><span>Clip</span><strong>{fmt(Math.round(trimIn*recDur))} – {fmt(Math.round(trimOut*recDur))}</strong></div>}
              <div className="m-lbl">Format</div>
              <div className="fmt-tabs">
                <div className={`fmt-tab${expFmt==='png'?' active':''}`} onClick={()=>setExpFmt('png')}>📷 PNG Still</div>
                <div className={`fmt-tab${expFmt==='webm'?' active':''}`} onClick={()=>setExpFmt('webm')}>🎬 WebM Video</div>
              </div>
              {expFmt==='webm'&&<>
                <div className="m-lbl">Aspect Ratio</div>
                <div className="q-row">
                  {(['16:9','9:16','1:1'] as ExportRatio[]).map(r=>(
                    <div key={r} className={`q-btn${exportRatio===r?' active':''}`} onClick={()=>setExportRatio(r)}
                      style={{flexDirection:'column',alignItems:'center'}}>
                      <span style={{fontSize:12,fontWeight:700}}>{r}</span>
                      <span style={{fontSize:8,opacity:.6,marginTop:2}}>
                        {r==='16:9'?'LinkedIn · Twitter':r==='9:16'?'Reels · TikTok · Shorts':'Instagram'}
                      </span>
                    </div>
                  ))}
                </div>
                {exportRatio!=='16:9'&&(
                  <div className="q-note" style={{color:'var(--accent)'}}>
                    ✦ Blur background auto-added — your 16:9 mockup centered in frame
                  </div>
                )}
                <div className="m-lbl">Quality</div>
                <div className="q-row">
                  <div className={`q-btn${quality==='high'?' active':''}`} onClick={()=>setQuality('high')}>
                    High<br/><span style={{fontSize:9,opacity:.6}}>40 Mbps</span>
                  </div>
                  <div className={`q-btn${quality==='ultra'?' active':''}`} onClick={()=>setQuality('ultra')}>
                    Ultra<br/><span style={{fontSize:9,opacity:.6}}>80 Mbps</span>
                  </div>
                </div>
                <div className="q-note">60 fps · grade baked in · starts from trim point</div>
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
